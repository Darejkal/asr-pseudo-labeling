"""
CLI command implementations.
"""

import asyncio
import sys
import traceback
from pathlib import Path
from typing import Dict, Any

from pipeline.processing import load_models, MODELS
from pipeline.llm import get_offline_model

from .config import ProcessConfig, BatchConfig, LabelStudioConfig, EntityDatasetConfig
from .workers import (
    diarization_worker,
    asr_worker,
    file_processor_worker,
    results_writer_worker,
)
from .utils import generate_srt


def to_label_studio_command(config: LabelStudioConfig):
    """Convert JSON results to Label Studio import format."""
    import json
    
    if not config.input_dir.exists() or not config.input_dir.is_dir():
        print(f"Error: Input directory does not exist or is not a directory: {config.input_dir}", file=sys.stderr)
        sys.exit(1)
        
    json_files = list(config.input_dir.glob("*.json"))
    if not json_files:
        print(f"No JSON files found in {config.input_dir}", file=sys.stderr)
        sys.exit(1)
        
    tasks = []
    task_id_counter = 1
    for json_file in json_files:
        with open(json_file, "r", encoding="utf-8") as f:
            result_data = json.load(f)
            
        audio_filename = result_data.get("audio_file")
        if not audio_filename:
            print(f"Warning: 'audio_file' not found in {json_file.name}, skipping.", file=sys.stderr)
            continue

        audio_url = f"{config.audio_url_prefix or ''}/{audio_filename}"
        
        segments = result_data.get("segments", [])
        results = []
        for segment in segments:
            results.append({
                "value": {
                    "start": segment.get("start_time"),
                    "end": segment.get("end_time"),
                    "text": segment.get("text")
                },
                "from_name": "transcription",
                "to_name": "audio",
                "type": "textarea"
            })
        
        task = {
            "id": task_id_counter,
            "data": {
                "audio": audio_url
            },
            "predictions": [{
                "model_version": "app-cli-v1",
                "result": results
            }]
        }
        tasks.append(task)
        task_id_counter += 1
        
    with open(config.output_file, "w", encoding="utf-8") as f:
        json.dump(tasks, f, ensure_ascii=False, indent=2)
        
    print(f"Successfully converted {len(tasks)} files.")
    print(f"Label Studio tasks saved to: {config.output_file.absolute()}")


async def process_single_file_async(
    audio_path: Path,
    output_dir: Path,
    output_format: str,
    verbose: bool,
    num_asr_workers: int = 2,
) -> bool:
    """Process a single audio file asynchronously using TaskGroup with worker pattern."""
    from datetime import datetime
    import json
    from pipeline.processing import load_and_resample_audio, segments_to_srt_text, extract_entities
    
    try:
        if verbose:
            print(f"Processing: {audio_path.name}")
        
        # Load and prepare audio
        target_sr = 16000
        waveform, sr = await asyncio.to_thread(load_and_resample_audio, audio_path, target_sr)
        
        # Create queues for pipeline stages
        diarization_queue = asyncio.Queue()
        results_queue = asyncio.Queue()
        
        # Use TaskGroup to manage workers
        try:
            from asyncio import TaskGroup
        except:
            from taskgroup import TaskGroup
            
        async with TaskGroup() as tg:
            # Start diarization worker
            tg.create_task(diarization_worker(
                audio_path, waveform, sr, diarization_queue, verbose
            ))
            
            # Start multiple ASR workers for parallel transcription
            for worker_id in range(num_asr_workers):
                tg.create_task(asr_worker(
                    worker_id, diarization_queue, results_queue, verbose
                ))
        
        # All workers finished - collect results
        segments = []
        while not results_queue.empty():
            segments.append(await results_queue.get())
        
        # Sort segments by start time
        segments.sort(key=lambda x: x["start_time"])
        
        if verbose:
            print(f"  [Pipeline] Collected {len(segments)} transcribed segments")
        
        # Extract entities
        model, tokenizer = get_offline_model()
        srt_text = segments_to_srt_text(segments)
        entities = await asyncio.to_thread(extract_entities, 
            text=srt_text,
            mode="offline",
            model=model,
            tokenizer=tokenizer
        )
        
        # Generate output
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_filename = f"{audio_path.stem}_{timestamp}"
        
        result = {
            "audio_file": audio_path.name,
            "segments": segments,
            "entities": entities,
            "processed_at": timestamp
        }
        
        if output_format in ["json", "both"]:
            json_path = output_dir / f"{base_filename}.json"
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
            if verbose:
                print(f"  → Saved JSON: {json_path.name}")
        
        if output_format in ["srt", "both"]:
            srt_path = output_dir / f"{base_filename}.srt"
            with open(srt_path, "w", encoding="utf-8") as f:
                f.write(generate_srt(segments))
            if verbose:
                print(f"  → Saved SRT: {srt_path.name}")
        
        return True
    except Exception as e:
        print(f"Error processing {audio_path.name}: {e}", file=sys.stderr)
        if verbose:
            traceback.print_exc()
        return False


async def process_directory(
    input_dir: Path,
    output_dir: Path,
    output_format: str,
    verbose: bool,
    recursive: bool,
    max_concurrent: int,
    num_asr_workers: int = 2,
) -> Dict[str, Any]:
    """Process all .wav files in a directory using worker pool pattern."""
    
    # Find audio files
    if recursive:
        audio_files = list(input_dir.rglob("*.wav"))
    else:
        audio_files = list(input_dir.glob("*.wav"))
    
    if not audio_files:
        print(f"No .wav files found in {input_dir}")
        return {"total": 0, "successful": 0, "failed": 0}
    
    print(f"Found {len(audio_files)} .wav file(s) to process")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create queues
    file_queue = asyncio.Queue()
    results_queue = asyncio.Queue()
    
    # Populate file queue
    for audio_path in audio_files:
        await file_queue.put(audio_path)
    
    # Use TaskGroup to manage workers
    try:
        from asyncio import TaskGroup
    except:
        from taskgroup import TaskGroup
    
    stats = {"total": 0, "successful": 0, "failed": 0}
    
    async with TaskGroup() as tg:
        # Start file processor workers
        for worker_id in range(max_concurrent):
            tg.create_task(file_processor_worker(
                worker_id,
                file_queue,
                results_queue,
                num_asr_workers,
                verbose
            ))
        
        # Signal end of files
        await file_queue.put(None)
        
        # Start results writer worker
        writer_task = tg.create_task(results_writer_worker(
            results_queue,
            output_dir,
            output_format,
            len(audio_files),
            verbose
        ))
    
    # Get stats from writer
    stats = await writer_task if hasattr(writer_task, '__await__') else stats
    
    return stats


def process_command(config: ProcessConfig):
    """Execute single file processing command."""
    if not config.input.exists():
        print(f"Error: Input path does not exist: {config.input}", file=sys.stderr)
        sys.exit(1)
    
    if not config.input.is_file() or config.input.suffix.lower() != ".wav":
        print(f"Error: Input must be a .wav file: {config.input}", file=sys.stderr)
        sys.exit(1)
    
    print("Loading models... This may take a few minutes on first run.")
    try:
        load_models()
        print("✓ Models loaded successfully\n")
    except Exception as e:
        print(f"Error loading models: {e}", file=sys.stderr)
        traceback.print_exc()
        sys.exit(1)
    
    config.output.mkdir(parents=True, exist_ok=True)
    success = asyncio.run(process_single_file_async(
        config.input,
        config.output,
        config.format,
        config.verbose,
        config.num_asr_workers
    ))
    sys.exit(0 if success else 1)


def batch_command(config: BatchConfig):
    """Execute batch processing command."""
    if not config.input.exists():
        print(f"Error: Input path does not exist: {config.input}", file=sys.stderr)
        sys.exit(1)
    
    if not config.input.is_dir():
        print(f"Error: Input must be a directory: {config.input}", file=sys.stderr)
        sys.exit(1)
    
    print("Loading models... This may take a few minutes on first run.")
    try:
        load_models()
        print("✓ Models loaded successfully\n")
    except Exception as e:
        print(f"Error loading models: {e}", file=sys.stderr)
        traceback.print_exc()
        sys.exit(1)
    
    stats = asyncio.run(process_directory(
        config.input,
        config.output,
        config.format,
        config.verbose,
        config.recursive,
        config.max_concurrent,
        config.num_asr_workers,
    ))
    
    print(f"\n{'='*50}")
    print(f"Processing complete!")
    print(f"  Total files: {stats['total']}")
    print(f"  Successful: {stats['successful']}")
    print(f"  Failed: {stats['failed']}")
    print(f"  Output directory: {config.output.absolute()}")
    print(f"{'='*50}")
    
    sys.exit(0 if stats['failed'] == 0 else 1)


def extract_entity_dataset_command(config: EntityDatasetConfig):
    """Extract audio chunks for entities and save as HuggingFace dataset."""
    import json
    import numpy as np
    import torchaudio
    from datasets import Dataset, Audio, Features, Value
    from pipeline.processing import load_and_resample_audio
    
    if not config.input_dir.exists() or not config.input_dir.is_dir():
        print(f"Error: Input directory does not exist or is not a directory: {config.input_dir}", file=sys.stderr)
        sys.exit(1)
    
    if not config.audio_dir.exists() or not config.audio_dir.is_dir():
        print(f"Error: Audio directory does not exist or is not a directory: {config.audio_dir}", file=sys.stderr)
        sys.exit(1)
    
    json_files = list(config.input_dir.glob("*.json"))
    if not json_files:
        print(f"No JSON files found in {config.input_dir}", file=sys.stderr)
        sys.exit(1)
    
    if config.verbose:
        print(f"Found {len(json_files)} JSON files to process")
    
    # Collect all entity data
    dataset_rows = []
    target_sr = 16000
    
    for json_file in json_files:
        try:
            with open(json_file, "r", encoding="utf-8") as f:
                result_data = json.load(f)
            
            audio_filename = result_data.get("audio_file")
            if not audio_filename:
                if config.verbose:
                    print(f"Warning: 'audio_file' not found in {json_file.name}, skipping.", file=sys.stderr)
                continue
            
            # Find the audio file
            audio_path = config.audio_dir / audio_filename
            if not audio_path.exists():
                if config.verbose:
                    print(f"Warning: Audio file not found: {audio_path}, skipping.", file=sys.stderr)
                continue
            
            # Load the full audio file
            waveform, sr = load_and_resample_audio(audio_path, target_sr)
            if waveform.ndim > 1:
                waveform = waveform[0]  # Take first channel
            waveform_np = waveform.numpy()
            
            entities = result_data.get("entities", [])
            segments = result_data.get("segments", [])
            
            if not segments:
                if config.verbose:
                    print(f"No segments found in {json_file.name}, skipping.")
                continue
            
            # Process each segment and find entities that overlap
            segments_with_entities = 0
            for segment in segments:
                seg_start = segment.get("start_time")
                seg_end = segment.get("end_time")
                seg_text = segment.get("text", "")
                
                if seg_start is None or seg_end is None:
                    continue
                
                # Find all entities that overlap with this segment
                overlapping_entities = []
                for entity in entities:
                    entity_start = entity.get("start_time")
                    entity_end = entity.get("end_time")
                    
                    if entity_start is None or entity_end is None:
                        continue
                    
                    # Check if entity overlaps with segment
                    if entity_start < seg_end and entity_end > seg_start:
                        overlapping_entities.append({
                            "value": entity.get("value", ""),
                            "predicted_type": entity.get("type", "unknown")
                        })
                
                # Only keep segments that have entities
                if not overlapping_entities:
                    continue
                
                segments_with_entities += 1
                
                # Extract audio chunk for this segment
                start_sample = int(seg_start * target_sr)
                end_sample = int(seg_end * target_sr)
                
                # Ensure valid range
                start_sample = max(0, start_sample)
                end_sample = min(len(waveform_np), end_sample)
                
                if start_sample >= end_sample:
                    continue
                
                audio_chunk = waveform_np[start_sample:end_sample]
                
                # Create dataset row
                row = {
                    "audio": {
                        "array": audio_chunk,
                        "sampling_rate": target_sr
                    },
                    "text": seg_text,
                    "entity": json.dumps(overlapping_entities, ensure_ascii=False),
                    "start": float(seg_start),
                    "end": float(seg_end),
                    "original_audio_file": audio_filename
                }
                dataset_rows.append(row)
            
            if config.verbose:
                print(f"Processed {json_file.name}: extracted {segments_with_entities} segments with entities")
        
        except Exception as e:
            print(f"Error processing {json_file.name}: {e}", file=sys.stderr)
            if config.verbose:
                traceback.print_exc()
            continue
    
    if not dataset_rows:
        print("No entity data extracted. Exiting.", file=sys.stderr)
        sys.exit(1)
    
    print(f"Extracted {len(dataset_rows)} entity audio chunks from {len(json_files)} files")
    
    # Create HuggingFace dataset
    print("Creating HuggingFace dataset...")
    
    dataset = Dataset.from_list(dataset_rows).cast_column("audio",Audio(sampling_rate=target_sr))
    
    # Save as parquet
    print(f"Saving dataset to {config.output_file}...")
    config.output_file.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_parquet(str(config.output_file))
    
    print(f"✓ Successfully created dataset with {len(dataset)} samples")
    print(f"✓ Saved to: {config.output_file.absolute()}")

