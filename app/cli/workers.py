"""
Worker functions for async processing pipeline.
"""

import asyncio
import traceback
from pathlib import Path
from typing import Any, Dict, List
import sys

from pipeline.processing import (
    load_and_resample_audio,
    transcribe_chunk,
    segments_to_srt_text,
    extract_entities,
    MODELS,
)
from pipeline.llm import get_offline_model


async def diarization_worker(
    audio_path: Path,
    waveform: Any,
    sr: int,
    diarization_queue: asyncio.Queue,
    verbose: bool,
):
    """Worker that performs diarization and pushes segments to queue."""
    try:
        if verbose:
            print(f"  [Diarization] Starting for {audio_path.name}")
        
        audio_data = {"waveform": waveform, "sample_rate": sr}
        diarization_pipeline = MODELS["diarization"]
        diarization = await asyncio.to_thread(diarization_pipeline, audio_data)
        
        segment_count = 0
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            segment_waveform = waveform[:, int(turn.start * sr):int(turn.end * sr)]
            await diarization_queue.put({
                "waveform": segment_waveform,
                "speaker": speaker,
                "start_time": turn.start,
                "end_time": turn.end,
            })
            segment_count += 1
        
        if verbose:
            print(f"  [Diarization] Complete - {segment_count} segments found")
    except Exception as e:
        print(f"  [Diarization] Error: {e}", file=sys.stderr)
        traceback.print_exc()
    finally:
        # Signal completion to ASR workers
        await diarization_queue.put(None)


async def asr_worker(
    worker_id: int,
    diarization_queue: asyncio.Queue,
    results_queue: asyncio.Queue,
    verbose: bool,
):
    """Worker that consumes diarized segments and transcribes them."""
    processed = 0
    try:
        while True:
            segment = await diarization_queue.get()
            
            # None signals end of diarization
            if segment is None:
                # Re-queue None for other workers
                await diarization_queue.put(None)
                break
            
            try:
                # Transcribe the segment
                transcription = await asyncio.to_thread(
                    transcribe_chunk, 
                    segment["waveform"]
                )
                
                # Put transcribed segment in results queue
                await results_queue.put({
                    "speaker": segment["speaker"],
                    "start_time": segment["start_time"],
                    "end_time": segment["end_time"],
                    "text": transcription,
                })
                processed += 1
                
                if verbose:
                    print(f"  [ASR-{worker_id}] Transcribed segment {processed}: {transcription}")
                    
            except Exception as e:
                print(f"  [ASR-{worker_id}] Error transcribing segment: {e}", file=sys.stderr)
            finally:
                diarization_queue.task_done()
                
    except Exception as e:
        print(f"  [ASR-{worker_id}] Worker error: {e}", file=sys.stderr)
        traceback.print_exc()
    finally:
        if verbose:
            print(f"  [ASR-{worker_id}] Finished - processed {processed} segments")


async def file_processor_worker(
    worker_id: int,
    file_queue: asyncio.Queue,
    results_queue: asyncio.Queue,
    num_asr_workers: int,
    verbose: bool,
):
    """Worker that processes entire audio files from queue."""
    processed = 0
    
    try:
        while True:
            audio_path = await file_queue.get()
            
            # None signals no more files
            if audio_path is None:
                # Re-queue None for other workers
                await file_queue.put(None)
                break
            
            try:
                if verbose:
                    print(f"[Worker-{worker_id}] Processing: {audio_path.name}")
                
                # Load audio
                target_sr = 16000
                waveform, sr = await asyncio.to_thread(load_and_resample_audio, audio_path, target_sr)
                
                # Create queues for this file's processing
                diarization_queue = asyncio.Queue()
                segment_results_queue = asyncio.Queue()
                
                # Use TaskGroup to manage workers for this file
                try:
                    from asyncio import TaskGroup
                except:
                    from taskgroup import TaskGroup
                
                async with TaskGroup() as tg:
                    # Start diarization worker
                    tg.create_task(diarization_worker(
                        audio_path, waveform, sr, diarization_queue, verbose
                    ))
                    
                    # Start multiple ASR workers
                    for asr_worker_id in range(num_asr_workers):
                        tg.create_task(asr_worker(
                            asr_worker_id, diarization_queue, segment_results_queue, verbose
                        ))
                
                # Collect segments
                segments = []
                while not segment_results_queue.empty():
                    segments.append(await segment_results_queue.get())
                
                # Sort segments by start time
                segments.sort(key=lambda x: x["start_time"])
                
                if verbose:
                    print(f"[Worker-{worker_id}] Collected {len(segments)} segments from {audio_path.name}")
                
                # Extract entities
                model, tokenizer = get_offline_model()
                srt_text = segments_to_srt_text(segments)
                entities = await asyncio.to_thread(extract_entities, 
                    text=srt_text,
                    mode="offline",
                    model=model,
                    tokenizer=tokenizer
                )
                
                # Put completed file result in results queue
                await results_queue.put({
                    "success": True,
                    "audio_path": audio_path,
                    "segments": segments,
                    "entities": entities,
                })
                
                processed += 1
                if verbose:
                    print(f"[Worker-{worker_id}] ✓ Completed: {audio_path.name}")
                    
            except Exception as e:
                print(f"[Worker-{worker_id}] Error processing {audio_path.name}: {e}", file=sys.stderr)
                if verbose:
                    traceback.print_exc()
                
                # Record failure
                await results_queue.put({
                    "success": False,
                    "audio_path": audio_path,
                    "error": str(e),
                })
            finally:
                file_queue.task_done()
                
    except Exception as e:
        print(f"[Worker-{worker_id}] Worker error: {e}", file=sys.stderr)
        traceback.print_exc()
    finally:
        if verbose:
            print(f"[Worker-{worker_id}] Finished - processed {processed} files")


async def results_writer_worker(
    results_queue: asyncio.Queue,
    output_dir: Path,
    output_format: str,
    total_files: int,
    verbose: bool,
):
    """Worker that gathers results and writes output files."""
    from datetime import datetime
    import json
    from .utils import generate_srt
    
    written = 0
    successful = 0
    failed = 0
    
    try:
        while written < total_files:
            result = await results_queue.get()
            
            if result["success"]:
                audio_path = result["audio_path"]
                segments = result["segments"]
                entities = result["entities"]
                
                # Generate output files
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                base_filename = f"{audio_path.stem}_{timestamp}"
                
                file_result = {
                    "audio_file": audio_path.name,
                    "segments": segments,
                    "entities": entities,
                    "processed_at": timestamp
                }
                
                if output_format in ["json", "both"]:
                    json_path = output_dir / f"{base_filename}.json"
                    with open(json_path, "w", encoding="utf-8") as f:
                        json.dump(file_result, f, ensure_ascii=False, indent=2)
                    if verbose:
                        print(f"  [Writer] → Saved JSON: {json_path.name}")
                
                if output_format in ["srt", "both"]:
                    srt_path = output_dir / f"{base_filename}.srt"
                    with open(srt_path, "w", encoding="utf-8") as f:
                        f.write(generate_srt(segments))
                    if verbose:
                        print(f"  [Writer] → Saved SRT: {srt_path.name}")
                
                successful += 1
            else:
                audio_path = result["audio_path"]
                if verbose:
                    print(f"  [Writer] ✗ Failed: {audio_path.name} - {result.get('error', 'Unknown error')}")
                failed += 1
            
            written += 1
            results_queue.task_done()
            
    except Exception as e:
        print(f"[Writer] Error: {e}", file=sys.stderr)
        traceback.print_exc()
    finally:
        if verbose:
            print(f"[Writer] Finished - {successful} successful, {failed} failed")
    
    return {"total": total_files, "successful": successful, "failed": failed}
