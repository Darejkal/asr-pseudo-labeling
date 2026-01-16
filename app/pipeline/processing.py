import torch
import torchaudio
from pathlib import Path
from typing import Dict, Any, List
import os
import traceback
import json
import re
from transformers import Wav2Vec2Processor
from .custom_w2v import CustomWav2Vec2ForCTC
from .llm import get_generator,GenerationConfig
from pyannote.audio import Pipeline
import openai
import json
import unicodedata
import re

def sanitize_json_string(raw_str):
    # 1. Remove non-printable control characters (except actual newlines/tabs)
    # This cleans up hidden bytes that cause "Unterminated string" errors
    clean_str = "".join(
        ch for ch in raw_str 
        if unicodedata.category(ch)[0] != "C" or ch in "\n\r\t"
    )
    
    # 2. Fix rogue newlines inside quotes
    # This looks for newlines that aren't following a comma or brace
    # (Note: This is a heuristic; use with caution if your data is very messy)
    clean_str = re.sub(r'(?<![,\{\[])\n(?![ \t]*["\}\]])', " ", clean_str)

    try:
        # Validate the fix
        parsed = json.loads(clean_str)
        return json.dumps(parsed, indent=2, ensure_ascii=False)
    except json.JSONDecodeError as e:
        return f"Sanitization failed: {e}"
# --- Model Storage ---
MODELS: Dict[str, Any] = {}

# Common Vietnamese filler words and particles that should not be extracted as entities
COMMON_FILLERS = {
    "ô kê", "ôkê", "ok", "okay",
    "vâng", "dạ", "ạ", "à", "ừ", "ừm",
    "rồi", "được", "xong",
    "em", "anh", "chị", "ông", "bà",
    "thế", "vậy", "nhé", "nhá", "nhỉ", "nhở",
    "tí", "chút", "một chút",
}

# Invalid entity types that indicate extraction errors
INVALID_TYPES = {"type", "text", "entity", "<unk>", "other", "unknown"}


def filter_entities(entities: List[Dict[str, str]], original_text: str) -> List[Dict[str, str]]:
    """
    Filter out noise entities that are not meaningful.
    
    Args:
        entities: Raw list of extracted entities
        original_text: The original text that was analyzed
        
    Returns:
        Filtered list of entities
    """
    filtered = []
    original_clean = original_text.strip().lower()
    
    for entity in entities:
        entity_value = entity.get("value", "").strip()
        entity_type = entity.get("type", "").strip().lower()
        
        if not entity_value or not entity_type:
            continue
            
        # Skip if entity is the full original text (duplicate)
        if entity_value.lower() == original_clean:
            continue
            
        # Skip common filler words
        if entity_value.lower() in COMMON_FILLERS:
            continue
            
        # Skip invalid types
        if entity_type in INVALID_TYPES:
            continue
            
        # Skip if entity is too similar to full text (>90% of original length)
        if len(entity_value) > 0.9 * len(original_text):
            continue
            
        filtered.append(entity)
    
    return filtered


def load_models():
    """
    Loads all machine learning models into the global MODELS dictionary.
    This function is called once at application startup.
    """
    print("Loading all models for the pipeline...")
    
    # 1. Speaker Diarization
    print("Loading Diarization model (pyannote)...")
    hf_token = os.getenv("HUGGINGFACE_ACCESS_TOKEN")
    if not hf_token:
        raise ValueError("HUGGINGFACE_ACCESS_TOKEN environment variable not set.")
        
    diarization_pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1",
        use_auth_token=hf_token
    )
    if torch.cuda.is_available():
        print("Moving diarization model to GPU...")
        diarization_pipeline.to(torch.device("cuda"))
    MODELS["diarization"] = diarization_pipeline

    # 2. Automatic Speech Recognition (ASR)
    asr_model_names = ["nguyenvulebinh/wav2vec2-large-vi-vlsp2020"]
    loaded_asr_models = []
    for name in asr_model_names:
        print(f"Loading ASR model: {name}...")
        loaded_asr_models.append({
            "name": name,
            "processor": Wav2Vec2Processor.from_pretrained(name),
            "model": CustomWav2Vec2ForCTC.from_pretrained(name)
        })
    MODELS["asr"] = loaded_asr_models
    
    print("All models loaded successfully.")

# --- Helper Functions ---

def load_and_resample_audio(file_path: Path, target_sr: int = 16000) -> tuple[torch.Tensor, int]:
    """Loads audio and resamples it to the target sample rate."""
    waveform, sr = torchaudio.load(file_path)
    if sr != target_sr:
        resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=target_sr)
        waveform = resampler(waveform)
    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)
    return waveform, target_sr

def transcribe_chunk(waveform: torch.Tensor) -> str:
    """Transcribes a single audio chunk using configured ASR models."""
    if waveform.ndim > 1:
        waveform = waveform.mean(dim=0)

    transcriptions = []
    for asr_model in MODELS.get("asr", []):
        input_values = asr_model["processor"](waveform, return_tensors="pt", sampling_rate=16000).input_values
        with torch.no_grad():
            logits = asr_model["model"](input_values).logits
        predicted_ids = torch.argmax(logits, dim=-1)
        transcription = asr_model["processor"].batch_decode(predicted_ids)[0]
        transcriptions.append(transcription)

    final_transcription = transcriptions[0] if transcriptions else ""
    return final_transcription.lower()

def segments_to_srt_text(segments: List[Dict[str, Any]]) -> str:
    """

    Convert segments to SRT-style timestamped text for entity extraction.
    
    Args:
        segments: List of segment dictionaries with speaker, start_time, end_time, text
        
    Returns:
        Formatted string with timestamps and text
    """
    srt_lines = []
    for idx, segment in enumerate(segments, 1):
        start_time = segment["start_time"]
        end_time = segment["end_time"]
        speaker = segment.get("speaker", "UNKNOWN")
        text = segment["text"]
        
        # Format: [00:00:01.04 - 00:00:02.31] SPEAKER_00: text
        start_h = int(start_time // 3600)
        start_m = int((start_time % 3600) // 60)
        start_s = start_time % 60
        
        end_h = int(end_time // 3600)
        end_m = int((end_time % 3600) // 60)
        end_s = end_time % 60
        
        timestamp = f"[{start_h:02d}:{start_m:02d}:{start_s:05.2f} - {end_h:02d}:{end_m:02d}:{end_s:05.2f}]"
        srt_lines.append(f"{timestamp} {speaker}: {text}")
    
    return "\n".join(srt_lines)

def parse_srt_timestamp(timestamp_str: str) -> tuple[float, float]:
    """
    Parse SRT timestamp format to get start and end times in seconds.
    
    Args:
        timestamp_str: String like '[00:00:01.04 - 00:00:02.31]'
        
    Returns:
        Tuple of (start_time, end_time) in seconds
    """
    try:
        # Remove brackets and split by ' - '
        timestamp_str = timestamp_str.strip('[]')
        start_str, end_str = timestamp_str.split(' - ')
        
        def time_to_seconds(time_str):
            parts = time_str.split(':')
            hours = int(parts[0])
            minutes = int(parts[1])
            seconds = float(parts[2])
            return hours * 3600 + minutes * 60 + seconds
        
        start_time = time_to_seconds(start_str)
        end_time = time_to_seconds(end_str)
        return start_time, end_time
    except Exception as e:
        print(f"Warning: Failed to parse timestamp '{timestamp_str}': {e}")
        return 0.0, 0.0

def extract_entities(text: str,mode="offline",tokenizer=None,model=None) -> List[Dict[str, Any]]:
    """
    Extracts named entities from text using an OpenAI-compatible API.
    
    Args:
        text: Input text to extract entities from (plain or SRT-formatted)
        
    Returns:
        List of entity dictionaries with 'type', 'value', and optionally 'start_time', 'end_time' keys
    """
    if not text.strip():
        return []
    
    prompt = (
        "You are an expert Linguistic Analyst for Vietnamese Call Center transcripts.\n"
        "Your task is to analyze a timestamped conversation between an Agent (Điện thoại viên) and a Customer (Khách hàng) to extract specific entities.\n\n"

        "### 1. CONTEXT & ROLE INFERENCE\n"
        "- **Scenario:** Inbound call to a Customer Service center (Banking, VETC, Logistics).\n"
        "- **Agent (ĐTV):** Uses polite markers ('dạ', 'vâng ạ', 'em hỗ trợ'), asks questions, or **repeats information to verify**.\n"
        "- **Customer (Khách):** Provides personal details, states problems.\n\n"

        "### 2. ENTITY TYPES\n"
        "1. **name**: Person's name (e.g., 'Hùng', 'Thắng'). **Constraint:** NO pronouns (anh, chị, em, mình) **WARNING** Be careful not filter out (tuấn anh, ngọc anh...).\n"
        "2. **id_number**: Tax ID, License Plate (Biển số), Account ID, Citizen ID.\n"
        "   - *Phonetic handling:* Interpret phonetic typos (e.g., 'hát' -> 'H', 'bê' -> 'B', 'vê tê xê' -> 'VETC').\n"
        "3. **organization names**: Company names, banks, brands.\n"

        "### 3. EXTRACTION LOGIC (CRITICAL - ECHO FILTER)\n"
        "- **GOAL:** Extract entities ONLY when they are provided as **NEW** information.\n"
        "- **IGNORE REPETITION:** If a speaker (usually the Agent) repeats an entity just to confirm/verify (e.g., 'Dạ biển số là 30H đúng không ạ?'), **DO NOT extract it**.\n"
        "- **EXCEPTION:** If the Agent provides *new* data from the system (e.g., 'Mã giao dịch của anh là XYZ'), then extract it.\n\n"

        "### 4. OUTPUT FORMAT\n"
        "- Return a valid JSON array. \n"
        "- Fields: `type`, `value` (normalized text), `timestamp`, `role` (inferred: 'customer' or 'agent').\n"
        "- If no entities, return [].\n\n"

        "### EXAMPLES:\n\n"

        "**Example 1: Basic Extraction (Customer giving clear info)**\n"
        "Input:\n"
        "[00:00:01] SPEAKER_00: tôi tên là hùng\n"
        "[00:00:03] SPEAKER_00: mã số thuế là không một hai ba bốn năm sáu\n"
        "Output:\n"
        "[\n"
        "  {\"type\": \"name\", \"value\": \"Hùng\", \"timestamp\": \"[00:00:01]\", \"role\": \"customer\"},\n"
        "  {\"type\": \"id_number\", \"value\": \"0123456\", \"timestamp\": \"[00:00:03]\", \"role\": \"customer\"}\n"
        "]\n\n"

        "**Example 2: The 'Echo' Filter (Agent repeating info - MUST IGNORE)**\n"
        "Input:\n"
        "[00:01:10] SPEAKER_01: dạ vâng ạ em nghe\n"
        "[00:01:12] SPEAKER_01: xe này thuộc công ty vận tải biển xanh biển số ba mươi hát chín chín tám tám\n"
        "[00:01:17] SPEAKER_00: dạ biển số là ba mươi hát chín chín tám tám đúng không ạ\n"
        "[00:01:19] SPEAKER_01: đúng rồi chị\n"
        "Output:\n"
        "[\n"
        "  {\"type\": \"organization\", \"value\": \"Công ty vận tải Biển Xanh\", \"timestamp\": \"[00:01:12]\", \"role\": \"customer\"},\n"
        "  {\"type\": \"id_number\", \"value\": \"30H-9988\", \"timestamp\": \"[00:01:12]\", \"role\": \"customer\"}\n"
        "]\n"
        "*(Note: SPEAKER_00's line at 00:01:17 was ignored because they were just repeating the license plate to verify)*\n\n"

        "**Example 3: Complex Context (Phonetic correction)**\n"
        "Input:\n"
        "[00:02:00] SPEAKER_00: em ơi kiểm tra giúp anh cái thẻ vê tê xê\n"
        "[00:02:05] SPEAKER_00: mã là e không một ức xì hai ba\n"
        "Output:\n"
        "[\n"
        "  {\"type\": \"organization\", \"value\": \"VETC\", \"timestamp\": \"[00:02:00]\", \"role\": \"customer\"},\n"
        "  {\"type\": \"id_number\", \"value\": \"E01X23\", \"timestamp\": \"[00:02:05]\", \"role\": \"customer\"}\n"
        "]\n\n"

        f"### Input Transcript:\n{text}\n\n"
        "### Output JSON:"
    )

    try:

        thinking_content, output_text=get_generator(GenerationConfig(
            mode=mode,
            max_new_tokens=4096,
            model=model,
            tokenizer=tokenizer
        ))(prompt)
        print("="*20)
        print(f"Input: {text}")
        print(f"Thinking: {thinking_content}")
        print(f"Raw Model Output: {output_text}")
        print("="*20)

        # ---------------------------------------------------------
        # PARSING LOGIC: Handle Markdown and extraction
        # ---------------------------------------------------------
        # Remove Markdown code blocks (```json ... ```)
        cleaned_output = re.sub(r"```json\s*", "", output_text, flags=re.IGNORECASE)
        cleaned_output = re.sub(r"```", "", cleaned_output).strip()
        cleaned_output=sanitize_json_string(cleaned_output)
        entities = []
        try:
            # Extract the first JSON-like array structure found in the response
            # This helps if the model adds chatty conversational text before/after the JSON
            match = re.search(r"\[.*\]", cleaned_output, re.DOTALL)
            if match:
                json_str = match.group(0)
                entities = json.loads(json_str)
            else:
                # Fallback: try parsing the whole cleaned string
                entities = json.loads(cleaned_output)
            
            # Validate structure
            if not isinstance(entities, list):
                print(f"Warning: Output was not a list. Received: {type(entities)}")
                entities = []
            
        except json.JSONDecodeError as e:
            print(f"JSON Parse Error: {e}")
            print(f"Failed string: {[cleaned_output]}")
            entities = []

        # Filter out noise entities
        # Ensure the entities passed to filter are dicts with 'value' keys
        valid_entities = [e for e in entities if isinstance(e, dict) and 'value' in e and 'type' in e]
        
        # Parse timestamps if present
        for entity in valid_entities:
            timestamp_str = entity.get('timestamp', '')
            if timestamp_str:
                start_time, end_time = parse_srt_timestamp(timestamp_str)
                entity['start_time'] = start_time
                entity['end_time'] = end_time
        
        filtered_entities = filter_entities(valid_entities, text)
        print(f"Extracted {len(valid_entities)} entities, filtered to {len(filtered_entities)}")
        
        return filtered_entities
        
    except Exception as e:
        print(f"Error during entity extraction: {e}")
        traceback.print_exc()
        return []

# --- Main Pipeline Function ---

def run_pipeline(file_path: Path) -> Dict[str, Any]:
    """
    Executes the full audio processing pipeline using pre-loaded models.
    
    Args:
        file_path: Path to the audio file to process
        
    Returns:
        Dictionary with 'segments' and 'entities' keys
    """
    print(f"Starting speaker diarization processing for: {file_path.name}...")
    target_sr = 16000

    diarization_pipeline = MODELS.get("diarization")
    if not diarization_pipeline or not MODELS.get("asr"):
        raise RuntimeError("Models are not loaded. The application might not have started correctly.")

    try:
        waveform, sr = load_and_resample_audio(file_path, target_sr)
        audio_data = {"waveform": waveform, "sample_rate": sr}
    except Exception as e:
        traceback.print_exc()
        raise RuntimeError(f"Failed to load or resample audio: {e}")

    try:
        print("Running diarization pipeline...")
        diarization = diarization_pipeline(audio_data)
    except Exception as e:
        print(f"Warning: Speaker diarization failed ({e}). Returning empty result.")
        traceback.print_exc()
        diarization = []

    all_segments = []
    all_entities = []
    # Iterate over the speaker segments
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        start_sample = int(turn.start * target_sr)
        end_sample = int(turn.end * target_sr)
        chunk = waveform[:, start_sample:end_sample]
        
        if chunk.numel() > 400:  # Avoid transcribing very short segments
            text = transcribe_chunk(chunk)
            if text:
                segment_start = round(turn.start, 2)
                segment_end = round(turn.end, 2)
                all_segments.append({
                    "speaker": speaker,
                    "start_time": segment_start,
                    "end_time": segment_end,
                    "text": text,
                })
    entities = extract_entities(
        text=segments_to_srt_text(all_segments)
    )
    all_entities.extend(entities)

    print(f"Finished speaker diarization processing for: {file_path.name}.")
    return {"segments": all_segments, "entities": all_entities}