import uuid
import shutil
from pathlib import Path
from contextlib import asynccontextmanager
from fastapi import FastAPI, UploadFile, File, BackgroundTasks, HTTPException
from pydantic import BaseModel, Field
from typing import List, Dict, Any
import traceback

# Import the pipeline functions
from pipeline.processing import run_pipeline, load_models

# --- Configuration ---
UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)


# --- Lifespan Management (Model Loading) ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Load the ML models
    print("Application startup: Loading models...")
    load_models()
    yield
    # Shutdown: Clean up resources if needed
    print("Application shutdown.")


# --- FastAPI App Initialization ---
app = FastAPI(
    title="Audio Processing Pipeline API",
    description="An API to process audio files for transcription, diarization, and entity recognition.",
    version="0.2.0", # Version bump for lifespan feature
    lifespan=lifespan
)


# --- Pydantic Models ---
class Job(BaseModel):
    """Represents a processing job."""
    job_id: str
    status: str = "processing"
    details: str = "File uploaded and processing started."

class DiarizationSegment(BaseModel):
    """Represents a single segment of speech with a speaker tag."""
    speaker: str
    start_time: float
    end_time: float
    text: str


class EntityResult(BaseModel):
    """Represents a single named entity found in the text."""
    type: str
    value: str


class ProcessingResult(BaseModel):
    """Represents the final result of the audio processing."""
    job_id: str
    original_filename: str
    language: str = "vietnamese"
    segments: List[DiarizationSegment]
    entities: List[EntityResult] = Field(default_factory=list)
    srt_transcript: str = ""


# --- In-Memory "Database" ---
jobs: Dict[str, Job] = {}
results: Dict[str, ProcessingResult] = {}


# --- Helper Functions ---
def format_srt_time(seconds: float) -> str:
    """Converts a float in seconds to an SRT time string HH:MM:SS,ms."""
    millis = int((seconds - int(seconds)) * 1000)
    seconds = int(seconds)
    minutes, seconds = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d},{millis:03d}"

def to_srt(segments: List[Dict[str, Any]]) -> str:
    """Converts a list of segment dicts to an SRT formatted string."""
    srt_string = ""
    for i, segment in enumerate(segments, 1):
        start_time = format_srt_time(segment['start_time'])
        end_time = format_srt_time(segment['end_time'])
        srt_string += f"{i}\n"
        srt_string += f"{start_time} --> {end_time}\n"
        srt_string += f"{segment['text']}\n\n"
    return srt_string

def process_audio_task(job_id: str, file_path: Path, original_filename: str):
    """
    A wrapper function to run the pipeline and store results.
    This function is executed in the background.
    """
    try:
        processed_data = run_pipeline(file_path)
        segments = processed_data.get("segments", [])
        
        # Generate SRT transcript
        srt_transcript = to_srt(segments)

        results[job_id] = ProcessingResult(
            job_id=job_id,
            original_filename=original_filename,
            segments=segments,
            entities=processed_data.get("entities", []),
            srt_transcript=srt_transcript
        )
        jobs[job_id].status = "completed"
        jobs[job_id].details = "Processing successfully completed."

    except Exception as e:
        print("An error occurred during audio processing:")
        traceback.print_exc()
        jobs[job_id].status = "failed"
        jobs[job_id].details = f"An error occurred: {str(e)}"


# --- API Endpoints ---
@app.post("/process-audio", response_model=Job, status_code=202)
async def create_processing_job(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(..., description="Audio file to process (e.g., WAV, MP3).")
):
    """
    Upload an audio file to start a processing job.
    """
    job_id = str(uuid.uuid4())
    file_path = UPLOAD_DIR / f"{job_id}_{file.filename}"

    with file_path.open("wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    job = Job(job_id=job_id)
    jobs[job_id] = job

    background_tasks.add_task(process_audio_task, job_id, file_path, file.filename)

    return job

@app.get("/jobs/{job_id}", response_model=Job)
def get_job_status(job_id: str):
    """
    Retrieve the status of a specific processing job.
    """
    job = jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found.")
    return job

@app.get("/results/{job_id}", response_model=ProcessingResult)
def get_job_result(job_id: str):
    """
    Retrieve the results of a completed processing job.
    """
    job = jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found.")
    if job.status != "completed":
        raise HTTPException(status_code=400, detail=f"Job status is '{job.status}', not 'completed'.")

    result = results.get(job_id)
    if not result:
        raise HTTPException(status_code=404, detail="Result not found, though job was marked completed.")

    return result

@app.get("/", include_in_schema=False)
def root():
    return {"message": "Welcome to the Audio Processing API. Visit /docs for documentation."}