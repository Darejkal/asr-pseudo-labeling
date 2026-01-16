"""
Pydantic configuration models for CLI arguments.
"""

from pathlib import Path
from typing import Literal, Optional
from pydantic import BaseModel, Field


class ProcessConfig(BaseModel):
    """Configuration for processing a single audio file.
    
    Attributes:
        input: Path to the audio file to process
        output: Output directory for results
        format: Output format (json, srt, or both)
        verbose: Enable verbose logging
        num_asr_workers: Number of parallel ASR workers
    """
    input: Path = Field(..., description="Path to the .wav file")
    output: Path = Field(default=Path("results"), description="Output directory")
    format: Literal["json", "srt", "both"] = Field(default="json", description="Output format")
    verbose: bool = Field(default=False, description="Print detailed progress")
    num_asr_workers: int = Field(default=2, ge=1, le=8, description="Number of parallel ASR workers")


class BatchConfig(BaseModel):
    """Configuration for batch processing audio files.
    
    Attributes:
        input: Path to directory containing audio files
        output: Output directory for results
        format: Output format (json, srt, or both)
        recursive: Search for .wav files recursively
        verbose: Enable verbose logging
        max_concurrent: Maximum number of concurrent file processing jobs
        num_asr_workers: Number of parallel ASR workers per file
    """
    input: Path = Field(..., description="Path to directory containing .wav files")
    output: Path = Field(default=Path("results"), description="Output directory")
    format: Literal["json", "srt", "both"] = Field(default="json", description="Output format")
    recursive: bool = Field(default=False, description="Search for .wav files recursively")
    verbose: bool = Field(default=False, description="Print detailed progress")
    max_concurrent: int = Field(default=2, ge=1, le=16, description="Max concurrent file processing jobs")
    num_asr_workers: int = Field(default=2, ge=1, le=8, description="Number of parallel ASR workers per file")


class LabelStudioConfig(BaseModel):
    """Configuration for converting results to Label Studio format.
    
    Attributes:
        input_dir: Directory containing the JSON results from the 'batch' command.
        output_file: Path to save the Label Studio JSON file.
        audio_url_prefix: The base URL where the audio files are hosted.
    """
    input_dir: Path = Field(..., description="Directory with JSON results")
    output_file: Path = Field(default=Path("label_studio_tasks.json"), description="Output file for Label Studio tasks")
    audio_url_prefix: Optional[str] = Field(default=None, description="Base URL for audio files")


class EntityDatasetConfig(BaseModel):
    """Configuration for extracting entity audio chunks as HF dataset.
    
    Attributes:
        input_dir: Directory containing the JSON results with entities
        audio_dir: Directory containing the original audio files
        output_file: Path to save the HF dataset (parquet format)
        verbose: Enable verbose logging
    """
    input_dir: Path = Field(..., description="Directory with JSON results containing entities")
    audio_dir: Path = Field(..., description="Directory with original .wav audio files")
    output_file: Path = Field(default=Path("entity_dataset.parquet"), description="Output parquet file for HF dataset")
    verbose: bool = Field(default=False, description="Print detailed progress")
