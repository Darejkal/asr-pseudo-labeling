#!/usr/bin/env python3
"""
Main CLI entry point using pydanclick.
"""

try:
    import click
    from pydanclick import from_pydantic
    PYDANCLICK_AVAILABLE = True
except ImportError:
    PYDANCLICK_AVAILABLE = False
    import warnings
    warnings.warn("pydanclick or click not available. CLI will use fallback mode.")

from .config import ProcessConfig, BatchConfig, LabelStudioConfig, EntityDatasetConfig
from .commands import process_command, batch_command, to_label_studio_command, extract_entity_dataset_command


if PYDANCLICK_AVAILABLE:
    @click.group()
    def cli():
        """Audio processing pipeline CLI tool."""
        pass

    @cli.command()
    @from_pydantic("config", ProcessConfig)
    def process(config: ProcessConfig):
        """Process a single audio file."""
        process_command(config)

    @cli.command()
    @from_pydantic("config", BatchConfig)
    def batch(config: BatchConfig):
        """Process all .wav files in a directory."""
        batch_command(config)

    @cli.command(name="to-label-studio")
    @from_pydantic("config", LabelStudioConfig)
    def to_label_studio(config: LabelStudioConfig):
        """Convert JSON results to Label Studio import format."""
        to_label_studio_command(config)

    @cli.command(name="extract-entity-dataset")
    @from_pydantic("config", EntityDatasetConfig)
    def extract_entity_dataset(config: EntityDatasetConfig):
        """Extract entity audio chunks and save as HuggingFace dataset."""
        extract_entity_dataset_command(config)

    def main():
        """Main entry point."""
        cli()

else:
    # Fallback to argparse when pydanclick is not available
    import argparse
    import sys
    from pathlib import Path
    
    def main():
        """Fallback main entry point using argparse."""
        parser = argparse.ArgumentParser(
            description="Process audio files through the speech-to-text pipeline",
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog="""
Examples:
  # Process a single file
  python -m app.cli process --input input.wav --output results/
  
  # Process all files in a directory
  python -m app.cli batch --input input_dir/ --output results/
  
  # Process recursively with SRT output and 4 concurrent jobs
  python -m app.cli batch --input input_dir/ --output results/ --format srt --recursive --max-concurrent 4
  
  # Convert results to Label Studio format
  python -m app.cli to-label-studio --input-dir results/ --output-file ls-tasks.json --audio-url-prefix http://my-server.com/audio
            """
        )
        
        subparsers = parser.add_subparsers(dest="command", help="Command to execute")
        
        # Process command (single file)
        process_parser = subparsers.add_parser("process", help="Process a single audio file")
        process_parser.add_argument("--input", type=Path, required=True, help="Path to the .wav file")
        process_parser.add_argument("--output", type=Path, default=Path("results"), help="Output directory")
        process_parser.add_argument("--format", choices=["json", "srt", "both"], default="json", help="Output format")
        process_parser.add_argument("--verbose", action="store_true", help="Print detailed progress")
        process_parser.add_argument("--num-asr-workers", type=int, default=2, help="Number of parallel ASR workers")
        
        # Batch command (directory)
        batch_parser = subparsers.add_parser("batch", help="Process all .wav files in a directory")
        batch_parser.add_argument("--input", type=Path, required=True, help="Path to directory containing .wav files")
        batch_parser.add_argument("--output", type=Path, default=Path("results"), help="Output directory")
        batch_parser.add_argument("--format", choices=["json", "srt", "both"], default="json", help="Output format")
        batch_parser.add_argument("--recursive", action="store_true", help="Search for .wav files recursively")
        batch_parser.add_argument("--verbose", action="store_true", help="Print detailed progress")
        batch_parser.add_argument("--max-concurrent", type=int, default=2, help="Max concurrent file processing jobs")
        batch_parser.add_argument("--num-asr-workers", type=int, default=2, help="Number of parallel ASR workers per file")

        # Label Studio command
        ls_parser = subparsers.add_parser("to-label-studio", help="Convert JSON results to Label Studio format")
        ls_parser.add_argument("--input-dir", type=Path, required=True, help="Directory with JSON results")
        ls_parser.add_argument("--output-file", type=Path, default=Path("label_studio_tasks.json"), help="Output file for Label Studio tasks")
        ls_parser.add_argument("--audio-url-prefix", type=str, help="Base URL for audio files")

        # Extract entity dataset command
        entity_parser = subparsers.add_parser("extract-entity-dataset", help="Extract entity audio chunks as HuggingFace dataset")
        entity_parser.add_argument("--input-dir", type=Path, required=True, help="Directory with JSON results containing entities")
        entity_parser.add_argument("--audio-dir", type=Path, required=True, help="Directory with original .wav audio files")
        entity_parser.add_argument("--output-file", type=Path, default=Path("entity_dataset.parquet"), help="Output parquet file for HF dataset")
        entity_parser.add_argument("--verbose", action="store_true", help="Print detailed progress")

        args = parser.parse_args()
        
        if not args.command:
            parser.print_help()
            sys.exit(1)
        
        if args.command == "process":
            config = ProcessConfig(
                input=args.input,
                output=args.output,
                format=args.format,
                verbose=args.verbose,
                num_asr_workers=args.num_asr_workers
            )
            process_command(config)
        
        elif args.command == "batch":
            config = BatchConfig(
                input=args.input,
                output=args.output,
                format=args.format,
                recursive=args.recursive,
                verbose=args.verbose,
                max_concurrent=args.max_concurrent,
                num_asr_workers=args.num_asr_workers
            )
            batch_command(config)

        elif args.command == "to-label-studio":
            config = LabelStudioConfig(
                input_dir=args.input_dir,
                output_file=args.output_file,
                audio_url_prefix=args.audio_url_prefix
            )
            to_label_studio_command(config)

        elif args.command == "extract-entity-dataset":
            config = EntityDatasetConfig(
                input_dir=args.input_dir,
                audio_dir=args.audio_dir,
                output_file=args.output_file,
                verbose=args.verbose
            )
            extract_entity_dataset_command(config)


if __name__ == "__main__":
    main()
