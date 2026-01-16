"""
Utility functions for CLI module.
"""

from typing import List, Dict, Any


def generate_srt(segments: List[Dict[str, Any]]) -> str:
    """
    Generate SRT subtitle format from segments.
    
    Args:
        segments: List of segment dictionaries with speaker, start_time, end_time, text
        
    Returns:
        SRT formatted string
    """
    srt_lines = []
    for idx, segment in enumerate(segments, 1):
        start_time = segment["start_time"]
        end_time = segment["end_time"]
        text = segment["text"]
        
        # Convert seconds to SRT time format (HH:MM:SS,mmm)
        start_h = int(start_time // 3600)
        start_m = int((start_time % 3600) // 60)
        start_s = int(start_time % 60)
        start_ms = int((start_time % 1) * 1000)
        
        end_h = int(end_time // 3600)
        end_m = int((end_time % 3600) // 60)
        end_s = int(end_time % 60)
        end_ms = int((end_time % 1) * 1000)
        
        srt_lines.append(f"{idx}")
        srt_lines.append(
            f"{start_h:02d}:{start_m:02d}:{start_s:02d},{start_ms:03d} --> "
            f"{end_h:02d}:{end_m:02d}:{end_s:02d},{end_ms:03d}"
        )
        srt_lines.append(text)
        srt_lines.append("")  # Empty line between entries
    
    return "\n".join(srt_lines)
