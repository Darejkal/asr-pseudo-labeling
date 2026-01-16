"""
CLI module for audio processing pipeline.
Uses pydanclick for CLI argument parsing with Pydantic models.
"""

try:
    from .main import main
    __all__ = ["main"]
except ImportError as e:
    # Allow graceful failure when CLI dependencies are not available
    import warnings
    warnings.warn(f"CLI module not available: {e}")
    __all__ = []
