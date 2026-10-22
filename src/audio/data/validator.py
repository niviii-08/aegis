"""Audio input validation utilities."""
from pathlib import Path

def validate_audio_file(file_path: Path) -> bool:
    allowed = {'.wav', '.flac', '.mp3', '.ogg', '.npy'}
    return file_path.suffix.lower() in allowed
