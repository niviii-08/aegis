"""Tests for audio validator."""
from pathlib import Path
from src.audio.data.validator import validate_audio_file

def test_audio_validator():
    assert validate_audio_file(Path('sample.flac')) is True
    assert validate_audio_file(Path('sample.wav')) is True
    assert validate_audio_file(Path('sample.npy')) is True
    assert validate_audio_file(Path('sample.exe')) is False
