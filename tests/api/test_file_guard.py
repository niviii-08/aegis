"""Tests for temporary file guard."""
from pathlib import Path
from api.utils.file_guard import safe_temp_file

def test_safe_temp():
    p = Path('temp_test_file.txt')
    p.write_text('test')
    with safe_temp_file(p):
        assert p.exists()
    assert not p.exists()
