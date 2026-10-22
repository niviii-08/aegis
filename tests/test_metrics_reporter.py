"""Tests for report formatting."""
from src.common.metrics_reporter import format_markdown_table

def test_markdown_table():
    tbl = format_markdown_table({'auc': 0.98})
    assert '| auc | 0.9800 |' in tbl
