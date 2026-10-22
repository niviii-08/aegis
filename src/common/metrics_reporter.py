"""Markdown report generator for benchmark metrics."""
def format_markdown_table(metrics: dict[str, float]) -> str:
    lines = ['| Metric | Score |', '|---|---|']
    for k, v in metrics.items():
        lines.append(f'| {k} | {v:.4f} |')
    return '\n'.join(lines)
