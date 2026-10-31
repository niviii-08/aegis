"""Verify all publication files on GitHub and local services."""

import json
import ssl
import urllib.request
from pathlib import Path

ctx = ssl.create_default_context()

print("=== 1. Verifying Public GitHub URLs on the Internet ===")
urls = [
    ("Main Repository", "https://github.com/niviii-08/aegis"),
    ("Research Paper MD", "https://raw.githubusercontent.com/niviii-08/aegis/main/RESEARCH_PAPER.md"),
    ("Publication Article MD", "https://raw.githubusercontent.com/niviii-08/aegis/main/docs/PUBLICATION_ARTICLE.md"),
    ("BibTeX Citation", "https://raw.githubusercontent.com/niviii-08/aegis/main/citation.bib"),
    ("Scholar HTML Page", "https://raw.githubusercontent.com/niviii-08/aegis/main/docs/index.html"),
    ("DOCX Manuscript", "https://raw.githubusercontent.com/niviii-08/aegis/main/AEGIS_Research_Paper.docx"),
]

for name, url in urls:
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    try:
        with urllib.request.urlopen(req, timeout=10, context=ctx) as resp:
            status = resp.status
            size = len(resp.read())
            print(f"  [OK {status}] {name:<22}: {size:,} bytes live at {url}")
    except Exception as exc:
        print(f"  [FAIL]  {name:<22}: {exc}")

print("\n=== 2. Verifying Google Scholar Meta Tags in docs/index.html ===")
with open("docs/index.html", "r", encoding="utf-8") as f:
    html = f.read()

scholar_tags = [
    "citation_title",
    "citation_author",
    "citation_publication_date",
    "citation_journal_title",
    "citation_pdf_url",
    "DC.title",
    "DC.creator",
    "DC.date",
]
for tag in scholar_tags:
    if tag in html:
        print(f'  [FOUND] <meta name="{tag}">')
    else:
        print(f'  [MISSING] <meta name="{tag}">')

print("\n=== 3. Verifying Local System Services ===")
try:
    with urllib.request.urlopen("http://127.0.0.1:8000/health", timeout=5) as resp:
        data = json.loads(resp.read().decode())
        print("  Backend Health (Port 8000):", data)
except Exception as exc:
    print("  Backend check error:", exc)

try:
    with urllib.request.urlopen("http://127.0.0.1:5173/", timeout=5) as resp:
        print(f"  Frontend (Port 5173): {resp.status} OK")
except Exception as exc:
    print("  Frontend check error:", exc)
