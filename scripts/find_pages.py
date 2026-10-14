"""Find every page whose extracted text matches a pattern, with context.

How Set D's answer pages were verified: a test question must list EVERY page
that states its answer (spec §5.4), so search the text, read each hit, and
list the pages that really say it.

    .venv/bin/python scripts/find_pages.py gpo 'net income' 100
    .venv/bin/python scripts/find_pages.py all '2\\.5 percent'

The first argument is a doc_id prefix or `all`; the pattern is a
case-insensitive regex over the clean page text with whitespace collapsed.
"""
from __future__ import annotations

import re
import sys

from rag_studio import paths
from rag_studio.parse import parse_pdf


def main(argv: list[str]) -> int:
    if len(argv) < 2:
        print(__doc__)
        return 1
    doc_arg, pattern = argv[0], argv[1]
    width = int(argv[2]) if len(argv) > 2 else 120
    rx = re.compile(pattern, re.I | re.S)
    for path in sorted(paths.PDFS.glob("*.pdf")):
        if doc_arg != "all" and not path.stem.startswith(doc_arg):
            continue
        for page in parse_pdf(path).pages:
            text = " ".join(page.text.split())
            for m in rx.finditer(text):
                a, b = max(0, m.start() - width), min(len(text), m.end() + width)
                flag = " (near-empty, not indexed)" if page.is_near_empty else ""
                print(f"{path.stem} p{page.page}{flag}: ...{text[a:b]}...")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
