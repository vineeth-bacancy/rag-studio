"""Clean: repair what extraction broke, keeping the raw text beside the result.

The baseline applies one rule, Unicode NFKC, which turns the ligature
characters PyMuPDF returns ("ﬁ" "ﬂ" "ﬀ") into plain letters. GEP alone has
1,507 of them, and a search for "financial" cannot match "ﬁnancial".

M2 adds the full order from the build spec §4: NFKC, then the broken "Th",
then running headers and footers (keeping the printed page number), dot
leaders, line-break hyphens, and whitespace last.
"""
from __future__ import annotations

import unicodedata

CLEANER_ID = "nfkc"


def clean(text: str) -> str:
    return unicodedata.normalize("NFKC", text)
