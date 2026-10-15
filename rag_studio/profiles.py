"""Index profiles: one embedding setup, bound to one index.

A profile fixes everything that decides what a vector means: the parser, the
cleaner and the chunker that made the passage text; whether a context line was
added; and the embedding model, its tag, quantization, dimensions and the exact
text put in front of passages and questions. Two vectors are comparable only if
all of these match, so the profile's manifest records every one of them, and a
query embedded any other way is refused (rule 3 in CLAUDE.md).

Profiles so far: the baseline (ladder row 1); two M2 parsing profiles (ladder
row 2), which change only the parser and cleaner; and the M3 chunkers (ladder
row 3), which change only how the cleaned text is cut and labelled. M4 adds
the embedding models.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, replace

from rag_studio import chunk, clean, parse


@dataclass(frozen=True)
class EmbeddingSpec:
    model: str              # Ollama tag
    dimensions: int
    doc_prefix: str         # put in front of every passage before embedding
    query_prefix: str       # put in front of every question
    backend: str = "ollama"
    batch_size: int = 32


@dataclass(frozen=True)
class IndexProfile:
    id: str
    label: str
    description: str
    parser: str
    cleaner: str
    chunker: chunk.ChunkerSpec
    contextual: bool
    embedding: EmbeddingSpec
    metric: str = "cosine"  # LanceDB defaults to squared L2: always set it

    def config(self) -> dict:
        """The recorded configuration, minus what only a build knows (digest)."""
        d = asdict(self)
        d["chunker"] = self.chunker.describe()
        d["parser_version"] = parser_version(self.parser)
        return d


def parser_version(parser: str) -> str:
    if parser == "docling":
        from rag_studio import parse_docling
        return parse_docling.version()
    return parse.PARSER_VERSION


BASELINE = IndexProfile(
    id="baseline",
    label="Baseline",
    description="The hand-built pipeline: PyMuPDF, NFKC, fixed 1,000-character passages, nomic-embed-text, exact dense search.",
    parser=parse.PARSER_ID,
    cleaner=clean.CLEANER_ID,
    chunker=chunk.BASELINE,
    contextual=False,
    embedding=EmbeddingSpec(
        model="nomic-embed-text:latest",
        dimensions=768,
        # nomic-embed-text is trained with these task prefixes; without them
        # passages and questions land in different parts of the space.
        doc_prefix="search_document: ",
        query_prefix="search_query: ",
    ),
)

# Ladder row 2 (M2): one change from the baseline each, so the scorecard can
# say which helped. Everything after the parser and cleaner is the baseline's.
PYMUPDF_CLEAN = replace(
    BASELINE, id="pymupdf-clean", label="PyMuPDF + cleaning",
    description="PyMuPDF with the full cleaning order and the injection guard: hidden text dropped, "
                "\"Th\" repaired, headers and footers moved to metadata, hyphens rejoined. "
                "Baseline passages, embedding and search.",
    cleaner=clean.FULL_CLEANER_ID)
DOCLING_CLEAN = replace(
    PYMUPDF_CLEAN, id="docling-clean", label="Docling + cleaning",
    description="Docling's layout, reading order and tables, then the same cleaning and guard. "
                "Baseline passages, embedding and search.",
    parser="docling")

# Ladder row 3 (M3): one change from a row-2 profile each, the chunker.
PYMUPDF_SNAP = replace(
    PYMUPDF_CLEAN, id="pymupdf-snap", label="PyMuPDF + cleaning, word-boundary fix",
    description="PyMuPDF + cleaning, with the baseline's passages fixed so that none starts "
                "mid-word. Everything else as PyMuPDF + cleaning.",
    chunker=chunk.FIXED_SNAP)
PYMUPDF_STRUCTURE = replace(
    PYMUPDF_CLEAN, id="pymupdf-structure", label="PyMuPDF + structure-aware passages",
    description="PyMuPDF + cleaning, cut into whole paragraphs that never cross a section, "
                "each starting with its section label (document and section).",
    chunker=chunk.STRUCTURE)
DOCLING_STRUCTURE = replace(
    DOCLING_CLEAN, id="docling-structure", label="Docling + structure-aware passages",
    description="Docling + cleaning, cut into whole paragraphs and whole tables that never cross "
                "a section, each starting with its section label.",
    chunker=chunk.STRUCTURE)
PYMUPDF_SEMANTIC = replace(
    PYMUPDF_CLEAN, id="pymupdf-semantic", label="PyMuPDF + semantic passages",
    description="PyMuPDF + cleaning, cut inside each section where neighbouring sentences stop "
                "being similar (by embedding), each starting with its section label.",
    chunker=chunk.SEMANTIC)

PROFILES: dict[str, IndexProfile] = {p.id: p for p in (
    BASELINE, PYMUPDF_CLEAN, DOCLING_CLEAN,
    PYMUPDF_SNAP, PYMUPDF_STRUCTURE, DOCLING_STRUCTURE, PYMUPDF_SEMANTIC)}
DEFAULT = BASELINE.id


def get(profile_id: str) -> IndexProfile:
    try:
        return PROFILES[profile_id]
    except KeyError:
        raise KeyError(f"No index profile named {profile_id!r}. Known: {', '.join(PROFILES)}.") from None
