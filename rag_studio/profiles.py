"""Index profiles: one embedding setup, bound to one index.

A profile fixes everything that decides what a vector means: the parser, the
cleaner and the chunker that made the passage text; whether a context line was
added; and the embedding model, its tag, quantization, dimensions and the exact
text put in front of passages and questions. Two vectors are comparable only if
all of these match, so the profile's manifest records every one of them, and a
query embedded any other way is refused (rule 3 in CLAUDE.md).

Profiles so far: the baseline (ladder row 1); two M2 parsing profiles (ladder
row 2), which change only the parser and cleaner; the M3 chunkers (ladder
row 3), which change only how the cleaned text is cut and labelled; and the
M4 embedding models (ladder row 5), which change only the model, on both
structure-aware bases.

**Where a profile runs is policy, not configuration.** `build_on` and
`query_on` say which machine may embed (this computer, the Colab runtime, or
either for a model both have). They are not part of the recorded config: the
same model and tag give the same vectors on either machine (the vector match
check proves it), and the build's manifest records where it actually ran.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, replace

from rag_studio import chunk, clean, parse


WHERE = ("local", "colab", "either")


@dataclass(frozen=True)
class EmbeddingSpec:
    model: str              # Ollama tag, or a Hugging Face repo for sentence-transformers
    dimensions: int
    doc_prefix: str         # put in front of every passage before embedding
    query_prefix: str       # put in front of every question
    backend: str = "ollama"  # "ollama" or "sentence-transformers" (in this process)
    batch_size: int = 32
    revision: str | None = None   # the Hugging Face commit (sentence-transformers only)
    dtype: str | None = None      # sentence-transformers only; never float16 (NaN, research §4)
    # Ollama options sent with every embed call, building and asking alike,
    # and recorded like any parameter (rule 2). num_ctx does not change a
    # vector (identical on a fresh cache, same order, 2026-10-15); it is there
    # for memory.
    options: dict | None = None
    build_on: str = "local"       # where passages may be embedded: local | colab | either
    query_on: str = "local"       # where questions are embedded: local | colab

    def __post_init__(self) -> None:
        if self.build_on not in WHERE or self.query_on not in ("local", "colab"):
            raise ValueError(f"{self.model}: build_on is one of {WHERE}, query_on local or colab.")

    def describe(self) -> dict:
        """What decides a vector's meaning, for the recorded config. Fields
        added after a build existed appear only when set, so every earlier
        build's config stays exactly as it was recorded."""
        d = asdict(self)
        for key in ("build_on", "query_on"):
            del d[key]
        for key in ("revision", "dtype", "options"):
            if d[key] is None:
                del d[key]
        return d

    def builds_on(self, where: str) -> bool:
        return self.build_on in (where, "either")


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
        d["embedding"] = self.embedding.describe()
        d["parser_version"] = parser_version(self.parser)
        if self.contextual:
            # Only profiles with context lines record how they were written;
            # every other profile's recorded setup stays byte-identical.
            from rag_studio import context
            d["context"] = context.describe()
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

# Ladder row 5 (M4): one change from each row-3 base (3a, 3b), the embedding
# model. Formats from the model cards (research B §1-2, §4).
QWEN3_TASK = "Given a web search query, retrieve relevant passages that answer the query"
# Passages get no instruction; questions get the card's default retrieval
# task. No space after "Query:" (the reference code and the ST config agree;
# the card's TEI example disagrees with both).
# A 1,024-token context: the longest passage is 1,346 characters (about 350
# tokens). Ollama's default 4,096 holds 2.26 GB of RAM for the 0.6B (a
# 1.2 GB compute buffer); 1,024 holds 1.34 GB. Measured 2026-10-15, after the
# default pushed this laptop into swap during a build.
QWEN3_06B = EmbeddingSpec(
    model="qwen3-embedding:0.6b", dimensions=1024,
    doc_prefix="", query_prefix=f"Instruct: {QWEN3_TASK}\nQuery:",
    build_on="either", options={"num_ctx": 1024})
QWEN3_8B = replace(QWEN3_06B, model="qwen3-embedding:8b", dimensions=4096,
                   build_on="colab", query_on="colab")
# Not through Ollama: its embeddinggemma-2 tags need the MLX runner, which
# Linux does not have (research B §2). In this process instead, text only,
# float32 (float16 gives NaN or degraded vectors, per the card).
EMBEDDINGGEMMA2 = EmbeddingSpec(
    model="google/embeddinggemma-2", dimensions=768,
    doc_prefix="title: none | text: ", query_prefix="task: search result | query: ",
    backend="sentence-transformers", batch_size=8,
    revision="914f7f89142e33e77833254d9c9b90c3cef7303b", dtype="float32")

# RAM a build on this computer needs, in GB: the model as loaded (measured
# with `ollama ps` / peak RSS) plus about 1 GB for the build itself (cleaned
# documents, passages, vectors). A local build refuses to start below this.
BUILD_RAM_GB = {
    "nomic-embed-text:latest": 1.6,
    "qwen3-embedding:0.6b": 2.4,
    "google/embeddinggemma-2": 3.2,
}

# Measured embedding speed in passages per second, for the time a build will
# take. Missing until measured (shown as "-", never guessed).
EMBED_RATE = {
    ("nomic-embed-text:latest", "local"): 2.9,       # M1: 1,190 passages in 406 s
    ("qwen3-embedding:0.6b", "local"): 0.6,           # M4: 32 real passages in 52-58 s
    ("google/embeddinggemma-2", "local"): 1.35,       # M4: 1,214 passages in 896 s (a whole build)
    # On a Colab T4 through the free tunnel (2026-10-15, the gateway before
    # gzip): the tunnel, not the GPU, sets these.
    ("qwen3-embedding:0.6b", "colab"): 5.3,            # 1,214 in 227 s; 1,428 in 258 s
    ("qwen3-embedding:8b", "colab"): 1.3,              # 1,214 in 931 s; 1,428 in 1,069 s
}

MODEL_NAMES = {
    "nomic-embed-text:latest": "nomic-embed-text",
    QWEN3_06B.model: "Qwen3-Embedding 0.6B",
    QWEN3_8B.model: "Qwen3-Embedding 8B",
    EMBEDDINGGEMMA2.model: "EmbeddingGemma 2",
}


def _row5(base: IndexProfile, spec: EmbeddingSpec, slug: str, where: str) -> IndexProfile:
    name = MODEL_NAMES[spec.model]
    return replace(
        base, id=f"{base.id}-{slug}", label=f"{base.label.split(' + ')[0]} + structure, {name}",
        description=f"{base.label}, with every passage and question embedded by {name} "
                    f"({spec.dimensions} dimensions, {where}) instead of nomic-embed-text.",
        embedding=spec)


ROW5 = [_row5(base, spec, slug, where)
        for spec, slug, where in ((QWEN3_06B, "qwen3-0.6b", "on this computer or the Colab GPU"),
                                  (EMBEDDINGGEMMA2, "embeddinggemma2", "in this app, on the CPU"),
                                  (QWEN3_8B, "qwen3-8b", "built and queried on the Colab GPU"))
        for base in (PYMUPDF_STRUCTURE, DOCLING_STRUCTURE)]

# Ladder row 7 (M5): one change from a row-5 profile each, a context line
# written for every passage (by qwen3:8b on the Colab GPU) and embedded with it.
ROW7 = [replace(
    base, id=f"{base.id}-ctx", label=f"{base.label}, with context lines", contextual=True,
    description=f"{base.description} Each passage also gets one or two sentences that place it in its "
                "document, written on Remote AI and embedded with the passage. The answer model still "
                "sees only the passage.")
    for base in (next(p for p in ROW5 if p.id == f"{b}-structure-embeddinggemma2") for b in ("pymupdf", "docling"))]

PROFILES: dict[str, IndexProfile] = {p.id: p for p in (
    BASELINE, PYMUPDF_CLEAN, DOCLING_CLEAN,
    PYMUPDF_SNAP, PYMUPDF_STRUCTURE, DOCLING_STRUCTURE, PYMUPDF_SEMANTIC, *ROW5, *ROW7)}
# The app's default index since M5 (2026-10-16): row 5d on the ladder. `./rag
# ingest` still builds BASELINE unless told otherwise (cli.py), because on a
# wiped data/ this default would start a 866 s Docling conversion.
DEFAULT = "docling-structure-embeddinggemma2"


def build_command(profile_id: str) -> str:
    """The command that builds a profile. `./rag ingest` is only the baseline's
    quick path; every other profile is built with `./rag index build`."""
    return "./rag ingest" if profile_id == BASELINE.id else f"./rag index build {profile_id}"


def get(profile_id: str) -> IndexProfile:
    try:
        return PROFILES[profile_id]
    except KeyError:
        raise KeyError(f"No index profile named {profile_id!r}. Known: {', '.join(PROFILES)}.") from None
