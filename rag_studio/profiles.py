"""Index profiles: one embedding setup, bound to one index.

A profile fixes everything that decides what a vector means: the parser, the
cleaner and the chunker that made the passage text; whether a context line was
added; and the embedding model, its tag, quantization, dimensions and the exact
text put in front of passages and questions. Two vectors are comparable only if
all of these match, so the profile's manifest records every one of them, and a
query embedded any other way is refused (rule 3 in CLAUDE.md).

M1 has one profile, the baseline. M3-M4 add the upgrades beside it.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

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
        d["parser_version"] = parse.PARSER_VERSION
        return d


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

PROFILES: dict[str, IndexProfile] = {p.id: p for p in (BASELINE,)}
DEFAULT = BASELINE.id


def get(profile_id: str) -> IndexProfile:
    try:
        return PROFILES[profile_id]
    except KeyError:
        raise KeyError(f"No index profile named {profile_id!r}. Known: {', '.join(PROFILES)}.") from None
