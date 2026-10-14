"""Embed: turn passages and questions into vectors.

An `Embedder` is built from an index profile's embedding spec plus the model
digest the index was built with. It refuses to embed with anything else: if
the installed model's digest differs from the manifest's (a re-pull changed
the weights), the vectors would no longer be comparable and every search
would quietly get worse. So the mismatch is an error, never a fallback.
"""
from __future__ import annotations

import math
import time
from typing import Callable

from rag_studio.llm import OllamaClient, OllamaError
from rag_studio.profiles import EmbeddingSpec


RECHECK_S = 60.0


class ProfileMismatch(RuntimeError):
    """The query cannot be embedded the way the index was. `message` is shown as is."""


def normalize(v: list[float]) -> list[float]:
    """L2-normalise, so cosine similarity is a plain dot product. Ollama already
    returns unit vectors for these models; doing it here too means the index
    never depends on that."""
    n = math.sqrt(sum(x * x for x in v))
    if not math.isfinite(n) or n == 0:
        raise ValueError("The embedding model returned an empty or invalid vector.")
    return [x / n for x in v]


class Embedder:
    def __init__(self, spec: EmbeddingSpec, client: OllamaClient | None = None,
                 expect_digest: str | None = None):
        self.spec = spec
        self.client = client or OllamaClient()
        self.expect_digest = expect_digest
        self._checked_at: float | None = None

    def digest(self) -> str | None:
        info = self.client.model_info(self.spec.model)
        return info.digest if info else None

    def check(self) -> str:
        """The installed model must exist and, when the index recorded one,
        have the same digest. Returns the digest."""
        info = self.client.model_info(self.spec.model)
        if info is None:
            raise ProfileMismatch(
                f"The embedding model {self.spec.model} is not installed. "
                f"Install it with: ollama pull {self.spec.model}")
        if self.expect_digest and info.digest != self.expect_digest:
            raise ProfileMismatch(
                f"The installed {self.spec.model} is a different build from the one this index "
                f"was made with ({(info.digest or '-')[:12]} vs {self.expect_digest[:12]}). "
                "Rebuild the index with ./rag ingest.")
        self._checked_at = time.monotonic()
        return info.digest or ""

    def _embed(self, texts: list[str]) -> list[list[float]]:
        # Re-checked once a minute, so a model re-pulled while the app runs is
        # caught on the next question rather than at the next restart.
        if self._checked_at is None or time.monotonic() - self._checked_at > RECHECK_S:
            self.check()
        try:
            vectors = self.client.embed(self.spec.model, texts)
        except OllamaError as e:
            raise RuntimeError(f"Embedding failed: {e.message}") from None
        out = [normalize(v) for v in vectors]
        for v in out:
            if len(v) != self.spec.dimensions:
                raise ProfileMismatch(
                    f"{self.spec.model} returned {len(v)} dimensions; the index expects "
                    f"{self.spec.dimensions}.")
        return out

    def documents(self, texts: list[str],
                  progress: Callable[[int, int], None] | None = None) -> list[list[float]]:
        vectors: list[list[float]] = []
        b = self.spec.batch_size
        for i in range(0, len(texts), b):
            vectors.extend(self._embed([self.spec.doc_prefix + t for t in texts[i:i + b]]))
            if progress:
                progress(len(vectors), len(texts))
        return vectors

    def query(self, text: str) -> list[float]:
        return self._embed([self.spec.query_prefix + text])[0]
