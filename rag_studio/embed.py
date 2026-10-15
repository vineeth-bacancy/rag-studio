"""Embed: turn passages and questions into vectors.

An `Embedder` is built from an index profile's embedding spec plus the model
digest the index was built with. It refuses to embed with anything else: if
the installed model's digest differs from the manifest's (a re-pull changed
the weights), the vectors would no longer be comparable and every search
would quietly get worse. So the mismatch is an error, never a fallback.
"""
from __future__ import annotations

import hashlib
import math
import os
import tempfile
import time
from pathlib import Path
from typing import Callable

import numpy as np

from rag_studio import paths
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


class VectorCache:
    """Vectors of short texts (sentences, for semantic chunking) by content
    hash, kept on disk per embedding model build. Semantic chunking embeds
    every sentence of every page; the cache makes a rebuild, or the chunk
    playground asking for the same page twice, cost nothing the second time.

    One `.npz` per model digest, so a re-pulled model never serves another
    build's vectors. Written atomically (rule 4); `np.savez` is given an open
    file, never a path, so it cannot append `.npz` to the temporary name.
    """

    def __init__(self, embedder: Embedder):
        self.embedder = embedder
        digest = embedder.check().split(":")[-1][:12] or "unknown"
        slug = "".join(c if c.isalnum() else "-" for c in embedder.spec.model)
        self.path = paths.CACHE / "vectors" / f"{slug}-{digest}.npz"
        self._vectors: dict[str, list[float]] = {}
        self._dirty = False
        self._load()

    @staticmethod
    def key(text: str) -> str:
        return hashlib.sha256(text.encode("utf-8")).hexdigest()[:32]

    def _load(self) -> None:
        if not self.path.is_file():
            return
        with np.load(self.path) as z:
            self._vectors = dict(zip(z["keys"].tolist(), z["vectors"].tolist()))

    def __len__(self) -> int:
        return len(self._vectors)

    def __contains__(self, text: str) -> bool:
        return self.key(text) in self._vectors

    def embed(self, texts: list[str], progress: Callable[[int, int], None] | None = None) -> list[list[float]]:
        keys = [self.key(t) for t in texts]
        todo = list(dict.fromkeys(t for t, k in zip(texts, keys) if k not in self._vectors))
        if todo:
            for t, v in zip(todo, self.embedder.documents(todo, progress=progress)):
                self._vectors[self.key(t)] = v
            self._dirty = True
        return [self._vectors[k] for k in keys]

    def save(self) -> None:
        if not self._dirty:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        keys = list(self._vectors)
        fd, tmp = tempfile.mkstemp(dir=self.path.parent, prefix=f".{self.path.name}.", suffix=".tmp")
        try:
            with os.fdopen(fd, "wb") as f:
                np.savez(f, keys=np.array(keys), vectors=np.array([self._vectors[k] for k in keys],
                                                                   dtype=np.float32))
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp, self.path)
        except BaseException:
            Path(tmp).unlink(missing_ok=True)
            raise
        self._dirty = False
