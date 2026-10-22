"""Rerank: read each candidate passage together with the question.

Search compares a question's vector with each passage's vector, so the two are
never read side by side. A **reranker** (a cross-encoder) reads the question and
one passage together and scores how well that passage answers it. It is slower
(one model pass per candidate), so it only re-orders the top `depth` candidates
that search found (20 by default), and everything below stays where it was.

The model is Qwen3-Reranker: a causal language model asked "does this document
meet the query? yes or no", scored by how much more it prefers "yes". The
sentence-transformers `CrossEncoder` ships that template, the yes/no token ids
and left padding (research §6), so this module does not re-implement them.

Where it runs is part of the setup (`RerankerSpec.where`): the 0.6B model runs
in this app on the CPU; the 4B and 8B run on the Colab GPU (M5, P2).
"""
from __future__ import annotations

import json
import math
import os
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

DEFAULT_DEPTH = 20

# RAM the 0.6B reranker needs on this computer, in GB, on top of what is
# already loaded. Set from `./rag rerank bench` (research §6).
RERANK_LOAD_GB = 4.0


class RerankError(RuntimeError):
    """A reranker cannot run now. `message` is shown as is: a short explanation, then what to do."""


# The instruction the model is told, and its version. A different wording is a
# different setup, so it is versioned and hashed like a prompt (rule 2).
INSTRUCTIONS: dict[str, str] = {
    "rerank-v1": "Given a question about a set of PDF reports, retrieve the passage that states the answer",
}
INSTRUCTION_VERSION = "rerank-v1"


def instruction_sha256(version: str = INSTRUCTION_VERSION) -> str:
    import hashlib
    return hashlib.sha256(INSTRUCTIONS[version].encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class RerankerSpec:
    """One reranker setup. `describe()` is what a run records."""
    id: str
    model: str
    revision: str
    dtype: str
    where: str                       # "local" (this app, CPU) or "colab" (the GPU runtime)
    label: str
    max_length: int = 1024
    batch_size: int = 4
    quantization: str | None = None  # "int8" for the 8B on the T4

    def describe(self, depth: int = DEFAULT_DEPTH) -> dict[str, Any]:
        d: dict[str, Any] = {
            "id": self.id, "model": self.model, "revision": self.revision, "dtype": self.dtype,
            "where": self.where, "depth": depth, "max_length": self.max_length,
            "instruction_version": INSTRUCTION_VERSION,
            "instruction_sha256": instruction_sha256(),
        }
        if self.quantization:
            d["quantization"] = self.quantization
        return d


# Revisions resolved from the Hugging Face API on 2026-10-16 (all three repos
# were last modified 2026-04-16, the sentence-transformers integration).
RERANKERS: dict[str, RerankerSpec] = {s.id: s for s in (
    RerankerSpec("qwen3-0.6b", "Qwen/Qwen3-Reranker-0.6B", "e61197ed45024b0ed8a2d74b80b4d909f1255473",
                 "float32", "local", "This computer · 0.6B"),
    RerankerSpec("qwen3-0.6b-gpu", "Qwen/Qwen3-Reranker-0.6B", "e61197ed45024b0ed8a2d74b80b4d909f1255473",
                 "float16", "colab", "Remote AI · 0.6B", batch_size=8),
    RerankerSpec("qwen3-4b", "Qwen/Qwen3-Reranker-4B", "22e683669bc0f0bd69640a1354a6d0aebcfeede5",
                 "float16", "colab", "Remote AI · 4B", batch_size=8),
    RerankerSpec("qwen3-8b", "Qwen/Qwen3-Reranker-8B", "77d193c791ed757ca307ee72715aa132723da912",
                 "float16", "colab", "Remote AI · 8B (8-bit)", batch_size=8, quantization="int8"),
)}


def get(reranker_id: str) -> RerankerSpec:
    try:
        return RERANKERS[reranker_id]
    except KeyError:
        raise ValueError(f"Unknown reranker {reranker_id!r}. Known: {', '.join(RERANKERS)}.") from None


# --------------------------------------------------------------------------- the model, on this computer

def snapshot(spec: RerankerSpec) -> Path | None:
    """The downloaded folder for the pinned revision, or None. Never touches the network."""
    from huggingface_hub import snapshot_download
    from huggingface_hub.errors import LocalEntryNotFoundError
    try:
        return Path(snapshot_download(spec.model, revision=spec.revision, local_files_only=True))
    except (LocalEntryNotFoundError, FileNotFoundError, OSError):
        return None


def download(spec: RerankerSpec) -> Path:
    """Fetch the pinned revision (the one network call this module makes;
    `./rag rerank download`)."""
    from huggingface_hub import snapshot_download
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    return Path(snapshot_download(spec.model, revision=spec.revision))


class LocalReranker:
    """A sentence-transformers CrossEncoder, loaded once per process. One
    inference at a time: an Ask can run while an app-started evaluation is
    reranking, and two passes would each take twice as long and double the
    memory in flight."""

    _loaded: dict[str, "LocalReranker"] = {}
    _lock = threading.Lock()
    _infer = threading.Lock()

    def __init__(self, spec: RerankerSpec, folder: Path):
        import torch
        from sentence_transformers import CrossEncoder
        from transformers.utils import logging as hf_logging
        hf_logging.disable_progress_bar()
        self.spec = spec
        self.model = CrossEncoder(
            str(folder), device="cpu", local_files_only=True, max_length=spec.max_length,
            prompts={"rag": INSTRUCTIONS[INSTRUCTION_VERSION]}, default_prompt_name="rag",
            model_kwargs={"dtype": getattr(torch, spec.dtype)})
        self.model.eval()

    @classmethod
    def is_loaded(cls, spec: RerankerSpec) -> bool:
        return spec.id in cls._loaded

    @classmethod
    def get(cls, spec: RerankerSpec) -> "LocalReranker":
        if spec.where != "local":
            raise RerankError(f"{spec.label} runs on the Remote AI runtime, not in this app. "
                              "Connect Remote AI in Settings, or pick the reranker on this computer.")
        with cls._lock:
            if spec.id not in cls._loaded:
                from rag_studio import ingest
                free = ingest.available_ram_gb()
                if free is not None and free < RERANK_LOAD_GB:
                    raise RerankError(
                        f"Loading the reranker needs about {RERANK_LOAD_GB} GB of memory and only {free:.1f} GB "
                        "is free. Close some apps and try again, or turn reranking off.")
                folder = snapshot(spec)
                if folder is None:
                    raise RerankError(f"The reranker is not downloaded yet (about 1.2 GB from Hugging Face). "
                                      f"Download it with: ./rag rerank download")
                cls._loaded[spec.id] = cls(spec, folder)
            return cls._loaded[spec.id]

    @classmethod
    def release(cls) -> None:
        """Unload every reranker (an evaluation does this between its search
        pass and its answer pass, to give the memory back)."""
        import gc
        with cls._lock:
            cls._loaded.clear()
        gc.collect()
        _trim()

    def score(self, query: str, documents: list[str]) -> list[float]:
        """One relevance score per document, higher is better: the raw logit
        difference yes - no (activation Identity), so only the order matters.
        Raises on NaN or inf."""
        import torch
        pairs = [(query, d) for d in documents]
        with self._infer, torch.inference_mode():
            raw = self.model.predict(pairs, batch_size=self.spec.batch_size, show_progress_bar=False,
                                     convert_to_numpy=True)
        scores = [float(x) for x in raw]
        check_scores(scores)
        return scores


def availability() -> list[dict[str, Any]]:
    """Every reranker, and whether it can run now with the reason when it
    cannot (the pickers show it; nothing is hidden). Cheap: no model is loaded
    and no request crosses the tunnel."""
    from rag_studio import ingest, remote
    rows = []
    for spec in RERANKERS.values():
        ok, reason, fix = True, None, None
        if spec.where == "local":
            free = ingest.available_ram_gb()
            if snapshot(spec) is None:
                ok, reason, fix = False, "The reranker is not downloaded yet (about 1.2 GB).", "./rag rerank download"
            elif free is not None and free < RERANK_LOAD_GB and not LocalReranker.is_loaded(spec):
                ok, reason = False, f"Loading it needs about {RERANK_LOAD_GB} GB of memory and {free:.1f} GB is free."
        else:
            conn = remote.connection()
            if not conn.get("connected"):
                ok, reason = False, "Needs Remote AI. Connect it in Settings."
            elif spec.id not in (conn.get("rerankers") or []):
                ok, reason = False, "The connected runtime is not running this reranker. Run the setup cell again with it."
        rows.append({"id": spec.id, "label": spec.label, "where": spec.where, "model": spec.model,
                     "can": {"ok": ok, "reason": reason, "fix": fix}})
    return rows


class RemoteReranker:
    """A reranker on the Colab runtime (tools/colab/worker.py), reached through
    the gateway: same `.spec` and `.score()` as `LocalReranker`, so the
    pipeline does not care where it runs. Each call's bytes and times go to
    `client.stats`, so a run can say how much was network and how much GPU."""

    def __init__(self, spec: RerankerSpec, client: Any):
        if spec.where != "colab":
            raise RerankError(f"{spec.label} runs in this app, not on the Remote AI runtime.")
        self.spec = spec
        self.client = client
        self.last: dict[str, Any] = {}
        self.last_net: dict[str, float] | None = None

    def score(self, query: str, documents: list[str]) -> list[float]:
        from rag_studio.llm import OllamaError, OllamaUnavailable
        from rag_studio.llm import diff_stats
        before = self.client.stats.snapshot()
        body = {"model": self.spec.id, "instruction": INSTRUCTIONS[INSTRUCTION_VERSION], "query": query,
                "documents": documents, "max_length": self.spec.max_length}
        try:
            out = self.client.post_json("/rerank", body)
        except OllamaUnavailable as e:
            raise RerankError(f"{e.reason} Check Remote AI in Settings, or use the reranker on this computer.") from None
        except OllamaError as e:
            raise RerankError(f"The Remote AI worker refused the request: {e.message}") from None
        scores = out.get("scores")
        if not isinstance(scores, list) or len(scores) != len(documents):
            raise RerankError("The Remote AI worker returned the wrong number of scores. "
                              "Run the setup cell again to restart it.")
        check_scores([float(x) for x in scores])
        got = {k: out.get(k) for k in ("model", "revision", "dtype", "quantization")}
        want = {"model": self.spec.model, "revision": self.spec.revision}
        if any(got[k] != v for k, v in want.items()):
            raise RerankError(f"The worker ran {got['model']}@{str(got['revision'])[:8]}, not the pinned "
                              f"{self.spec.model}@{self.spec.revision[:8]}. Run the setup cell again.")
        self.last = {k: out.get(k) for k in ("compute_ms", "max_memory_mb", "swapped_in", "load_s", "dtype")}
        self.last_net = diff_stats(self.client.stats.snapshot(), before)
        return [float(x) for x in scores]


class CachedScorer:
    """Remembers every score a reranker gave, on disk, by what was asked:
    reranker (model, revision, dtype, quantisation, length), instruction, the
    question and the exact passages in order. Scores are a pure function of
    that, so a retrieval-only run and the scorecard that follows it see the
    **same** numbers, and the scorecard needs no Colab runtime at all. Each
    answer says whether it came from the cache, and the time the model
    originally took. Pass `cache=False` for a fresh measurement."""

    def __init__(self, inner: Any):
        self.inner = inner
        self.spec = inner.spec
        self.last_cache: str | None = None
        self.last: dict[str, Any] = {}
        self.last_net: dict[str, float] | None = None

    def _path(self, query: str, documents: list[str]) -> Path:
        import hashlib
        from rag_studio import paths
        s = self.spec
        ns = hashlib.sha256(json.dumps([s.model, s.revision, s.dtype, s.quantization, s.max_length,
                                        INSTRUCTIONS[INSTRUCTION_VERSION]]).encode()).hexdigest()[:16]
        key = hashlib.sha256(json.dumps([query, documents]).encode()).hexdigest()
        return paths.CACHE / "rerank" / s.id / ns / f"{key}.json"

    def score(self, query: str, documents: list[str]) -> list[float]:
        path = self._path(query, documents)
        try:
            hit = json.loads(path.read_text())
            if isinstance(hit.get("scores"), list) and len(hit["scores"]) == len(documents):
                check_scores(hit["scores"])
                self.last_cache, self.last_net = "hit", None
                self.last = {"compute_ms": hit.get("compute_ms"), "dtype": self.spec.dtype, "from_cache": True}
                return [float(x) for x in hit["scores"]]
        except (OSError, ValueError, RerankError):
            pass                    # missing, unreadable or damaged: ask the model again
        t = time.perf_counter()
        scores = self.inner.score(query, documents)
        ms = (time.perf_counter() - t) * 1000
        compute = (getattr(self.inner, "last", None) or {}).get("compute_ms") or round(ms, 1)
        from rag_studio.files import write_json_atomic
        path.parent.mkdir(parents=True, exist_ok=True)
        write_json_atomic(path, {"scores": scores, "compute_ms": compute, "wall_ms": round(ms, 1),
                                 "at": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "documents": len(documents)})
        self.last_cache = "miss"
        self.last = {**(getattr(self.inner, "last", None) or {}), "from_cache": False}
        self.last_net = getattr(self.inner, "last_net", None)
        return scores


def scorer_for(spec: RerankerSpec, *, cache: bool = True) -> Any:
    """The object that scores for this reranker: in this app, or on the runtime
    (and remembered, unless `cache` is off). A cache hit never loads the model
    or touches the runtime."""
    def build() -> Any:
        if spec.where == "local":
            return LocalReranker.get(spec)
        from rag_studio import remote
        try:
            return RemoteReranker(spec, remote.client())
        except remote.RemoteNotConnected:
            raise RerankError(f"{spec.label} runs on the Remote AI runtime, and none is connected. "
                              "Connect one in Settings, or pick the reranker on this computer.") from None

    if not cache:
        return build()
    return CachedScorer(_Lazy(spec, build))


class _Lazy:
    """Builds the real scorer on first use, so a run that only replays the cache never loads a model or opens a connection."""

    def __init__(self, spec: RerankerSpec, build: Any):
        self.spec = spec
        self._build = build
        self._real: Any = None

    @property
    def last(self) -> dict[str, Any]:
        return getattr(self._real, "last", {}) if self._real else {}

    @property
    def last_net(self) -> dict[str, float] | None:
        return getattr(self._real, "last_net", None) if self._real else None

    def score(self, query: str, documents: list[str]) -> list[float]:
        if self._real is None:
            self._real = self._build()
        return self._real.score(query, documents)


def runtime_facts(reranker_id: str) -> dict[str, Any]:
    """What a run records about where the reranker ran, apart from its stable
    setup: library versions and CPU threads."""
    spec = get(reranker_id)
    if spec.where != "local":
        from rag_studio import remote
        info = remote.worker_info() if remote.connected() else None
        if info is None:
            return {"reranker": {"where": spec.where, "runtime": "not connected: scores come from the cache"}}
        return {"reranker": {"where": spec.where, "packages": info.get("packages"),
                             "gpu": (info.get("vram") or {}).get("name"), "worker_version": info.get("version")}}
    import sentence_transformers
    import torch
    return {"reranker": {"where": "local", "sentence_transformers": sentence_transformers.__version__,
                         "torch": torch.__version__, "threads": torch.get_num_threads()}}


def _trim() -> None:
    """Ask glibc to give freed memory back to the OS (a Python process otherwise keeps it)."""
    try:
        import ctypes
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except (OSError, AttributeError):
        pass


def check_scores(scores: list[float]) -> None:
    if any(not math.isfinite(s) for s in scores):
        raise RerankError("The reranker returned NaN or infinite scores, so its order cannot be trusted. "
                          "Turn reranking off for this question, or try the reranker on this computer.")
