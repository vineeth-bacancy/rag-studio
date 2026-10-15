"""Embed: turn passages and questions into vectors.

An `Embedder` is built from an index profile's embedding spec plus the model
digest the index was built with, and refuses to embed with anything else: if
the model's digest differs from the manifest's (a re-pull changed the
weights), the vectors would no longer be comparable and every search would
quietly get worse. So a mismatch is an error, never a fallback.

Two backends (M4):

* **Ollama**, on this computer or on the connected Colab runtime (same API,
  plus a token). `on` names the machine. A Colab embedder looks up the
  connection on every call, so a disconnect is refused at once and a
  reconnect (a new tunnel URL) is picked up without a restart.
* **sentence-transformers, in this process**, for EmbeddingGemma 2 (Ollama's
  tags for it need the MLX runner, which Linux lacks). The "digest" is the
  Hugging Face commit, pinned in the profile and loaded from the local cache
  only: a question never makes a network call.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np

from rag_studio import paths
from rag_studio.llm import OllamaClient, OllamaError, OllamaUnavailable
from rag_studio.profiles import MODEL_NAMES, EmbeddingSpec

RECHECK_S = 60.0
ST = "sentence-transformers"
ST_LOAD_GB = 2.2   # EmbeddingGemma 2's text tower in float32, with Python (peak RSS, 2026-10-15)


class ProfileMismatch(RuntimeError):
    """The query cannot be embedded the way the index was. `message` is shown as is."""


class RemoteUnavailable(ProfileMismatch):
    """This index embeds on the Colab runtime, and it is not usable now."""


def normalize(v: list[float]) -> list[float]:
    """L2-normalise, so cosine similarity is a plain dot product. Both
    backends already return unit vectors for these models; doing it here too
    means the index never depends on that."""
    n = math.sqrt(sum(x * x for x in v))
    if not math.isfinite(n) or n == 0:
        raise ValueError("The embedding model returned an empty or invalid vector.")
    return [x / n for x in v]


def model_name(spec: EmbeddingSpec) -> str:
    return MODEL_NAMES.get(spec.model, spec.model)


# --------------------------------------------------------------------------- sentence-transformers

def st_snapshot(spec: EmbeddingSpec) -> Path | None:
    """The downloaded model folder for the pinned revision, or None. Never
    touches the network."""
    from huggingface_hub import snapshot_download
    from huggingface_hub.errors import LocalEntryNotFoundError
    try:
        return Path(snapshot_download(spec.model, revision=spec.revision, local_files_only=True))
    except (LocalEntryNotFoundError, FileNotFoundError, OSError):
        return None


def st_download(spec: EmbeddingSpec) -> Path:
    """Fetch the pinned revision from Hugging Face (the one network call this
    backend makes; `./rag index download`)."""
    from huggingface_hub import snapshot_download
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    return Path(snapshot_download(spec.model, revision=spec.revision))


def st_versions() -> dict[str, str]:
    import sentence_transformers
    import torch
    import transformers
    return {"sentence_transformers": sentence_transformers.__version__,
            "transformers": transformers.__version__, "torch": torch.__version__}


class STModel:
    """One sentence-transformers model, loaded once per process (about 1.1 GB
    of RAM for EmbeddingGemma 2's text tower in float32)."""

    _loaded: dict[tuple, "STModel"] = {}
    _lock = threading.Lock()

    def __init__(self, spec: EmbeddingSpec, folder: Path):
        import torch
        from sentence_transformers import SentenceTransformer
        from transformers.utils import logging as hf_logging
        hf_logging.disable_progress_bar()       # no "Loading weights" bars in the API's log
        self.spec = spec
        dtype = getattr(torch, spec.dtype or "float32")
        # Text only: the vision and audio towers (about 470M parameters) stay
        # unloaded until figures need them (M7).
        self.model = SentenceTransformer(
            str(folder), device="cpu", local_files_only=True,
            model_kwargs={"dtype": dtype},
            config_kwargs={"vision_config": None, "audio_config": None})
        self.model.eval()

    @classmethod
    def get(cls, spec: EmbeddingSpec) -> "STModel":
        key = (spec.model, spec.revision, spec.dtype)
        with cls._lock:
            if key not in cls._loaded:
                # Loading takes 2.2 GB (measured peak RSS). Refuse rather than
                # push this laptop into swap (the 2026-10-15 crash).
                from rag_studio import ingest
                free = ingest.available_ram_gb()
                if free is not None and free < ST_LOAD_GB:
                    raise ProfileMismatch(
                        f"Loading {model_name(spec)} needs about {ST_LOAD_GB} GB of memory and only {free:.1f} GB "
                        "is free. Close some apps and try again, or pick another index.")
                folder = st_snapshot(spec)
                if folder is None:
                    raise ProfileMismatch(
                        f"{model_name(spec)} is not downloaded yet (about 1.5 GB from Hugging Face). "
                        f"Download it with: ./rag index download {spec.model}")
                cls._loaded[key] = cls(spec, folder)
            return cls._loaded[key]

    def embed(self, texts: list[str]) -> list[list[float]]:
        import torch
        with torch.inference_mode():
            # prompt="" so no prompt from the model's own config is added on
            # top of ours: the profile's prefix is the whole prefix.
            out = self.model.encode(texts, prompt="", batch_size=self.spec.batch_size,
                                    convert_to_numpy=True, normalize_embeddings=False,
                                    show_progress_bar=False)
        if not np.isfinite(out).all():
            raise ValueError(f"{model_name(self.spec)} returned NaN or infinite values.")
        return out.astype(np.float64).tolist()


# --------------------------------------------------------------------------- the embedder

class Embedder:
    def __init__(self, spec: EmbeddingSpec, client: Any = None, expect_digest: str | None = None,
                 *, on: str = "local"):
        if on not in ("local", "colab"):
            raise ValueError("An embedder runs on 'local' or 'colab'.")
        if spec.backend == ST and on != "local":
            raise ProfileMismatch(f"{model_name(spec)} runs in this app, not on the Colab runtime.")
        self.spec = spec
        self.on = on
        self._client = client
        self.expect_digest = expect_digest
        self._checked_at: float | None = None
        self._checked_url: str | None = None

    @property
    def backend(self) -> str:
        """Which machine embeds, as the UI words it: local or remote."""
        return "remote" if self.on == "colab" else "local"

    def client(self) -> Any:
        if self._client is not None:
            return self._client
        if self.on == "colab":
            from rag_studio import remote
            try:
                return remote.client()
            except remote.RemoteNotConnected:
                raise RemoteUnavailable(
                    f"This index embeds questions with {model_name(self.spec)} on the Colab GPU, and no "
                    "runtime is connected. Connect one in Settings > Model backend, or pick another index.") from None
        return OllamaClient()

    def digest(self) -> str | None:
        if self.spec.backend == ST:
            return f"hf:{self.spec.revision}" if st_snapshot(self.spec) else None
        info = self.client().model_info(self.spec.model)
        return info.digest if info else None

    def check(self) -> str:
        """The model must exist where this embedder runs and, when the index
        recorded one, be the same build. Returns the digest."""
        if self.spec.backend == ST:
            if st_snapshot(self.spec) is None:
                raise ProfileMismatch(
                    f"{model_name(self.spec)} is not downloaded yet (about 1.5 GB from Hugging Face). "
                    f"Download it with: ./rag index download {self.spec.model}")
            digest = f"hf:{self.spec.revision}"
        else:
            c = self.client()
            try:
                info = c.model_info(self.spec.model)
            except OllamaUnavailable as e:
                if self.on == "colab":
                    raise RemoteUnavailable(f"{e.reason} Re-run the Colab cell and connect again, "
                                            "or pick another index.") from None
                raise
            if info is None:
                if self.on == "colab":
                    raise RemoteUnavailable(
                        f"The Colab runtime does not serve {self.spec.model}. Run the setup cell with that "
                        "model (Settings > Model backend), then connect again.")
                raise ProfileMismatch(
                    f"The embedding model {self.spec.model} is not installed. "
                    f"Install it with: ollama pull {self.spec.model}")
            digest = info.digest or ""
            self._checked_url = getattr(c, "base_url", None)
        if self.expect_digest and digest != self.expect_digest:
            where = "the Colab runtime's" if self.on == "colab" else "the installed"
            raise ProfileMismatch(
                f"{where.capitalize()} {self.spec.model} is a different build from the one this index "
                f"was made with ({(digest or '-').split(':')[-1][:12]} vs "
                f"{self.expect_digest.split(':')[-1][:12]}). Rebuild the index.")
        self._checked_at = time.monotonic()
        return digest

    def _embed(self, texts: list[str]) -> list[list[float]]:
        # Re-checked once a minute, so a model re-pulled while the app runs is
        # caught on the next question rather than at the next restart; and at
        # once when the Colab connection moved to another runtime.
        stale = self._checked_at is None or time.monotonic() - self._checked_at > RECHECK_S
        if self.spec.backend == ST:
            if stale:
                self.check()
            vectors = STModel.get(self.spec).embed(texts)
        else:
            c = self.client()
            if stale or getattr(c, "base_url", None) != self._checked_url:
                self.check()
            try:
                vectors = c.embed(self.spec.model, texts, options=self.spec.options)
            except OllamaUnavailable as e:
                if self.on == "colab":
                    raise RemoteUnavailable(f"{e.reason} Re-run the Colab cell and connect again.") from None
                raise
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

    def runtime(self) -> dict[str, Any]:
        """What a manifest records about the machine and software that embedded."""
        if self.spec.backend == ST:
            return {"backend": ST, "built_on": "local", "revision": self.spec.revision,
                    "dtype": self.spec.dtype, **st_versions()}
        c = self.client()
        out: dict[str, Any] = {"backend": "ollama", "built_on": self.on, "ollama_version": c.version()}
        info = c.model_info(self.spec.model)
        out["quantization"] = info.quantization if info else None
        if self.on == "colab":
            try:
                gpus = (c.gpu() or {}).get("gpus") or []
                out["gpu"] = gpus[0]["name"] if gpus else None
            except (OllamaUnavailable, OllamaError):
                out["gpu"] = None
        return out


# --------------------------------------------------------------------------- the vector cache

class VectorCache:
    """Vectors by content hash, kept on disk per model build, prefix and
    machine. Two jobs:

    * Semantic chunking embeds every sentence of every page; the cache makes
      a rebuild, or the chunk playground asking for a page again, free.
    * **Index builds resume from it.** Vectors are saved every `save_every`
      new passages, so a build cut off (Colab reclaimed the runtime, Ctrl-C)
      starts again where it stopped.

    One `.npz` per (model digest, passage prefix, dimensions, options, machine): a
    re-pulled model never serves another build's vectors, and vectors made on
    the Colab GPU never mix into a local build (the match check measures how
    close the two are; it does not assume it). Written atomically (rule 4);
    `np.savez` is given an open file, never a path, so it cannot append `.npz`
    to the temporary name.
    """

    def __init__(self, embedder: Embedder):
        self.embedder = embedder
        spec = embedder.spec
        digest = embedder.check().split(":")[-1][:12] or "unknown"
        slug = "".join(c if c.isalnum() else "-" for c in spec.model)
        ident = f"{spec.doc_prefix}|{spec.dimensions}|{spec.backend}"
        if spec.options:
            ident += "|" + json.dumps(spec.options, sort_keys=True)
        tag = hashlib.sha256(ident.encode()).hexdigest()[:8]
        folder = paths.CACHE / "vectors"
        self.path = folder / f"{slug}-{digest}-{tag}-{embedder.on}.npz"
        # Before M4 the name was model + digest only, and only local nomic
        # profiles (one prefix) wrote these, so a local cache can take it over.
        self._legacy = folder / f"{slug}-{digest}.npz" if embedder.on == "local" else None
        self._vectors: dict[str, np.ndarray] = {}
        self._dirty = False
        self._load()

    @staticmethod
    def key(text: str) -> str:
        return hashlib.sha256(text.encode("utf-8")).hexdigest()[:32]

    def _load(self) -> None:
        path = self.path if self.path.is_file() else self._legacy
        if path is None or not path.is_file():
            return
        with np.load(path) as z:
            self._vectors = dict(zip(z["keys"].tolist(), np.asarray(z["vectors"], dtype=np.float32)))

    def __len__(self) -> int:
        return len(self._vectors)

    def __contains__(self, text: str) -> bool:
        return self.key(text) in self._vectors

    def embed(self, texts: list[str], progress: Callable[[int, int], None] | None = None,
              save_every: int = 256) -> list[list[float]]:
        """Vectors for `texts`, embedding only those not cached, saved to disk
        every `save_every`. `progress(done, total)` counts the ones embedded
        now; it may raise to stop (the vectors so far are kept)."""
        keys = [self.key(t) for t in texts]
        todo = list(dict.fromkeys(t for t, k in zip(texts, keys) if k not in self._vectors))
        done = 0
        for i in range(0, len(todo), save_every):
            part = todo[i:i + save_every]
            vectors = self.embedder.documents(
                part, progress=(lambda d, _t, base=done: progress(base + d, len(todo))) if progress else None)
            for t, v in zip(part, vectors):
                self._vectors[self.key(t)] = np.asarray(v, dtype=np.float32)
            done += len(part)
            self._dirty = True
            self.save()
        return [self._vectors[k].astype(np.float64).tolist() for k in keys]

    def save(self) -> None:
        if not self._dirty:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        keys = list(self._vectors)
        fd, tmp = tempfile.mkstemp(dir=self.path.parent, prefix=f".{self.path.name}.", suffix=".tmp")
        try:
            with os.fdopen(fd, "wb") as f:
                np.savez(f, keys=np.array(keys), vectors=np.stack([self._vectors[k] for k in keys]))
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp, self.path)
        except BaseException:
            Path(tmp).unlink(missing_ok=True)
            raise
        self._dirty = False


# --------------------------------------------------------------------------- the vector match check

MATCH_THRESHOLD = 0.999
MATCH_PASSAGES = 50
MATCH_QUESTIONS = 5
NOISE_PASSAGES = 20


def match_path(profile_id: str) -> Path:
    return paths.INDEXES / "checks" / f"{profile_id}.json"


def _cosines(a: list[list[float]], b: list[list[float]]) -> dict[str, float | int]:
    x, y = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    x /= np.linalg.norm(x, axis=1, keepdims=True)
    y /= np.linalg.norm(y, axis=1, keepdims=True)
    c = np.sum(x * y, axis=1)
    return {"n": int(len(c)), "min": float(c.min()), "median": float(np.median(c)),
            "mean": float(c.mean())}


def _one_by_one(e: Embedder, texts: list[str], order: list[int] | None = None) -> list[list[float]]:
    """One text per request, in `order`; returned in the texts' own order."""
    out: list[list[float] | None] = [None] * len(texts)
    for i in (order if order is not None else range(len(texts))):
        out[i] = e.documents([texts[i]])[0]
    return out  # type: ignore[return-value]


def match_check(profile_id: str, *, clients: dict[str, Any] | None = None,
                n: int = MATCH_PASSAGES) -> dict[str, Any]:
    """Do two computers embed this index's passages the same way?

    Measured 2026-10-15: through Ollama 0.32.6 **a text's vector depends on
    what the server embedded just before it**. The server reuses the cached
    keys and values of a prefix shared with the previous request, computed in
    a differently shaped calculation, so the same text moves in the 4th
    decimal (Qwen3 0.6B) or the 3rd (the 4-bit 8B on a T4). One machine is
    not exactly equal to itself, so the check measures that first:

    * **noise floor**, per machine: 20 passages embedded one per request in
      two orders (this machine against itself);
    * **across machines** (spec §2.2): `n` passages and 5 test questions, one
      per request, in the same order on this computer and on the Colab
      runtime. Passes when every cosine is at least 0.999;
    * **against the index**, per machine: the same passages against the
      vectors stored at build time (embedded in batches). Reported, not
      judged: it shows how far a fresh embedding sits from the stored one.

    With one machine there is nothing to compare across, and `passed` is
    None. `clients` maps "local" / "colab" to an Ollama client (tests pass
    fakes). Saved for the Indexes page."""
    from rag_studio import eval as evaluation
    from rag_studio import profiles, store
    from rag_studio.files import write_json_atomic

    profile = profiles.get(profile_id)
    manifest = store.active_manifest(profile_id)
    if manifest is None:
        raise LookupError(f"The {profile.label} index is not built yet. Build it first.")
    if manifest["config"] != profile.config():
        raise ProfileMismatch(f"The {profile.label} index was built with a different setup. Rebuild it first.")
    spec = profile.embedding
    t = store.open_table(manifest["table"]).to_arrow().select(["chunk_id", "search_text", "vector"])
    picks = sorted({int(i) for i in np.linspace(0, t.num_rows - 1, min(n, t.num_rows))})
    texts = [t["search_text"][i].as_py() for i in picks]
    stored = [t["vector"][i].as_py() for i in picks]
    questions = [q["question"] for s in evaluation.load_sets().values() for q in s["questions"]][:MATCH_QUESTIONS]
    k = min(NOISE_PASSAGES, len(texts))

    sides: dict[str, dict[str, Any]] = {}
    got: dict[str, tuple[list[list[float]], list[list[float]]]] = {}
    pairs: list[dict[str, Any]] = []
    for on in ("local", "colab"):
        if spec.backend == ST and on == "colab":
            sides[on] = {"available": False, "reason": f"{model_name(spec)} runs only in this app."}
            continue
        if on == "local" and spec.query_on == "colab" and not spec.builds_on("local") and not (clients or {}).get(on):
            # Not installed here by choice (the user's M4 decision for the 8B).
            try:
                if OllamaClient().model_info(spec.model) is None:
                    sides[on] = {"available": False,
                                 "reason": f"{spec.model} is not installed on this computer: this index "
                                           "is queried on the Colab runtime only."}
                    continue
            except OllamaUnavailable as e:
                sides[on] = {"available": False, "reason": e.reason}
                continue
        try:
            e = Embedder(spec, client=(clients or {}).get(on), expect_digest=manifest["embedding"]["digest"], on=on)
            started = time.perf_counter()
            docs = _one_by_one(e, texts)
            again = _one_by_one(e, texts[:k], list(reversed(range(k))))
            qs = [e.query(q) for q in questions]
            seconds = time.perf_counter() - started
            rt = e.runtime()
        except (ProfileMismatch, OllamaUnavailable, OllamaError) as err:
            reason = err.reason if isinstance(err, OllamaUnavailable) else (
                err.message if isinstance(err, OllamaError) else str(err))
            sides[on] = {"available": False, "reason": reason}
            continue
        got[on] = (docs, qs)
        sides[on] = {"available": True, "seconds": round(seconds, 2),
                     **{k_: v for k_, v in rt.items() if k_ in ("ollama_version", "gpu", "torch", "quantization")}}
        pairs.append({"a": on, "b": on, "what": "passages", "kind": "noise", **_cosines(docs[:k], again)})
        pairs.append({"a": on, "b": "stored", "what": "passages", "kind": "index", **_cosines(docs, stored)})
    if len(got) == 2:
        pairs.append({"a": "local", "b": "colab", "what": "passages", "kind": "cross",
                      **_cosines(got["local"][0], got["colab"][0])})
        pairs.append({"a": "local", "b": "colab", "what": "questions", "kind": "cross",
                      **_cosines(got["local"][1], got["colab"][1])})
    cross = [p for p in pairs if p["kind"] == "cross"]
    noise = [p["min"] for p in pairs if p["kind"] == "noise"]
    result = {
        "profile": profile_id, "label": profile.label, "table": manifest["table"],
        "model": spec.model, "digest": manifest["embedding"]["digest"],
        "built_on": manifest["embedding"].get("built_on", "local"),
        "passages": len(texts), "questions": len(questions), "noise_passages": k,
        "threshold": MATCH_THRESHOLD, "sides": sides, "pairs": pairs,
        "cross_machine": bool(cross),
        "passed": all(p["min"] >= MATCH_THRESHOLD for p in cross) if cross else None,
        "noise_floor": min(noise) if noise else None,
        "method": "one text per request; noise floor = the same machine in two orders",
        "checked_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    write_json_atomic(match_path(profile_id), result)
    return result


def last_match(profile_id: str) -> dict[str, Any] | None:
    from rag_studio.files import read_json
    return read_json(match_path(profile_id))
