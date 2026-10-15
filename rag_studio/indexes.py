"""The Indexes page's facts: every profile, its builds, and what can run now.

For each index profile: its embedding setup, its builds, its last build job
and vector match check, and three answers with a reason whenever the answer
is no: can it be built on this computer, can it be built on the Colab
runtime, and can a question be embedded for it right now. The page and the
pickers read this; nothing here starts work.

Cheap on purpose (the page polls it during a build): one `/api/tags` call to
local Ollama, and the Colab runtime's models as saved at connect time, never a
request through the tunnel.
"""
from __future__ import annotations

from typing import Any

from rag_studio import embed, ingest, profiles, remote, store
from rag_studio.llm import OllamaClient, OllamaUnavailable, same_model

SEARCH_METHODS = [
    {"id": "dense", "label": "Dense", "explain": "By meaning: the nearest vectors."},
    {"id": "bm25", "label": "BM25", "explain": "By words: the full-text index."},
    {"id": "hybrid", "label": "Hybrid", "explain": "Both lists, fused by rank (RRF)."},
]


def _yes() -> dict[str, Any]:
    return {"ok": True, "reason": None}


def _no(reason: str, fix: str | None = None) -> dict[str, Any]:
    return {"ok": False, "reason": reason, "fix": fix}


def _text_key(p: profiles.IndexProfile) -> tuple:
    return (p.parser, p.cleaner, repr(p.chunker.describe()))


def overview() -> dict[str, Any]:
    conn = remote.connection()
    try:
        local = OllamaClient().installed()
        local_error = None
    except OllamaUnavailable as e:
        local, local_error = [], e.reason
    served = conn.get("models") or []
    downloaded: dict[str, bool] = {}

    def here(spec: profiles.EmbeddingSpec) -> dict[str, Any]:
        name = embed.model_name(spec)
        if spec.backend == embed.ST:
            if spec.model not in downloaded:
                downloaded[spec.model] = embed.st_snapshot(spec) is not None
            return _yes() if downloaded[spec.model] else _no(
                f"{name} is not downloaded yet (about 1.5 GB).", f"./rag index download {spec.model}")
        if local_error:
            return _no(f"AI unavailable. {local_error}")
        if any(same_model(m.name, spec.model) for m in local):
            return _yes()
        return _no(f"{name} is not installed on this computer.", f"ollama pull {spec.model}")

    def there(spec: profiles.EmbeddingSpec) -> dict[str, Any]:
        name = embed.model_name(spec)
        if spec.backend == embed.ST:
            return _no(f"{name} runs in this app, on this computer only.")
        if not conn["connected"]:
            return _no("Remote AI is not connected.", "./rag llm connect")
        if served and not any(same_model(m, spec.model) for m in served):
            return _no(f"The Colab runtime does not serve {spec.model}. Run the setup cell with it.")
        return _yes()

    # Profiles that cut the same text have the same passages: a build's count
    # stands in for its siblings' until they are built.
    manifests = {p.id: store.active_manifest(p.id) for p in profiles.PROFILES.values()}
    passages_by_text: dict[tuple, int] = {}
    for p in profiles.PROFILES.values():
        if manifests[p.id]:
            passages_by_text.setdefault(_text_key(p), manifests[p.id]["rows"])

    def minutes(p: profiles.IndexProfile, where: str) -> float | None:
        rate = profiles.EMBED_RATE.get((p.embedding.model, where))
        n = passages_by_text.get(_text_key(p))
        return round(n / rate / 60, 1) if rate and n else None

    out = []
    for p in profiles.PROFILES.values():
        spec = p.embedding
        m = manifests[p.id]
        name = embed.model_name(spec)
        build_local = (here(spec) if spec.builds_on("local") else
                       _no(f"Built on the Colab GPU: {name} is too large to embed every passage on this CPU."))
        build_colab = there(spec) if spec.builds_on("colab") else (
            _no(f"{name} runs in this app, on this computer only.") if spec.backend == embed.ST
            else _no("Built on this computer."))
        build_local = {**build_local, "minutes": minutes(p, "local")}
        build_colab = {**build_colab, "minutes": minutes(p, "colab")}
        query = here(spec) if spec.query_on == "local" else there(spec)
        if m and m["config"] != p.config():
            query = _no("Built with a different setup than the app now uses for it. Rebuild it.")
        elif m and spec.backend == "ollama" and spec.query_on == "local" and query["ok"]:
            # The installed model must be the build the index was made with.
            info = next((x for x in local if same_model(x.name, spec.model)), None)
            if info and info.digest != m["embedding"]["digest"]:
                query = _no(f"The installed {spec.model} is a different build from the one this index was "
                            "made with. Rebuild it.")
        out.append({
            "id": p.id, "label": p.label, "description": p.description, "parser": p.parser,
            "chunker": p.chunker.method, "labels": p.chunker.labels,
            "word_index": bool(m and m.get("word_index")),
            "active": m, "builds": len(store.builds(p.id)),
            "history": [{"table": b["table"], "built_at": b["built_at"], "rows": b["rows"],
                         "built_on": b["embedding"].get("built_on", "local"),
                         "seconds": (b.get("timings") or {}).get("total_s")} for b in store.builds(p.id)],
            "embedding": {"model": spec.model, "name": name, "dimensions": spec.dimensions,
                          "backend": spec.backend, "build_on": spec.build_on, "query_on": spec.query_on,
                          "doc_prefix": spec.doc_prefix, "query_prefix": spec.query_prefix,
                          "options": spec.options, "revision": spec.revision},
            "can": {"build_local": build_local, "build_colab": build_colab, "query": query},
            "build_ram_gb": profiles.BUILD_RAM_GB.get(spec.model),
            "job": ingest.last_job(p.id),
            "check": embed.last_match(p.id),
        })
    free = ingest.available_ram_gb()
    return {"profiles": out, "search_methods": SEARCH_METHODS,
            "remote": {k: conn.get(k) for k in ("connected", "host", "models", "connected_at")},
            "free_ram_gb": round(free, 1) if free is not None else None,
            "active_build": ingest.active_build()}
