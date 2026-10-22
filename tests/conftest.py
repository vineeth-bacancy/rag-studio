"""Shared test setup.

Tests that need Ollama carry `@pytest.mark.ollama` and skip cleanly when it is
not reachable, so `./rag test` passes on a machine with nothing running.
"""
from __future__ import annotations

import sys
from pathlib import Path

import httpx
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "web" / "api"))

from rag_studio.config import OLLAMA_URL  # noqa: E402


def _ollama_up() -> bool:
    try:
        return httpx.get(f"{OLLAMA_URL}/api/version", timeout=1.0).status_code == 200
    except httpx.HTTPError:
        return False


def _colab_up() -> bool:
    """A Colab runtime is connected (in the real state folder) and answers."""
    from rag_studio import remote
    try:
        return remote.connected() and remote.probe()["reachable"]
    except Exception:  # noqa: BLE001 - any failure means skip
        return False


def _st_ready() -> bool:
    """EmbeddingGemma 2 is on disk and loading it (2.2 GB) cannot push this
    laptop into swap: the same bar a local build must clear."""
    from rag_studio import embed, ingest, profiles
    free = ingest.available_ram_gb()
    return (embed.st_snapshot(profiles.EMBEDDINGGEMMA2) is not None and free is not None
            and free >= profiles.BUILD_RAM_GB[profiles.EMBEDDINGGEMMA2.model])


def _rerank_ready() -> bool:
    """The reranker snapshot is on disk and loading it cannot push this laptop into swap."""
    from rag_studio import ingest, rerank
    free = ingest.available_ram_gb()
    return (rerank.snapshot(rerank.RERANKERS["qwen3-0.6b"]) is not None and free is not None
            and free >= rerank.RERANK_LOAD_GB)


def pytest_collection_modifyitems(config, items):
    if any("ollama" in item.keywords for item in items) and not _ollama_up():
        skip = pytest.mark.skip(reason=f"Ollama is not reachable at {OLLAMA_URL}")
        for item in items:
            if "ollama" in item.keywords:
                item.add_marker(skip)
    if any("st" in item.keywords for item in items) and not _st_ready():
        skip = pytest.mark.skip(reason="EmbeddingGemma 2 is not downloaded, or under 3.2 GB of memory is free")
        for item in items:
            if "st" in item.keywords:
                item.add_marker(skip)
    if any("rerank" in item.keywords for item in items) and not _rerank_ready():
        skip = pytest.mark.skip(reason="The reranker is not downloaded, or too little memory is free")
        for item in items:
            if "rerank" in item.keywords:
                item.add_marker(skip)
    if any("colab" in item.keywords for item in items) and not _colab_up():
        skip = pytest.mark.skip(reason="No Colab runtime is connected (./rag llm connect)")
        for item in items:
            if "colab" in item.keywords:
                item.add_marker(skip)


@pytest.fixture(autouse=True)
def neutral_memory(monkeypatch):
    """Builds in tests use fake models: never refuse them for the host's
    free memory. Tests of the memory guard set it themselves."""
    from rag_studio import ingest
    monkeypatch.setattr(ingest, "available_ram_gb", lambda: None)


@pytest.fixture(autouse=True)
def private_state(request, tmp_path_factory, monkeypatch):
    """No test reads or writes the real ~/.local/state/rag-studio (the Colab
    token lives there), except the ones marked `colab`, which need the real
    connection."""
    if "colab" in request.keywords:
        yield None
        return
    from rag_studio import paths
    folder = tmp_path_factory.mktemp("state")
    monkeypatch.setattr(paths, "STATE", folder)
    yield folder


@pytest.fixture
def client():
    from fastapi.testclient import TestClient

    import main

    return TestClient(main.app, base_url="http://127.0.0.1:8775")


@pytest.fixture
def tmp_data(tmp_path, monkeypatch):
    """Point every rebuilt folder (indexes, runs, cache) at a temporary
    directory. The PDFs and test questions stay the real, committed ones."""
    from rag_studio import paths, retrieve

    for name in ("LANCEDB", "INDEXES", "RUNS", "CACHE", "CHAT", "CONTEXT"):
        monkeypatch.setattr(paths, name, tmp_path / name.lower())
    monkeypatch.setattr(retrieve, "_cache", {})
    return tmp_path
