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


def pytest_collection_modifyitems(config, items):
    if any("ollama" in item.keywords for item in items) and not _ollama_up():
        skip = pytest.mark.skip(reason=f"Ollama is not reachable at {OLLAMA_URL}")
        for item in items:
            if "ollama" in item.keywords:
                item.add_marker(skip)


@pytest.fixture
def client():
    from fastapi.testclient import TestClient

    import main

    return TestClient(main.app, base_url="http://127.0.0.1:8775")
