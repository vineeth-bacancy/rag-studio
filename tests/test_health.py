"""System Health: correct with Ollama down, and lists real models when it is up."""
from __future__ import annotations

import pytest

from rag_studio import health
from rag_studio.llm import OllamaClient, same_model


def test_same_model_ignores_latest_suffix():
    assert same_model("nomic-embed-text", "nomic-embed-text:latest")
    assert not same_model("qwen3:1.7b", "qwen3:8b")


def test_report_survives_ollama_down(monkeypatch):
    # Port 9 (discard) refuses connections on loopback: Ollama "down".
    down = OllamaClient("http://127.0.0.1:9")
    monkeypatch.setattr(health, "OllamaClient", lambda: down)
    report = health.report()
    ai = next(s for s in report["sections"] if s["id"] == "ai")
    assert ai["checks"][0]["status"] == "unavailable"
    assert ai["checks"][0]["message"].startswith("AI unavailable.")
    assert report["overall"] == "needs_attention"
    # Roles are still listed, with "unknown" rather than a guessed status.
    assert all(r["installed"] is None for r in report["roles"])


def test_foreign_resident_model_is_reported_busy():
    snap = {"up": True, "host": "x", "version": "0", "error": None,
            "installed": [{"name": "qwen3:1.7b"}, {"name": "qwen3:8b"}, {"name": "nomic-embed-text:latest"}],
            "resident": [{"name": "gemma4:e4b", "size": 6_600_000_000, "expires_at": None}]}
    checks = health._ai_checks(snap, health.role_rows(snap))
    busy = next(c for c in checks if c.id == "busy")
    assert busy.status == "warning"
    assert "gemma4:e4b" in busy.message


def test_status_endpoint_shape(client):
    r = client.get("/api/status")
    assert r.status_code == 200
    body = r.json()
    assert set(body) >= {"ai", "documents", "indexed", "version"}
    assert body["ai"]["backend"] == "local"
    assert body["documents"] == 3


@pytest.mark.ollama
def test_health_lists_installed_models(client):
    body = client.get("/api/health").json()
    assert body["ollama"]["up"] is True
    names = [m["name"] for m in body["ollama"]["installed"]]
    assert any(same_model(n, "nomic-embed-text") for n in names)
    for m in body["ollama"]["installed"]:
        assert m["capabilities"], f"{m['name']} reported no capabilities"
