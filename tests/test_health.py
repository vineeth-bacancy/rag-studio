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
    # Roles that run in the local Ollama are still listed, with "unknown" rather than a guessed status.
    # (Roles that run in this app or on Remote AI do not depend on the local Ollama.)
    local_ollama = [r for r in report["roles"] if r["runs_on"] == "ollama" and r["where"] == "local"]
    assert local_ollama and all(r["installed"] is None for r in local_ollama)
    assert [d["key"] for d in report["defaults"]][:3] == ["index", "prompt", "search"]


def test_python_roles_report_their_download_without_loading(monkeypatch):
    from rag_studio import embed, rerank
    monkeypatch.setattr(embed, "st_snapshot", lambda spec: None)
    monkeypatch.setattr(rerank, "snapshot", lambda spec: None)
    # Another test in the same process may have loaded the real models: pin "not loaded".
    monkeypatch.setattr(embed.STModel, "is_loaded", classmethod(lambda cls, spec: False))
    monkeypatch.setattr(rerank.LocalReranker, "is_loaded", classmethod(lambda cls, spec: False))
    rows = {r["role"]: r for r in health.role_rows({"up": False, "installed": [], "resident": []})}
    assert rows["embed"]["installed"] is False and rows["embed"]["loaded"] is False
    assert rows["rerank"]["installed"] is False and rows["rerank"]["loaded"] is False


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


def test_remote_roles_are_not_checked_against_the_local_ollama(monkeypatch):
    from rag_studio import health
    names = ["qwen3:1.7b", "nomic-embed-text:latest"]                   # no local role needs qwen3:8b (the trading app's)
    snap = {"up": True, "installed": [{"name": n} for n in names], "resident": [], "host": "h", "version": "0",
            "error": None}
    rows = {r["role"]: r for r in health.role_rows(snap)}
    for key in ("rewrite", "context"):
        assert rows[key]["where"] == "remote" and rows[key]["model"] == "qwen3:8b"
        assert rows[key]["installed"] is None                      # not connected: unknown, not "missing"
    assert health._ai_checks(snap, list(rows.values()))[1].status == "healthy"
    snap["installed"] = [{"name": "nomic-embed-text:latest"}]       # the answer model is gone locally
    miss = health._ai_checks(snap, list(health.role_rows(snap)))[1]
    assert miss.status == "needs_attention" and "ollama pull qwen3:1.7b" in miss.fix
    assert "qwen3:8b" not in miss.message and "qwen3:8b" not in (miss.fix or "")   # remote roles never ask for a pull
    assert rows["embed"]["runs_on"] == "python" and rows["rerank"]["runs_on"] == "python"


def test_a_connected_runtime_says_whether_it_serves_the_remote_roles(monkeypatch):
    from rag_studio import health, remote
    monkeypatch.setattr(remote, "connection", lambda: {"connected": True, "models": ["qwen3:8b"]})
    snap = {"up": True, "installed": [], "resident": [], "host": "h", "version": "0", "error": None}
    rows = {r["role"]: r for r in health.role_rows(snap)}
    assert rows["rewrite"]["installed"] is True and rows["rewrite"]["connected"] is True
    monkeypatch.setattr(remote, "connection", lambda: {"connected": True, "models": []})
    assert {r["role"]: r for r in health.role_rows(snap)}["context"]["installed"] is False


def test_the_models_this_app_loads_exclude_remote_roles():
    from rag_studio import config, health
    assert {"qwen3:1.7b", "nomic-embed-text:latest"} <= health._our_models()
    assert [r.key for r in config.ollama_roles()] == ["answer"]   # rewrite and context run on Remote AI; no judge in this version
