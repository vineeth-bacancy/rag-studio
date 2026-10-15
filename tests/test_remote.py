"""Remote AI on the laptop side: the connect code, the private state files,
the setup cell, start.sh's helpers, and a full connect through the real
gateway on loopback.
"""
from __future__ import annotations

import base64
import gzip
import hashlib
import importlib.util
import json
import os
import re
import stat
import subprocess
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from rag_studio import colab, remote

ROOT = Path(__file__).resolve().parent.parent
START = ROOT / "tools" / "colab" / "start.sh"


# --------------------------------------------------------------------------- the connect code

def test_a_code_round_trips():
    code = remote.encode_connect_code("https://a-b.trycloudflare.com/", "t" * 48,
                                      models=["qwen3-embedding:8b", "qwen3-embedding:0.6b"], gateway="1")
    assert code.startswith("rsc1.")
    d = remote.decode_connect_code(code)
    assert d["url"] == "https://a-b.trycloudflare.com"
    assert d["token"] == "t" * 48
    assert d["models"] == ["qwen3-embedding:8b", "qwen3-embedding:0.6b"]
    assert d["gateway"] == "1" and isinstance(d["issued_at"], int)


def test_a_wrapped_paste_still_decodes():
    code = remote.encode_connect_code("https://x.trycloudflare.com", "a" * 48)
    wrapped = "\n".join(code[i:i + 20] for i in range(0, len(code), 20))
    assert remote.decode_connect_code("  " + wrapped + "\n")["token"] == "a" * 48


@pytest.mark.parametrize("bad, because", [
    ("https://x.trycloudflare.com", "not a connect code"),
    ("rsc1.", "empty"),
    ("rsc1.!!!!", "damaged"),
    ("rsc1." + base64.urlsafe_b64encode(b"not json").decode().rstrip("="), "damaged"),
    ("rsc1." + base64.urlsafe_b64encode(b'["a"]').decode().rstrip("="), "does not hold"),
    ("rsc1." + base64.urlsafe_b64encode(b'{"t": "x"}').decode().rstrip("="), "no endpoint"),
    ("rsc1." + base64.urlsafe_b64encode(b'{"u": "ftp://x"}').decode().rstrip("="), "not a web address"),
])
def test_every_bad_paste_is_refused_with_a_reason(bad, because):
    with pytest.raises(ValueError, match=re.compile(because, re.IGNORECASE)):
        remote.decode_connect_code(bad)


def test_a_truncated_code_is_refused():
    code = remote.encode_connect_code("https://x.trycloudflare.com", "a" * 48)
    with pytest.raises(ValueError):
        remote.decode_connect_code(code[:-7])


def test_an_absurdly_long_string_is_refused():
    with pytest.raises(ValueError, match="far longer"):
        remote.decode_connect_code("rsc1." + "A" * 5000)


def _bash_code(*args: str) -> str:
    out = subprocess.run(["bash", "-c", f'RAG_COLAB_LIB=1 source "{START}"; connect_code "$@"', "_", *args],
                         capture_output=True, text=True, timeout=30, check=True)
    return out.stdout.strip()


def test_what_start_sh_prints_is_what_the_laptop_reads():
    code = _bash_code("https://q.trycloudflare.com/", "f" * 48, "qwen3-embedding:8b qwen3-embedding:0.6b", "1")
    d = remote.decode_connect_code(code)
    assert d == {**d, "url": "https://q.trycloudflare.com", "token": "f" * 48,
                 "models": ["qwen3-embedding:8b", "qwen3-embedding:0.6b"], "gateway": "1"}


def test_start_sh_omits_what_it_was_not_given():
    d = remote.decode_connect_code(_bash_code("https://q.trycloudflare.com", "", "", ""))
    assert d["token"] is None and d["models"] == [] and d["gateway"] == ""


def test_the_gateway_version_comes_from_server_py():
    assert colab.gateway_version() == re.search(r'^VERSION = "(.*)"', (ROOT / "tools/colab/server.py").read_text(),
                                                re.M).group(1)


# --------------------------------------------------------------------------- the private files

def _mode(path: Path) -> int:
    return stat.S_IMODE(path.stat().st_mode)


def test_the_state_file_is_private_and_atomic(private_state):
    remote.save_state({"url": "https://x.trycloudflare.com", "token": "s" * 48})
    path = remote.state_path()
    assert path.parent == private_state and _mode(path) == 0o600
    assert not list(path.parent.glob(".*.tmp"))
    assert remote.load_state()["token"] == "s" * 48


def test_a_corrupt_state_file_is_no_connection():
    remote.state_path().parent.mkdir(parents=True, exist_ok=True)
    remote.state_path().write_text("{not json")
    assert remote.load_state() == {} and not remote.connected()


def test_the_remembered_token_is_private_and_forgettable():
    assert remote.remembered_token() is None
    token = remote.mint_token()
    assert re.fullmatch(r"[0-9a-f]{48}", token)
    remote.remember_token(token)
    assert remote.remembered_token() == token and _mode(remote.token_path()) == 0o600
    assert remote.forget_token() and remote.remembered_token() is None and not remote.forget_token()


def test_the_connection_never_carries_the_token():
    remote.save_state({"url": "https://x.trycloudflare.com", "token": "SECRET" * 8, "models": ["m"]})
    remote.remember_token("REMEMBERED" * 5)
    text = json.dumps(remote.connection())
    assert "SECRET" not in text and "REMEMBERED" not in text
    assert remote.connection()["authenticated"] is True


def test_a_failure_is_recorded_and_success_clears_it():
    remote.save_state({"url": "https://x.trycloudflare.com", "token": "t" * 48})
    remote.record_result(ok=False, error="boom\nsecond line", kind="unreachable")
    s = remote.load_state()
    assert s["last_error"] == "boom" and s["last_error_kind"] == "unreachable"
    remote.record_result(ok=True)
    s = remote.load_state()
    assert "last_error" not in s and s["last_ok_at"]


def test_recording_never_creates_a_state_file():
    remote.record_result(ok=False, error="x")
    assert not remote.state_path().exists()


# --------------------------------------------------------------------------- connect checks

def test_plain_http_off_this_machine_is_refused_before_saving():
    with pytest.raises(ValueError, match="https"):
        remote.connect(url="http://example.com", token="t" * 48)
    assert not remote.connected()


def test_a_url_with_no_token_anywhere_is_refused():
    with pytest.raises(ValueError, match="No token"):
        remote.connect(url="https://x.trycloudflare.com")
    assert not remote.connected()


def test_a_bare_url_uses_the_remembered_token(monkeypatch):
    remote.remember_token("r" * 48)
    monkeypatch.setattr(remote, "smoke_test", lambda c=None: {"ok": False, "stage": "connect",
                                                              "error": "x", "error_kind": "unreachable",
                                                              "seconds": 0, "embedded": []})
    result = remote.connect(url="https://x.trycloudflare.com")
    assert not result["ok"] and remote.load_state()["token"] == "r" * 48   # saved even though the test failed


def test_not_connected_says_how_to_connect():
    with pytest.raises(remote.RemoteNotConnected, match="Settings > Model backend"):
        remote.client()
    assert remote.probe()["error_kind"] == "not_connected"


# --------------------------------------------------------------------------- end to end through the gateway

@pytest.fixture
def runtime():
    """The real gateway on loopback in front of a stub Ollama with one
    embedding model: a Colab runtime without the GPU."""
    class Ollama(BaseHTTPRequestHandler):
        def log_message(self, format, *args):  # noqa: A002
            pass

        def _reply(self, obj):
            body = json.dumps(obj).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):  # noqa: N802
            if self.path == "/api/tags":
                return self._reply({"models": [{"name": "qwen3-embedding:8b", "digest": "sha256:8b",
                                                "size": 4, "details": {"quantization_level": "Q4_K_M"},
                                                "capabilities": ["embedding"]}]})
            if self.path == "/api/ps":
                return self._reply({"models": [{"name": "qwen3-embedding:8b", "size": 4, "size_vram": 4}]})
            return self._reply({"version": "0.32.6"})

        def do_POST(self):  # noqa: N802
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            return self._reply({"embeddings": [[0.0] * 4095 + [1.0] for _ in body["input"]]})

    up = ThreadingHTTPServer(("127.0.0.1", 0), Ollama)
    threading.Thread(target=up.serve_forever, daemon=True).start()
    spec = importlib.util.spec_from_file_location("gw", ROOT / "tools/colab/server.py")
    assert spec and spec.loader
    gw = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gw)
    token = remote.mint_token()
    gw.Config.ollama, gw.Config.token, gw.Config.require_token = f"http://127.0.0.1:{up.server_address[1]}", token, True
    gw.Handler._done = lambda *a, **k: None      # its log is tested in test_colab_gateway.py
    door = ThreadingHTTPServer(("127.0.0.1", 0), gw.Handler)
    threading.Thread(target=door.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{door.server_address[1]}", token
    door.shutdown()
    up.shutdown()


def test_connect_test_probe_and_disconnect(runtime):
    url, token = runtime
    code = remote.encode_connect_code(url, token, models=["qwen3-embedding:8b"], gateway="1")
    result = remote.connect(code=code)
    assert result["ok"], result
    assert result["embedded"][0] == {**result["embedded"][0], "model": "qwen3-embedding:8b", "dimensions": 4096}
    assert result["missing"] == []          # the 8B profiles need only the 8B there
    assert remote.remembered_token() == token
    info = remote.probe()
    assert info["reachable"] and info["loaded"][0]["on_gpu"]
    assert remote.connection()["last_ok_at"]
    remote.disconnect()
    assert not remote.connected() and remote.remembered_token() == token
    remote.disconnect(forget=True)
    assert remote.remembered_token() is None


def test_a_wrong_token_is_reported_as_auth(runtime):
    url, _ = runtime
    result = remote.connect(code=remote.encode_connect_code(url, "w" * 48))
    assert not result["ok"] and result["error_kind"] == "auth"
    assert remote.connection()["last_error_kind"] == "auth"


# --------------------------------------------------------------------------- the setup cell

@pytest.fixture(scope="module")
def cell() -> str:
    return colab.build_cell(models=["qwen3-embedding:8b"], ollama_version="0.32.6")


def _section(cell: str) -> str:
    """The cell up to (not including) running start.sh: decode, check, write."""
    return cell.split("# Streamed, so the seven checks", 1)[0]


def test_it_carries_exactly_the_two_files_and_they_reconstruct(cell, tmp_path):
    os.chdir(tmp_path)
    exec(compile(_section(cell), "cell", "exec"), {})
    for name in colab.PAYLOAD_FILES:
        assert (tmp_path / name).read_bytes() == (ROOT / "tools/colab" / name).read_bytes()
    assert sorted(p.name for p in tmp_path.iterdir()) == sorted(colab.PAYLOAD_FILES)


def test_a_damaged_paste_writes_nothing(cell, tmp_path):
    os.chdir(tmp_path)
    blob = re.search(r'"start\.sh": \(\n\s+"([^"]+)"', cell).group(1)
    damaged = cell.replace(blob, blob[:-8] + "AAAAAAAA", 1)
    with pytest.raises(SystemExit, match="did not survive the paste"):
        exec(compile(_section(damaged), "cell", "exec"), {})
    assert not list(tmp_path.iterdir())


def test_the_same_source_makes_the_same_cell():
    a = colab.build_cell(models=["m:1"], ollama_version="0.32.6")
    b = colab.build_cell(models=["m:1"], ollama_version="0.32.6")
    assert a == b


def test_the_tokenless_cell_carries_no_credential(cell):
    assert 'os.environ["RAG_STUDIO_TOKEN"]' not in cell
    # No 48-hex run (a token's shape) anywhere; the 64-hex digests do not count.
    assert not re.search(r"(?<![0-9a-f])[0-9a-f]{48}(?![0-9a-f])", cell)
    assert "this runtime will make its own" in cell


def test_the_token_cell_exports_the_name_start_sh_reads():
    token = "c" * 48
    cell = colab.build_cell(models=["m:1"], token=token)
    assert f'os.environ["RAG_STUDIO_TOKEN"] = "{token}"' in cell
    assert "RAG_STUDIO_TOKEN" in START.read_text()


def test_models_and_the_ollama_pin_reach_start_sh(cell):
    assert "MODELS = 'qwen3-embedding:8b'" in cell and "OLLAMA_VERSION = '0.32.6'" in cell
    assert '"--models", MODELS' in cell and '"--ollama-version", OLLAMA_VERSION' in cell


@pytest.mark.parametrize("models", [[], ["bad name; rm -rf /"], ["x" * 100]])
def test_a_bad_model_name_never_reaches_the_cell(models):
    with pytest.raises(ValueError):
        colab.build_cell(models=models)


def test_a_bad_ollama_version_never_reaches_the_cell():
    with pytest.raises(ValueError):
        colab.build_cell(ollama_version="0.32; curl evil")


def test_the_payload_is_valid_bash_and_python(tmp_path):
    for name, data in colab.payload().items():
        raw = gzip.decompress(base64.b64decode(data["b64"]))
        assert hashlib.sha256(raw).hexdigest() == data["sha256"]
    assert subprocess.run(["bash", "-n", str(START)]).returncode == 0
    compile((ROOT / "tools/colab/server.py").read_text(), "server.py", "exec")


def test_the_cell_is_small_enough_to_paste(cell):
    stats = colab.cell_stats(cell)
    assert 5 < stats["kilobytes"] < 60
    assert max(len(line) for line in cell.splitlines()) < 140


# --------------------------------------------------------------------------- start.sh helpers

def _sh(script: str) -> subprocess.CompletedProcess:
    return subprocess.run(["bash", "-c", f'RAG_COLAB_LIB=1 source "{START}"; {script}'],
                          capture_output=True, text=True, timeout=60)


def test_sourcing_start_sh_starts_nothing():
    r = _sh("echo sourced")
    assert r.returncode == 0 and r.stdout.strip() == "sourced"


@pytest.mark.parametrize("code, verdict, rc", [("401", "locked", 0), ("403", "locked", 0), ("200", "OPEN", 2),
                                               ("000", "unreachable", 1), ("502", "unexpected:502", 1)])
def test_each_tunnel_probe_is_classified(code, verdict, rc):
    r = _sh(f"tunnel_verdict {code}")
    assert r.stdout == verdict and r.returncode == rc


def test_a_refused_connection_is_exactly_000():
    assert _sh("http_code http://127.0.0.1:9/").stdout == "000"


def test_gpu_residency_matches_the_exact_tag(tmp_path):
    """qwen3-embedding:0.6b and :8b share a family; trading's prefix match
    would report the wrong one."""
    ps = {"models": [{"name": "qwen3-embedding:0.6b", "size": 100, "size_vram": 0},
                     {"name": "qwen3-embedding:8b", "size": 400, "size_vram": 400}]}
    fake = tmp_path / "curl"
    fake.write_text(f"#!/bin/sh\necho '{json.dumps(ps)}'\n")
    fake.chmod(0o755)
    env = {**os.environ, "PATH": f"{tmp_path}:{os.environ['PATH']}"}
    for tag, want in (("qwen3-embedding:8b", "400 400"), ("qwen3-embedding:0.6b", "0 100"), ("other:1b", "")):
        r = subprocess.run(["bash", "-c", f'RAG_COLAB_LIB=1 source "{START}"; gpu_residency {tag}'],
                           capture_output=True, text=True, env=env, timeout=30)
        assert r.stdout.strip() == want, tag


def test_the_lock_is_proved_on_loopback_before_the_tunnel_starts():
    text = START.read_text()
    assert text.index('LOCAL_UNAUTH="$(http_code') < text.index("nohup cloudflared")
    assert "Refusing to expose it" in text


def test_the_token_is_never_printed_by_start_sh():
    for line in START.read_text().splitlines():
        if re.search(r"\b(echo|printf|cat)\b", line) and "$TOKEN" in line and "TOKEN_FILE" not in line:
            pytest.fail(f"start.sh may print the token: {line.strip()}")


def test_an_open_tunnel_is_shut_down_without_printing_the_url():
    block = START.read_text().split("    OPEN)", 1)[1].split(";;", 1)[0]
    assert "pkill -f cloudflared" in block and "exit 1" in block and "$URL_FILE" in block


def test_waiting_gives_up_at_the_deadline():
    t = time.monotonic()
    r = _sh("wait_for_http http://127.0.0.1:9/ 3 200")
    assert r.stdout == "000" and time.monotonic() - t < 15
