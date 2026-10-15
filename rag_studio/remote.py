"""Remote AI: a Colab GPU runtime the user starts, reached through its gateway.

The runtime is a computation backend, not a second app server. What crosses
the network is a batch of passages to embed (or, from M5, to rerank) and the
vectors that come back. No file leaves this machine: not the PDFs, not the
index, not the repository. The runtime keeps nothing it is sent.

Adapted from the trading app's design (docs/02-remote-gpu.md):

* **The connection lives outside the repository**, in
  `~/.local/state/rag-studio/remote.json` (mode 0600, written atomically).
  Nothing under `data/` or `web/` ever holds the token, and no API response
  carries it: a credential travels browser -> API, never API -> browser.
* **The token the Colab cell was built with is remembered separately**
  (`colab-token`), so after Colab reclaims a runtime, re-running the same cell
  and pasting the new URL is the whole reconnect.
* **The connect code** (`rsc1.` + base64url JSON) carries the URL, the token,
  the models the runtime serves and the gateway version as one string, so a
  half-copied pair cannot happen. It is NOT encryption.
* **Strict by default.** Nothing falls back to another model or machine
  silently; an index that needs the runtime refuses when it is gone (rule 3).

Connect checks run in this order, here and nowhere else (the CLI and the API
both call `connect`): decode -> https for a non-loopback host -> a token ->
save -> a real embedding request -> the models the Colab profiles need.
"""
from __future__ import annotations

import base64
import binascii
import json
import os
import secrets
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from rag_studio import paths
from rag_studio.llm import OllamaClient, OllamaError, OllamaUnavailable, same_model

CODE_PREFIX = "rsc1."
MAX_CODE_CHARS = 4096
MAX_URL_CHARS = 300
TOKEN_CHARS = (16, 256)

STATE_FILE = "remote.json"
TOKEN_FILE = "colab-token"


class RemoteNotConnected(RuntimeError):
    """No Colab runtime is connected. The message says how to connect one."""


# --------------------------------------------------------------------------- files

def state_path() -> Path:
    return paths.STATE / STATE_FILE


def token_path() -> Path:
    return paths.STATE / TOKEN_FILE


def _write_private(path: Path, text: str) -> None:
    """Atomic, and readable by this user only from the first byte: the
    temporary file is created 0600 (mkstemp), then renamed over the target."""
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(text)
            f.flush()
            os.fsync(f.fileno())
        os.chmod(tmp, 0o600)
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


def load_state() -> dict[str, Any]:
    """The connected runtime, or {}. A corrupt file is no file."""
    try:
        data = json.loads(state_path().read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) and data.get("url") else {}


def save_state(state: dict[str, Any]) -> None:
    _write_private(state_path(), json.dumps(state, indent=2) + "\n")


def clear_state() -> bool:
    try:
        state_path().unlink()
        return True
    except FileNotFoundError:
        return False


def remembered_token() -> str | None:
    """The token embedded in the last Colab cell this laptop generated."""
    try:
        token = token_path().read_text(encoding="utf-8").strip()
    except OSError:
        return None
    return token or None


def remember_token(token: str) -> None:
    _write_private(token_path(), token.strip())


def forget_token() -> bool:
    try:
        token_path().unlink()
        return True
    except FileNotFoundError:
        return False


def mint_token() -> str:
    """48 hex characters, the width start.sh makes on the runtime."""
    return secrets.token_hex(24)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def record_result(*, ok: bool, error: str = "", kind: str = "") -> None:
    """Note the outcome of a real request, so status can answer "what went
    wrong last time" after the laptop slept. Never creates the state file."""
    state = load_state()
    if not state:
        return
    if ok:
        state["last_ok_at"] = _now()
        for k in ("last_error", "last_error_at", "last_error_kind"):
            state.pop(k, None)
    else:
        state["last_error"] = (str(error).splitlines() or ["failed"])[0][:300]
        state["last_error_at"] = _now()
        state["last_error_kind"] = kind or "error"
    try:
        save_state(state)
    except OSError:
        pass


# --------------------------------------------------------------------------- the connect code

def encode_connect_code(url: str, token: str | None, *, models: list[str] | None = None,
                        gateway: str = "") -> str:
    """One pasteable string. Mirrored by `connect_code` in tools/colab/start.sh."""
    payload: dict[str, Any] = {"u": url.strip().rstrip("/")}
    if token:
        payload["t"] = token.strip()
    if models:
        payload["m"] = [m.strip() for m in models if m.strip()]
    if gateway:
        payload["g"] = gateway.strip()
    payload["ts"] = int(time.time())
    blob = base64.urlsafe_b64encode(json.dumps(payload, separators=(",", ":")).encode()).decode()
    return CODE_PREFIX + blob.rstrip("=")


def looks_like_code(text: str) -> bool:
    return text.strip().startswith(CODE_PREFIX)


def decode_connect_code(text: str) -> dict[str, Any]:
    """{url, token, models, gateway, issued_at}, or ValueError naming what went
    wrong with the paste."""
    raw = "".join(text.split())                       # a wrapped paste loses its line breaks
    if not raw.startswith(CODE_PREFIX):
        raise ValueError(f"That is not a connect code. A connect code starts with {CODE_PREFIX}")
    blob = raw[len(CODE_PREFIX):]
    if not blob:
        raise ValueError("The connect code is empty after its prefix. Copy the whole line again.")
    if len(raw) > MAX_CODE_CHARS:
        raise ValueError("That is far longer than a connect code. Copy just the code.")
    try:
        # A strict round trip: b64decode silently skips characters outside the
        # alphabet, which would turn a mangled paste into a confusing JSON error.
        decoded = base64.urlsafe_b64decode(blob + "=" * (-len(blob) % 4))
        if base64.urlsafe_b64encode(decoded).decode().rstrip("=") != blob.rstrip("="):
            raise binascii.Error("not base64")
    except (ValueError, binascii.Error):
        raise ValueError("The connect code is damaged: it was probably copied incompletely. "
                         "Copy the whole line again.") from None
    try:
        data = json.loads(decoded)
    except (ValueError, UnicodeDecodeError):
        raise ValueError("The connect code is damaged: it was probably copied incompletely. "
                         "Copy the whole line again.") from None
    if not isinstance(data, dict):
        raise ValueError("The connect code does not hold an endpoint. Copy the whole line again.")
    url = str(data.get("u") or "").strip().rstrip("/")
    if not url:
        raise ValueError("The connect code carries no endpoint URL.")
    if not url.startswith(("http://", "https://")):
        raise ValueError("The connect code's endpoint is not a web address.")
    token = data.get("t")
    models = data.get("m")
    if isinstance(models, str):
        models = models.split()
    issued = data.get("ts")
    return {
        "url": url,
        "token": token.strip() if isinstance(token, str) and token.strip() else None,
        "models": [str(m) for m in models] if isinstance(models, list) else [],
        "gateway": str(data.get("g") or "").strip(),
        "issued_at": int(issued) if isinstance(issued, (int, float)) else None,
    }


# --------------------------------------------------------------------------- transport

def host_of(url: str) -> str:
    try:
        return urlparse(url).netloc or url
    except ValueError:
        return "?"


def is_loopback(url: str) -> bool:
    host = (urlparse(url).hostname or "").lower()
    return host in ("127.0.0.1", "localhost", "::1")


def check_transport(url: str) -> None:
    """Passages and questions are the user's data: never send them in clear
    text off this machine. Loopback is allowed (that is how the gateway is
    tested before a tunnel exists); anything else must be https."""
    scheme = (urlparse(url).scheme or "").lower()
    if is_loopback(url) or scheme == "https":
        return
    raise ValueError(f"Refusing to send passages to {host_of(url)} over {scheme or 'an unknown scheme'}. "
                     "A remote runtime must use https, as the Colab tunnel does.")


# --------------------------------------------------------------------------- the connection

def connected() -> bool:
    return bool(load_state())


def client() -> OllamaClient:
    """A client for the connected runtime, or RemoteNotConnected."""
    state = load_state()
    if not state:
        raise RemoteNotConnected("No Colab runtime is connected. Connect one in Settings > Model backend, "
                                 "or run: ./rag llm connect")
    return OllamaClient(state["url"], token=state.get("token"), remote=True)


def needed_models() -> list[str]:
    """The models the index profiles that embed on Colab need there."""
    from rag_studio import profiles
    return sorted({p.embedding.model for p in profiles.PROFILES.values()
                   if p.embedding.backend == "ollama" and "colab" in (p.embedding.build_on, p.embedding.query_on)})


def connection() -> dict[str, Any]:
    """The connection as the UI and CLI show it. Never carries the token."""
    state = load_state()
    out: dict[str, Any] = {
        "connected": bool(state),
        "backend": "remote" if state else "local",
        "token_remembered": remembered_token() is not None,
        "state_path": str(state_path()).replace(str(Path.home()), "~"),
        "code_prefix": CODE_PREFIX,
        "needed_models": needed_models(),
        "fallback": "strict",
    }
    if state:
        out.update({
            "url": state["url"], "host": host_of(state["url"]),
            "authenticated": bool(state.get("token")),
            "models": state.get("models") or [], "gateway": state.get("gateway"),
            "connected_at": state.get("connected_at"), "last_ok_at": state.get("last_ok_at"),
            "last_error": state.get("last_error"), "last_error_at": state.get("last_error_at"),
            "last_error_kind": state.get("last_error_kind"),
        })
    return out


def _kind(e: Exception) -> str:
    if isinstance(e, OllamaUnavailable):
        return e.kind
    if isinstance(e, OllamaError):
        return "model" if e.status == 404 or "not found" in e.message.lower() else "error"
    return "error"


def _message(e: Exception) -> str:
    if isinstance(e, OllamaUnavailable):
        return e.reason
    if isinstance(e, OllamaError):
        return f"The Colab runtime returned an error: {e.message}"
    return str(e)


def probe(c: OllamaClient | None = None) -> dict[str, Any]:
    """What the runtime has right now: Ollama's version, its models (with
    digests), what is loaded and on which device, and the GPU. No inference."""
    if c is None:
        try:
            c = client()
        except RemoteNotConnected as e:
            return {"reachable": False, "error": str(e), "error_kind": "not_connected"}
    t = time.perf_counter()
    try:
        version = c.version()
        latency_ms = round((time.perf_counter() - t) * 1000)
        installed = c.installed()
        resident = c.resident()
    except (OllamaUnavailable, OllamaError) as e:
        return {"reachable": False, "error": _message(e), "error_kind": _kind(e)}
    try:
        gpu = c.gpu()
    except (OllamaUnavailable, OllamaError):
        gpu = {"available": False, "reason": "this endpoint has no GPU probe"}
    loaded = [{"name": m.name, "size": m.size, "size_vram": m.size_vram,
               "on_gpu": m.size_vram > 0, "fully_on_gpu": m.size > 0 and m.size_vram >= m.size}
              for m in resident]
    have = [m.name for m in installed]
    missing = [m for m in needed_models() if not any(same_model(m, h) for h in have)]
    return {"reachable": True, "latency_ms": latency_ms, "ollama_version": version,
            "models": [{"name": m.name, "digest": m.digest, "size": m.size,
                        "quantization": m.quantization, "capabilities": list(m.capabilities)}
                       for m in installed],
            "loaded": loaded, "gpu": gpu, "missing": missing, "error": None, "error_kind": None}


def smoke_test(c: OllamaClient | None = None) -> dict[str, Any]:
    """A real request, not a reachability check: embed one sentence with each
    embedding model the runtime serves, and report dimensions and time. The
    outcome is recorded in the state file for `status`."""
    started = time.perf_counter()
    stage = "connect"
    try:
        c = c or client()
        stage = "list"
        installed = c.installed()
        embedders = [m for m in installed if "embedding" in m.capabilities]
        if not embedders:
            raise OllamaError(404, "the runtime serves no embedding model")
        stage = "embed"
        results = []
        for m in embedders:
            t = time.perf_counter()
            vec = c.embed(m.name, ["RAG Studio connection test."])[0]
            results.append({"model": m.name, "digest": m.digest, "dimensions": len(vec),
                            "seconds": round(time.perf_counter() - t, 2)})
    except RemoteNotConnected as e:
        return {"ok": False, "stage": stage, "error": str(e), "error_kind": "not_connected",
                "seconds": None, "embedded": []}
    except (OllamaUnavailable, OllamaError) as e:
        record_result(ok=False, error=_message(e), kind=_kind(e))
        return {"ok": False, "stage": stage, "error": _message(e), "error_kind": _kind(e),
                "seconds": round(time.perf_counter() - started, 2), "embedded": []}
    record_result(ok=True)
    return {"ok": True, "stage": "done", "error": None, "error_kind": None,
            "seconds": round(time.perf_counter() - started, 2), "embedded": results}


def connect(*, code: str = "", url: str = "", token: str = "") -> dict[str, Any]:
    """Point the app at a Colab runtime, then test it with a real request.
    Raises ValueError (nothing saved) for a bad code, URL, transport or a
    missing token. A failed test still leaves the connection saved: the next
    step is fixing the far side and testing again, not re-pasting."""
    code, url, token = code.strip(), url.strip(), token.strip()
    if len(code) > MAX_CODE_CHARS or len(url) > MAX_URL_CHARS or len(token) > TOKEN_CHARS[1]:
        raise ValueError("That is longer than any connect code, URL or token. Copy just the code.")
    models: list[str] = []
    gateway = ""
    if code:
        decoded = decode_connect_code(code)
        url, token = decoded["url"], decoded["token"] or token
        models, gateway = decoded["models"], decoded["gateway"]
    elif not url:
        raise ValueError("Paste the connect code (or the URL) the Colab cell printed.")
    else:
        url = url.rstrip("/")
        if not url.startswith(("http://", "https://")):
            raise ValueError(f"That is neither a URL nor a connect code. A connect code starts with {CODE_PREFIX}")
        token = token or (remembered_token() or "")
    check_transport(url)
    if not token:
        raise ValueError("No token for this runtime. Paste the whole connect code the Colab cell "
                         "showed, not just the URL.")
    if not (TOKEN_CHARS[0] <= len(token) <= TOKEN_CHARS[1]) or not token.isprintable() or " " in token:
        raise ValueError("That token is not one a runtime makes. Paste the whole connect code again.")

    save_state({"url": url, "token": token, "models": models, "gateway": gateway or None,
                "connected_at": _now(),
                "note": "Written by RAG Studio (./rag llm connect or Settings). A Colab tunnel dies "
                        "with its runtime: re-run the cell and connect again."})
    if token != remembered_token():
        remember_token(token)
    result = smoke_test()
    info = probe() if result["ok"] else None
    served = [m["name"] for m in (info or {}).get("models", [])] or models
    if info:
        # What the runtime really serves, for the Indexes page (no probe there).
        state = load_state()
        if state:
            state["models"] = served
            save_state(state)
    missing = [m for m in needed_models() if served and not any(same_model(m, s) for s in served)]
    return {**result, "missing": missing, "remote": info}


def disconnect(*, forget: bool = False) -> None:
    """Back to this computer. Deleting the state file is the whole switch:
    with no runtime connected, everything runs locally. `forget` also drops
    the remembered token, so the next cell must carry a new one."""
    clear_state()
    if forget:
        forget_token()
