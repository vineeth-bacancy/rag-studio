"""The one place that talks to Ollama.

Local Ollama and the Colab gateway (M4) speak the same API, so one client
serves both; the gateway only adds a bearer token. Every call has an explicit
timeout, because a model swap on this shared CPU can stall a request for
minutes and nothing should hang waiting on it silently.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Iterator

import httpx

from rag_studio.config import OLLAMA_URL

# Status calls are cheap when Ollama is idle. A short timeout keeps Health
# responsive when it is busy loading a model for another app.
STATUS_TIMEOUT = httpx.Timeout(3.0, connect=1.0)
# Model calls can wait minutes: a cold load, or a swap with the trading app's
# qwen3:8b (an embed once took 257 s). `read` is the gap between two bytes, so
# a streamed answer only needs it to cover the prompt-reading pause.
MODEL_TIMEOUT = httpx.Timeout(connect=2.0, read=600.0, write=60.0, pool=5.0)


class OllamaUnavailable(RuntimeError):
    """Ollama did not answer. `reason` is short and safe to show."""

    def __init__(self, reason: str):
        super().__init__(reason)
        self.reason = reason


@dataclass(frozen=True)
class InstalledModel:
    name: str
    digest: str | None
    size: int
    family: str | None
    parameter_size: str | None
    quantization: str | None
    context_length: int | None
    capabilities: tuple[str, ...]
    modified_at: str | None


@dataclass(frozen=True)
class ResidentModel:
    name: str
    size: int
    size_vram: int
    expires_at: str | None
    context_length: int | None


class OllamaClient:
    def __init__(self, base_url: str = OLLAMA_URL, token: str | None = None):
        self.base_url = base_url.rstrip("/")
        self._headers = {"Authorization": f"Bearer {token}"} if token else {}

    def _get(self, path: str) -> Any:
        try:
            r = httpx.get(f"{self.base_url}{path}", headers=self._headers, timeout=STATUS_TIMEOUT)
        except httpx.ConnectError:
            raise OllamaUnavailable("Ollama is not running.") from None
        except httpx.TimeoutException:
            raise OllamaUnavailable("Ollama did not answer in time; it may be loading a model.") from None
        except httpx.HTTPError as e:
            raise OllamaUnavailable(f"Ollama request failed ({type(e).__name__}).") from None
        if r.status_code != 200:
            raise OllamaUnavailable(f"Ollama answered {path} with HTTP {r.status_code}.")
        return r.json()

    def _post(self, path: str, body: dict[str, Any]) -> Any:
        try:
            r = httpx.post(f"{self.base_url}{path}", json=body, headers=self._headers,
                           timeout=MODEL_TIMEOUT)
        except httpx.ConnectError:
            raise OllamaUnavailable("Ollama is not running.") from None
        except httpx.TimeoutException:
            raise OllamaUnavailable("Ollama stopped answering; it may be busy with another app.") from None
        except httpx.HTTPError as e:
            raise OllamaUnavailable(f"Ollama request failed ({type(e).__name__}).") from None
        if r.status_code != 200:
            raise OllamaError(r.status_code, _error_text(r.text))
        return r.json()

    def version(self) -> str:
        return str(self._get("/api/version").get("version", ""))

    def installed(self) -> list[InstalledModel]:
        models = []
        for m in self._get("/api/tags").get("models", []):
            d = m.get("details") or {}
            models.append(InstalledModel(
                name=m.get("name", ""),
                digest=m.get("digest"),
                size=int(m.get("size") or 0),
                family=d.get("family"),
                parameter_size=d.get("parameter_size"),
                quantization=d.get("quantization_level"),
                context_length=d.get("context_length"),
                capabilities=tuple(m.get("capabilities") or ()),
                modified_at=m.get("modified_at"),
            ))
        return sorted(models, key=lambda m: m.name)

    def resident(self) -> list[ResidentModel]:
        return [
            ResidentModel(
                name=m.get("name", ""),
                size=int(m.get("size") or 0),
                size_vram=int(m.get("size_vram") or 0),
                expires_at=m.get("expires_at"),
                context_length=m.get("context_length"),
            )
            for m in self._get("/api/ps").get("models", [])
        ]


    def model_info(self, name: str) -> InstalledModel | None:
        """The installed model with this name, or None. Tags compare loosely
        (`x` is `x:latest`); the digest is what proves two are the same."""
        return next((m for m in self.installed() if same_model(m.name, name)), None)

    def embed(self, model: str, texts: list[str], *, dimensions: int | None = None,
              keep_alive: str | int | None = None) -> list[list[float]]:
        """One batch through `/api/embed`.

        `truncate: false` on purpose: by default Ollama silently cuts an input
        longer than the model's context, and the vector then describes only the
        start of the passage. An error we can see beats a vector that lies.
        """
        body: dict[str, Any] = {"model": model, "input": texts, "truncate": False}
        if dimensions is not None:
            body["dimensions"] = dimensions
        if keep_alive is not None:
            body["keep_alive"] = keep_alive
        out = self._post("/api/embed", body).get("embeddings") or []
        if len(out) != len(texts):
            raise OllamaError(200, f"asked for {len(texts)} embeddings, got {len(out)}")
        return out

    def chat_stream(self, model: str, messages: list[dict[str, str]], *,
                    options: dict[str, Any], think: bool = False,
                    keep_alive: str | int | None = None) -> Iterator[dict[str, Any]]:
        """`/api/chat` with `stream: true`, one parsed line at a time.

        `think` is always sent. Qwen 3.5 thinks unless told not to, Gemma 4 the
        opposite, and qwen3:4b here ignores `false` entirely (spec §2.3), so
        the caller must never rely on a model's default.

        Closing the generator early closes the connection, which is how Ollama
        learns to stop generating for a reader who has gone.
        """
        body: dict[str, Any] = {"model": model, "messages": messages, "stream": True,
                                "think": think, "options": options}
        if keep_alive is not None:
            body["keep_alive"] = keep_alive
        try:
            with httpx.stream("POST", f"{self.base_url}/api/chat", json=body,
                              headers=self._headers, timeout=MODEL_TIMEOUT) as r:
                if r.status_code != 200:
                    raise OllamaError(r.status_code, _error_text(r.read().decode("utf-8", "replace")))
                for line in r.iter_lines():
                    if not line.strip():
                        continue
                    data = json.loads(line)
                    if data.get("error"):
                        raise OllamaError(200, str(data["error"]))
                    yield data
        except httpx.ConnectError:
            raise OllamaUnavailable("Ollama is not running.") from None
        except httpx.TimeoutException:
            raise OllamaUnavailable("Ollama stopped answering; it may be busy with another app.") from None
        except httpx.HTTPError as e:
            raise OllamaUnavailable(f"Ollama request failed ({type(e).__name__}).") from None


class OllamaError(RuntimeError):
    """Ollama answered, with an error. `message` is Ollama's own text."""

    def __init__(self, status: int, message: str):
        super().__init__(f"Ollama returned HTTP {status}: {message}")
        self.status = status
        self.message = message


def _error_text(body: str) -> str:
    try:
        return str(json.loads(body).get("error") or body)[:300]
    except (ValueError, AttributeError):
        return body[:300]


def same_model(a: str, b: str) -> bool:
    """`nomic-embed-text` and `nomic-embed-text:latest` are one model."""
    def norm(name: str) -> str:
        return name if ":" in name else f"{name}:latest"
    return norm(a) == norm(b)
