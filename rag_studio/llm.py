"""The one place that talks to Ollama.

Local Ollama and the Colab gateway (M4) speak the same API, so one client
serves both; the gateway only adds a bearer token. Every call has an explicit
timeout, because a model swap on this shared CPU can stall a request for
minutes and nothing should hang waiting on it silently.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import httpx

from rag_studio.config import OLLAMA_URL

# Status calls are cheap when Ollama is idle. A short timeout keeps Health
# responsive when it is busy loading a model for another app.
STATUS_TIMEOUT = httpx.Timeout(3.0, connect=1.0)


class OllamaUnavailable(RuntimeError):
    """Ollama did not answer. `reason` is short and safe to show."""

    def __init__(self, reason: str):
        super().__init__(reason)
        self.reason = reason


@dataclass(frozen=True)
class InstalledModel:
    name: str
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

    def version(self) -> str:
        return str(self._get("/api/version").get("version", ""))

    def installed(self) -> list[InstalledModel]:
        models = []
        for m in self._get("/api/tags").get("models", []):
            d = m.get("details") or {}
            models.append(InstalledModel(
                name=m.get("name", ""),
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


def same_model(a: str, b: str) -> bool:
    """`nomic-embed-text` and `nomic-embed-text:latest` are one model."""
    def norm(name: str) -> str:
        return name if ":" in name else f"{name}:latest"
    return norm(a) == norm(b)
