"""Stand-ins for Ollama, so the engine can be tested without a model running."""
from __future__ import annotations

import hashlib
import random
from typing import Any, Iterator

from rag_studio.llm import InstalledModel


def fake_vector(text: str, dim: int = 768) -> list[float]:
    """Deterministic per text: the same passage always gets the same vector."""
    rng = random.Random(hashlib.sha256(text.encode()).digest())
    return [rng.gauss(0, 1) for _ in range(dim)]


class FakeOllama:
    """Answers `installed`, `embed`, `chat_stream` and `version` like Ollama."""

    def __init__(self, *, digest: str = "sha256:aaaa", answer: str = "It is 166 [1].",
                 models: tuple[str, ...] = ("nomic-embed-text:latest", "qwen3:1.7b"), dim: int = 768,
                 noise: float = 0.0, fail_after: int | None = None, base_url: str = "http://fake"):
        self.digest = digest
        self.answer_text = answer
        self.models = models
        self.dim = dim
        self.noise = noise              # per-component noise: a different machine's float error
        self.fail_after = fail_after    # raise on this embed call (1-based), like a reclaimed runtime
        self.base_url = base_url
        self.embed_calls = 0
        self.embed_options: list[Any] = []
        self.chats: list[dict[str, Any]] = []

    def version(self) -> str:
        return "0.0-test"

    def installed(self) -> list[InstalledModel]:
        return [InstalledModel(name=n, digest=self.digest, size=1, family=None,
                               parameter_size="1B", quantization="F16", context_length=2048,
                               capabilities=("embedding",) if "embed" in n else ("completion",),
                               modified_at=None) for n in self.models]

    def model_info(self, name: str) -> InstalledModel | None:
        return next((m for m in self.installed() if m.name == name), None)

    def resident(self) -> list:
        return []

    def gpu(self) -> dict[str, Any]:
        return {"available": True, "gpus": [{"name": "Tesla T4", "memory_total_mb": 15360,
                                              "memory_used_mb": 1000, "utilization_pct": 0}]}

    def embed(self, model: str, texts: list[str], **kw: Any) -> list[list[float]]:
        self.embed_calls += 1
        self.embed_options.append(kw.get("options"))
        if self.fail_after is not None and self.embed_calls >= self.fail_after:
            from rag_studio.llm import OllamaUnavailable
            raise OllamaUnavailable("The Colab runtime is not answering; Colab may have reclaimed it.")
        out = [fake_vector(t.split(": ", 1)[-1], self.dim) for t in texts]
        if self.noise:
            rng = random.Random(self.embed_calls)
            out = [[x + rng.gauss(0, self.noise) for x in v] for v in out]
        return out

    def chat_stream(self, model: str, messages: list[dict[str, str]], *, options: dict[str, Any],
                    think: bool = False, **_: Any) -> Iterator[dict[str, Any]]:
        self.chats.append({"model": model, "messages": messages, "options": options, "think": think})
        words = self.answer_text.split(" ")
        for i, w in enumerate(words):
            yield {"message": {"content": w + (" " if i < len(words) - 1 else "")}, "done": False}
        yield {"message": {"content": ""}, "done": True, "done_reason": "stop",
               "prompt_eval_count": 1350, "eval_count": len(words),
               "load_duration": 1_000_000, "prompt_eval_duration": 2_000_000_000,
               "eval_duration": 1_000_000_000}
