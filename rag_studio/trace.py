"""The in-app trace: what happened while answering one question.

Each finished step is recorded with its duration and a few numbers, in the
order it ran. The Ask page's "What happened" panel and every evaluation run
read this, so the timings a user sees are the ones that were measured.

A Phoenix export (OpenTelemetry, loopback only) was planned for M6 and cut
(2026-10-22, docs/plans/m6-m8.md): these steps are the only trace.
"""
from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class Step:
    id: str               # stable key: embed, search, prompt, generate
    label: str            # plain-language past tense: "Searched 1,190 passages"
    ms: float
    detail: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["ms"] = round(self.ms, 1)
        return d


class Trace:
    def __init__(self) -> None:
        self.steps: list[Step] = []
        self._t0 = time.perf_counter()

    def add(self, id: str, label: str, ms: float, **detail: Any) -> Step:
        step = Step(id, label, ms, detail)
        self.steps.append(step)
        return step

    @property
    def elapsed_ms(self) -> float:
        return (time.perf_counter() - self._t0) * 1000

    def as_list(self) -> list[dict[str, Any]]:
        return [s.as_dict() for s in self.steps]
