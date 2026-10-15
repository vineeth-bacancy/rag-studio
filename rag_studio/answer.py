"""Generate and cite: answer from the retrieved passages, with citations built by code.

`ask()` is the one answering path. The Ask page streams it, `./rag eval` and
the Evaluation page consume the same generator, so an answer cannot be
produced one way in the browser and another way in a scorecard.

The event order is the contract the UI relies on:

    stage  one per step that FINISHED (never a prediction)
    ready  the sources and the prompt budget, before any model text
    delta  the model's text, in pieces
    done   the full answer, its citations, usage and the trace
    error  instead of done, with the stage that failed

**Citations are built by code** (rule 5). The model only writes `[n]`; this
module maps each n to the passage it was shown, and so to a document and a
1-based page. A number outside 1..k is reported as an invalid citation, never
silently dropped or "fixed".
"""
from __future__ import annotations

import re
import time
from dataclasses import asdict, dataclass
from typing import Any, Iterator

from rag_studio import guard, profiles
from rag_studio.config import OLLAMA_URL, role_model
from rag_studio.llm import OllamaClient, OllamaError, OllamaUnavailable
from rag_studio.prompt import (DEFAULT_VERSION, REFUSAL, build_messages, estimate_tokens,
                               looks_truncated, prompt_chars, prompt_sha256)
from rag_studio.retrieve import DENSE, Retriever, SearchSpec, retriever as active_retriever
from rag_studio.trace import Trace

Event = tuple[str, dict[str, Any]]


@dataclass(frozen=True)
class GenerationSpec:
    """How the answer model is called. Recorded with every answer and run.
    The baseline's settings: thinking off, temperature 0, an 8K context."""
    model: str
    think: bool = False
    temperature: float = 0.0
    num_ctx: int = 8192

    def options(self) -> dict[str, Any]:
        return {"temperature": self.temperature, "num_ctx": self.num_ctx}


DEFAULT_K = 5

_CITE = re.compile(r"\[([\d,\s]+)\]")


def cited_numbers(text: str) -> list[int]:
    """`[1]`, `[1][2]` and `[1, 2]` -> [1, 1, 2] in order of appearance."""
    nums: list[int] = []
    for group in _CITE.findall(text):
        nums += [int(n) for n in group.replace(",", " ").split()]
    return nums


def is_refusal(text: str) -> bool:
    """Strict on purpose (spec §5.7): a wrong fact followed by the refusal
    sentence is an answer, not a refusal."""
    return text.strip() == REFUSAL


def citations(text: str, sources: list[dict[str, Any]]) -> dict[str, Any]:
    nums = cited_numbers(text)
    valid = sorted({n for n in nums if 1 <= n <= len(sources)})
    invalid = sorted({n for n in nums if not 1 <= n <= len(sources)})
    return {
        "numbers": nums,
        "valid": valid,
        "invalid": invalid,
        "map": [{"n": n, "chunk_id": sources[n - 1]["chunk_id"], "doc_id": sources[n - 1]["doc_id"],
                 "page": sources[n - 1]["page"]} for n in valid],
    }


def _source(hit: dict[str, Any], n: int) -> dict[str, Any]:
    return {"n": n, "rank": hit["rank"], "chunk_id": hit["chunk_id"], "doc_id": hit["doc_id"],
            "page": hit["page"], "page_label": hit.get("page_label"),
            "section_path": hit.get("section_path"), "label": hit.get("label"), "type": hit["type"],
            "kind": hit.get("kind") or "body", "score": round(hit["score"], 4), "text": hit["text"],
            "char_start": hit["char_start"], "char_end": hit["char_end"],
            # Where it ranked in each list before fusion (hybrid) or the cap.
            "ranks": hit.get("ranks") or {}, "capped": bool(hit.get("capped")),
            # Checked live, so a passage from any index is screened the same way.
            "flags": guard.instruction_flags(hit["text"])}


_BY = {"dense": "by meaning", "bm25": "by words (BM25)", "hybrid": "by meaning and by words (hybrid)"}


def ask(question: str, *, k: int = DEFAULT_K, model: str | None = None,
        profile_id: str = profiles.DEFAULT, prompt_version: str = DEFAULT_VERSION,
        search: SearchSpec = DENSE,
        retriever: Retriever | None = None, client: OllamaClient | None = None) -> Iterator[Event]:
    question = question.strip()
    if not question:
        yield "error", {"stage": "input", "error": "Type a question first."}
        return
    gen = GenerationSpec(model=model or role_model("answer"))
    client = client or OllamaClient()
    trace = Trace()
    stage = "search"
    try:
        r = retriever or active_retriever(profile_id)
        hits, found = r.search(question, k, search)
        t = found["timings"]
        if "embed_ms" in t:
            trace.add("embed", "Embedded the question", t["embed_ms"],
                      model=r.profile.embedding.model, prefix=r.profile.embedding.query_prefix)
            yield "stage", trace.steps[-1].as_dict()
        search_ms = t.get("search_ms", 0) + t.get("bm25_ms", 0) + t.get("fuse_ms", 0)
        trace.add("search", f"Searched {r.manifest['rows']:,} passages {_BY[search.method]}", search_ms,
                  **search.describe(), k=k,
                  lists={name: [{"rank": h["rank"], "chunk_id": h["chunk_id"]} for h in hits_[:max(k, 10)]]
                         for name, hits_ in found["lists"].items()},
                  capped=sum(1 for h in hits if h.get("capped")),
                  timings={key: round(v, 1) for key, v in t.items()})
        yield "stage", trace.steps[-1].as_dict()

        stage = "guard"
        t0 = time.perf_counter()
        sources = [_source(h, n) for n, h in enumerate(hits, start=1)]
        asked = guard.instruction_flags(question)
        flagged = [s["n"] for s in sources if s["flags"]]
        trace.add("guard", _guard_label(asked, flagged), (time.perf_counter() - t0) * 1000,
                  question_flags=asked, flagged_passages=flagged)
        yield "stage", trace.steps[-1].as_dict()

        stage = "prompt"
        t0 = time.perf_counter()
        messages = build_messages(question, sources, prompt_version)
        budget = {"estimated_tokens": estimate_tokens(messages), "chars": prompt_chars(messages),
                  "num_ctx": gen.num_ctx, "passages": len(sources)}
        trace.add("prompt", f"Built the prompt from {len(sources)} passages",
                  (time.perf_counter() - t0) * 1000, version=prompt_version, **budget)
        yield "stage", trace.steps[-1].as_dict()

        info = {
            "question": question,
            "sources": sources,
            "prompt": {"version": prompt_version, "sha256": prompt_sha256(prompt_version),
                       "messages": messages, **budget},
            "model": {"name": gen.model, "backend": "local", "host": OLLAMA_URL, **asdict(gen)},
            "index": r.index_info(),
            "search": search.describe(),
        }
        yield "ready", info

        stage = "generate"
        t0 = time.perf_counter()
        first_token_ms: float | None = None
        parts: list[str] = []
        final: dict[str, Any] = {}
        for chunk in client.chat_stream(gen.model, messages, options=gen.options(), think=gen.think):
            piece = (chunk.get("message") or {}).get("content") or ""
            if piece:
                if first_token_ms is None:
                    first_token_ms = (time.perf_counter() - t0) * 1000
                parts.append(piece)
                yield "delta", {"text": piece}
            if chunk.get("done"):
                final = chunk
        text = "".join(parts)
        gen_ms = (time.perf_counter() - t0) * 1000
        prompt_tokens = final.get("prompt_eval_count")
        answer_tokens = final.get("eval_count")
        usage = {
            "prompt_tokens": prompt_tokens,
            "answer_tokens": answer_tokens,
            "first_token_ms": round(first_token_ms, 1) if first_token_ms is not None else None,
            "load_ms": _ns_ms(final.get("load_duration")),
            "prompt_ms": _ns_ms(final.get("prompt_eval_duration")),
            "answer_ms": _ns_ms(final.get("eval_duration")),
            "done_reason": final.get("done_reason"),
            "truncated": looks_truncated(messages, prompt_tokens),
        }
        trace.add("generate", f"{gen.model} wrote the answer", gen_ms, **usage)
        yield "stage", trace.steps[-1].as_dict()

        cites = citations(text, sources)
        yield "done", {
            **info,
            "answer": text,
            "refused": is_refusal(text),
            "citations": cites,
            "usage": usage,
            "trace": trace.as_list(),
            "total_ms": round(trace.elapsed_ms, 1),
        }
    except OllamaUnavailable as e:
        yield "error", {"stage": stage, "error": f"AI unavailable. {e.reason}",
                        "trace": trace.as_list()}
    except OllamaError as e:
        yield "error", {"stage": stage, "error": f"The model returned an error: {e.message}",
                        "trace": trace.as_list()}
    except Exception as e:  # noqa: BLE001 - the stream must end with a reason, not a dropped socket
        yield "error", {"stage": stage, "error": str(e) or type(e).__name__,
                        "trace": trace.as_list()}


def _guard_label(asked: list[dict[str, Any]], flagged: list[int]) -> str:
    """What the guard found, in plain words (docs/product-language.md)."""
    if asked and flagged:
        return f"Found instructions in the question and in {len(flagged)} passage(s)"
    if asked:
        return "Found an instruction in the question; the rules still apply"
    if flagged:
        return f"Found instruction-like text in {len(flagged)} passage(s)"
    return "Checked the question and passages for instructions"


def answer(question: str, **kwargs: Any) -> dict[str, Any]:
    """`ask()` run to completion: the `done` payload, or the `error` one with
    `error` set."""
    result: dict[str, Any] = {}
    for event, data in ask(question, **kwargs):
        if event in ("done", "error"):
            result = data
    return result


def _ns_ms(ns: int | None) -> float | None:
    return round(ns / 1e6, 1) if ns else None
