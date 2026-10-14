"""Prompt: number the passages and put them in front of the model, with strict rules.

Every system prompt is versioned and its sha256 recorded with each answer and
each evaluation run. Whitespace is part of the version: on qwen3:1.7b at
temperature 0, one extra blank line flipped the executive-summary answer
between a refusal and a false attribution (spec §5.6). Two runs are comparable
only if their prompt hashes match.

Passages are numbered `[1]..[k]` so the model can point at one with a single
token. The model never writes a page number; code maps `[n]` back to the
document and page (rule 5).
"""
from __future__ import annotations

import hashlib
from typing import Any

REFUSAL = "I could not find this in the documents."

# System prompt v2 exactly as tested (spec §6): one blank line after the first
# line, no trailing newline. Rule 4 is relaxed in a later version, once
# passages carry section labels the model can actually see.
SYSTEM_PROMPTS: dict[str, str] = {
    "v2": (
        "You answer questions about a set of PDF reports, using only the numbered context passages you are given.\n"
        "\n"
        "Rules:\n"
        "1. Use ONLY facts written in the passages. Never use your own knowledge, even if you know the answer.\n"
        "2. First check whether a passage directly states the answer. If none does, reply with exactly this sentence and nothing else:\n"
        f"{REFUSAL}\n"
        "3. After each fact, put the number of the passage that states it in square brackets, e.g. [2]. Only cite a passage that really contains that fact.\n"
        "4. Never say which section, chapter, or summary a fact comes from. You cannot see that; the citation is enough.\n"
        "5. Answer in at most 3 short sentences. Do not repeat yourself."
    ),
}
# The prompt the hand-built `answer.py` actually sent when the reference
# scorecard was measured (PASS 9 | REFUSED 0 | WRONG 2): v2 plus one blank line
# after the first line and a trailing newline. Kept only to prove parity with
# that run; v2 is the tested text and the default.
SYSTEM_PROMPTS["v2-handbuilt"] = SYSTEM_PROMPTS["v2"].replace(
    "given.\n\nRules:", "given.\n\n\nRules:") + "\n"
DEFAULT_VERSION = "v2"

# For the context-budget bar before the model has counted. English prose in
# these reports runs about 4 characters per token; the real count comes back
# from Ollama as `prompt_eval_count` and replaces this estimate.
CHARS_PER_TOKEN = 4.0


def system_prompt(version: str = DEFAULT_VERSION) -> str:
    try:
        return SYSTEM_PROMPTS[version]
    except KeyError:
        raise KeyError(f"No system prompt version {version!r}. Known: {', '.join(SYSTEM_PROMPTS)}.") from None


def prompt_sha256(version: str = DEFAULT_VERSION) -> str:
    return hashlib.sha256(system_prompt(version).encode("utf-8")).hexdigest()


def context_block(passages: list[dict[str, Any]]) -> str:
    """`[n] chunk_id` then the passage text, one blank line between passages.
    The baseline format, unchanged."""
    return "\n\n".join(f"[{i}] {p['chunk_id']}\n{p['text']}" for i, p in enumerate(passages, start=1))


def build_messages(question: str, passages: list[dict[str, Any]],
                   version: str = DEFAULT_VERSION) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": system_prompt(version)},
        {"role": "user", "content": f"Context:\n{context_block(passages)}\n\nQuestion: {question}"},
    ]


def prompt_chars(messages: list[dict[str, str]]) -> int:
    return sum(len(m["content"]) for m in messages)


def estimate_tokens(messages: list[dict[str, str]]) -> int:
    return round(prompt_chars(messages) / CHARS_PER_TOKEN)


# Ollama does not fail a prompt longer than `num_ctx`: it answers HTTP 200
# after silently dropping the start (measured, docs/research.md §2). The only
# trace is `prompt_eval_count`, which then counts far fewer tokens than the
# text holds. Normal prose here is 3-5 characters per token; above this, most
# of the prompt was not read.
TRUNCATION_CHARS_PER_TOKEN = 8.0


def looks_truncated(messages: list[dict[str, str]], prompt_tokens: int | None) -> bool | None:
    """True if the model read far fewer tokens than the prompt holds. None when
    Ollama did not report a count."""
    if not prompt_tokens:
        return None
    return prompt_chars(messages) / prompt_tokens > TRUNCATION_CHARS_PER_TOKEN
