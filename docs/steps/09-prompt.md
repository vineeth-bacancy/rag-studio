# 9. Prompt

**What it does:** puts the numbered passages in front of the model, with
strict rules about using only them.

## Baseline (M1)

`rag_studio/prompt.py`. The user message is
`Context:\n[1] <chunk_id>\n<text>\n\n[2] …\n\nQuestion: <question>`, and the
system prompt is **v2**, exactly as spec §6 prints it (sha256 `d30ea3a1…`,
pinned in `tests/test_prompt.py`).

Passages are numbered so the model can point at one with a single token. The
model never writes a page number.

## Whitespace is part of the prompt (measured)

The hand-built `answer.py` sent v2 with **one extra blank line** and a trailing
newline. That variant is kept as `v2-handbuilt`, byte-identical to
`answer.py` (tested), only to prove parity:

| Prompt | Set A on qwen3:1.7b | A6 (executive summary) | T2 (FY 2019 trap) |
|---|---|---|---|
| `v2-handbuilt` | PASS 9 · REFUSED 0 · WRONG 2 | claims "according to the executive summary" | "$20.4 million… I could not find this" |
| `v2` (the default) | **PASS 10 · REFUSED 1 · WRONG 0** | refuses | refuses |

Same model, same passages, temperature 0: one blank line moves two rows. That
is why every run records the prompt version and its hash, and two runs are
compared only when the hashes match.

## The context budget

A Set A prompt is 1,258-1,815 tokens (Ollama's own count) of an 8,192-token
window. Ollama does **not** fail a prompt that overflows `num_ctx`: it drops
the start and answers anyway. Its `prompt_eval_count` still counts the whole
prompt when a prefix is cached (measured: 274 tokens both times), so the app
compares characters to tokens. Prose here runs about 4 characters per token,
and above 8 the trace says "the start of the prompt was likely dropped".

## Next (M6)

Relax rule 4 once passages carry section labels the model can see; How it
works step 9 shows the exact messages and the budget bar.
