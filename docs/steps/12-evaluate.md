# 12. Evaluate

**What it does:** asks every test question with one setup and scores what
comes back, so each upgrade has to earn its place.

## Test questions (`data/golden/`, committed)

| Set | Questions | Scored | What it tests |
|---|---|---|---|
| A | your 9 + traps T1, T2 | yes | the main scorecard (spec §7) |
| B | 6 | retrieval only | single facts stated on several pages |
| D | 11 + traps T3-T5 | yes | edge cases: near-miss numbers, arithmetic, table cells, a flattened table, a paraphrase, typos, Spanish, two documents at once, a false premise, an instruction override, a wrong product, a wrong year |

Set C (figures) arrives in M7, after its answer is confirmed from the chart's
drawing commands. Every Set D answer was found in the extracted text, and every
page that states it is listed (`scripts/find_pages.py`, e.g.
`.venv/bin/python scripts/find_pages.py gpo 'net income' 100`). D3's key was widened on 2026-10-15: page 44
calls the *adjusted* figure plain "net income", so a $17.9M decrease is as
right as $6.3M.

## The scorecard (`rag_studio/eval.py`)

- **Pass:** every expected fact is present, or a trap was refused.
- **Refused:** an answerable question got the exact refusal. A safe failure.
- **Wrong:** a fact is missing or wrong, or a trap was answered. The dangerous one.

Facts are case-insensitive substrings (Set A, unchanged from the baseline),
lists of alternatives ("20.4" or "20,384"), or patterns with number boundaries
(so "45%" does not count as "5%"). Retrieval is scored beside it: **Hit@N**, or
**Miss** with the rank where the first answer page really was.

Keyword scoring checks that facts are present, not that every claim is
supported: A1 passes while also claiming "166 reflects the 3-year total".
The faithfulness judge (DeepEval, M6) is what catches that.

## Runs

A run is test questions × one configuration. It records the index build,
embedding digest, answer model and digest, options, prompt version and hash,
and every question's passages, answer and timings. It is saved after every
question (`data/runs/`), one run at a time across the CLI and the browser (a
file lock). `./rag eval --set a b --ladder 1` or the Evaluation page.

The results are in [`docs/ladder.md`](../ladder.md).

`--profile` chooses the index (`baseline`, `pymupdf-clean`, `docling-clean`)
and `--prompt` the prompt version; both are recorded.

## The ladder has variants (M2)

A ladder row is one change from the row above, but a row can hold more than one
way of making that change: row 2 has **PyMuPDF + cleaning** and **Docling +
cleaning**, each one change from the baseline. Runs tagged with a row are
grouped by setup (index profile, embedding, k, answer model, prompt), so the
Set A, B and D runs of one setup form one variant, whatever their labels.

## Comparing two runs (M2)

`GET /api/eval/compare?a=&b=` and the Evaluation page's comparison: run A is
"before", run B "after". It lists the setup keys that differ (the parser, the
prompt version, the index), the recall and Pass · Refused · Wrong of each, and
every question both asked, marking the ones whose outcome or Hit/Miss changed.
"Changed only" hides the rest. That is how an upgrade's effect is read: not as
a total, but question by question.
