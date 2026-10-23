# 12. Evaluate

**What it does:** asks every test question with one setup and scores what
comes back, so each upgrade has to earn its place.

## Test questions (`data/golden/`, committed)

| Set | Questions | Scored | What it tests |
|---|---|---|---|
| A | your 9 + traps T1, T2 | yes | the main scorecard (spec §7) |
| B | 6 | retrieval only | single facts stated on several pages |
| C | 4 (M7) | yes | values that only a chart shows: iMac p2's footprint bar, three GEP chart panels; every key measured from the PDF's drawing (`truth` block per question) |
| D | 11 + traps T3-T5 | yes | edge cases: near-miss numbers, arithmetic, table cells, a flattened table, a paraphrase, typos, Spanish, two documents at once, a false premise, an instruction override, a wrong product, a wrong year |
| E | 8 follow-ups (M5) | yes | conversation memory: each question is asked after a recorded first turn and only makes sense with it ("And for 2027?"), including a refusal and a change of subject |

Set C's keys were measured from the chart's drawing commands (`get_drawings()`
heights calibrated by the axis ticks, then a 300 dpi crop), never read by eye
(`docs/steps/11-figures.md`). Every Set D answer was found in the extracted text, and every
page that states it is listed (`scripts/find_pages.py`, e.g.
`.venv/bin/python scripts/find_pages.py gpo 'net income' 100`). D3's key was widened on 2026-10-15: page 44
calls the *adjusted* figure plain "net income", so a $17.9M decrease is as
right as $6.3M.

Set E is **remote-only** (`"remote_only": true` in `set_e.json`): Remote AI
(qwen3:8b on Colab) rewrites each follow-up into a question that stands alone
before it is searched. `./rag eval` with no `--set` runs A, B and D only; name
Set E on purpose (`--set e`). `--as-typed` searches each follow-up exactly as
written instead, with no Remote AI at all: the run that shows what memory is
worth. The Evaluation page says "needs Remote AI" beside Set E and will not
start it without a connection unless "Ask follow-ups as typed" is on.

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
A faithfulness judge (DeepEval) would catch that; it was **cut** from this
version (2026-10-22, `docs/plans/m6-m8.md`), with RAGAS and tracing, so this
deterministic scorecard is the only judge. The Judge role reads "Arrives in a
later version".

**Set E is judged in order: the rewrite, then retrieval, then the answer.**
Each follow-up lists words its rewritten question must contain
(`standalone_expect`, e.g. "South Asia" and "2027") and words it must not
(`standalone_forbid`: the topic switch from GPO to the iMac stand must not
carry GPO over). The run's summary keeps the chain
(`summary.followups`: rewritten right, then found, then answered right), and
the Evaluation page shows it as its own table, because it measures the
conversation's memory, not the search.

## Ranking beyond the top 5 (M5)

Every run searches 50 deep, so beside recall@5 it reports, over the questions
that have answer pages:

- **MRR** (mean reciprocal rank): the average of 1 / rank of the first answer
  page, 0 for a question whose page is not in the top 50. 1.0 means every
  answer page was at rank 1.
- **recall@20**: how many questions have an answer page in the top 20, the
  pool a reranker reads.
- both again for questions about **one document** and questions that need
  **two** (D's cross-document question), because a two-document question only
  counts once both pages are in.

They come from the per-question `best_rank` every run has recorded since M1, so
runs made before M5 show them too (computed when the run is read; the file is
not rewritten). A question with no answer page does not count; with none
counted, the value is "-", never 0.

## Runs

A run is test questions × one configuration. It records the index build,
embedding digest, answer model and digest, options, prompt version and hash,
and every question's passages, answer and timings. It is saved after every
question (`data/runs/`), one run at a time across the CLI and the browser (a
file lock). `./rag eval --set a b --profile baseline --prompt v2 --ladder 1` or the Evaluation page.

The results are in [`docs/ladder.md`](../ladder.md).

`--profile` chooses the index (`baseline`, `pymupdf-clean`, `docling-clean`)
and `--prompt` the prompt version; both are recorded.

### Runs with M5's switches

`--rerank ID`, `--rewrite decompose|multi|hyde` and `--agent ROUNDS` (the
Evaluation page's Reranker, Query rewriting and Multi-step search) are each
recorded under `config.retrieval` as the component's full, stable description:
model, Hugging Face revision or Ollama digest, dtype, where it runs, prompt or
instruction version and sha256, depth, `num_ctx`. Facts that change between
runs of one setup (whether Remote AI was connected, its GPU) go under
`config.runtime`, never `retrieval`, so they never split a ladder variant.

**A run with any of them searches in two passes.** Pass 1 searches every
question (rewrite, each search, fusion, the reranker); then the reranker and
EmbeddingGemma 2 are released; pass 2 answers every question from the
passages pass 1 found. The answer is always written from the same search that
was scored, and the answer model never shares memory with the search models.

**Remote outputs are remembered.** Reranker scores (`data/cache/rerank/`), and
rewrites, follow-up rewrites and multi-step decisions (the remote output cache)
are kept by exactly what was asked. A retrieval-only run on Colab fills them,
and the scorecard that follows replays the same values with no Colab at all;
each question records `cache: hit | miss` per step. `--fresh-rerank` scores
again for a timing. If Remote AI goes away mid-run and a value is not
remembered, the run **fails** with the questions it finished, rather than
finishing as a run full of errors.

**What a run shows per question** (open a question on the Evaluation page,
"How it was searched"): for a follow-up, the earlier turn and "Searched for: …"
with whether the rewrite was right; for rewriting, the phrasings, parts or
drafted passage; for multi-step search, each round (the normal search, then
each new query, the document it was limited to, how many new passages it
added, the controller's one-sentence reason) and why it stopped (enough
evidence, the round limit, the time or token budget, a repeated query, or bad
output); for the reranker, where the first answer page was before and after
reranking and whether the scores were remembered. A Remote AI call shows its
network time against the runtime's time.

## The ladder has variants (M2)

Each variant's line names its setup (index, search, rewriting, multi-step
search, reranker, answer model, prompt) and, since M5, its MRR and recall@20.

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

## How it works 12 (M6)

`learn.evaluate_lab(run, against)`, `/learn/evaluate`: one recorded run's
scorecard taken apart, read from the saved run (no model call).

- **The grid:** Hit / Miss / Trap (no answer page) × Pass / Refused / Wrong,
  with the question ids in each cell and what each corner means (a Miss that
  is Wrong is the dangerous one: nothing was found and it answered anyway).
- **One question, scored:** the expected facts, where each matched in the
  answer (`eval.fact_spans`, the same rule as `fact_found`), and an editable
  copy scored again by `score_answer` (`/learn/answer-check`).
- **MRR and recall@20:** one bar per question, height 1 / the deep rank, so
  MRR is visibly their average.
- **Set E's chain** when the run has follow-ups.
- **A hint, not proof:** `eval.pass_interval` (Wilson 95%) for the pass
  rate, and against another run `eval.sign_test` (exact two-sided McNemar) on
  the questions that changed between Pass and not. Two baseline Set D runs on
  2026-10-14 wrote D3 word for word the same (only the key changed between
  them), so at temperature 0 the uncertainty is in which 25 questions were
  asked, not in sampling. Example: 6d (22 · 1 · 2) against 5d (20 · 3 · 2): D7
  and D10 became Passes, none got worse, p = 0.5; 22 of 25 is consistent with a
  true pass rate of 70-96%.
