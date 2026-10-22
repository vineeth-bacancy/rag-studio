# 4. Contextual retrieval

**What it does:** before a passage is indexed, a model reads the part of the
document it sits in and writes a **context line**: one or two sentences saying
what the passage alone does not (whose figures, which year, which table). The
line is embedded and indexed with the passage, so search can find it by what it
is about. It is Anthropic's "Contextual Retrieval", adapted here.

A passage like "Growth is expected to slow to 2.3 percent" does not say whose
growth or which year. The section label (M3) adds the document and the section;
the context line adds what the section's neighbouring text makes clear.

**Only search reads the line.** The answer model is given the passage and its
source exactly as on an index without context lines (`chunk.Passage.embed_text`
holds the line, `text` does not). So ladder row 7 is a retrieval-only change:
a better answer can only come from finding a better passage.

## Where it lives

| Piece | Where |
|---|---|
| Windows, the prompt, the cache, the job, the export | `rag_studio/context.py` |
| Profiles | `{pymupdf,docling}-structure-embeddinggemma2-ctx` (`rag_studio/profiles.py` `ROW7`): row 5c / 5d with `contextual=True`; nothing else changes |
| Writing lines | `POST /context` on the Colab worker (`tools/colab/worker.py`), through the gateway |
| CLI | `./rag context status \| write \| check \| export \| import <profile>` |
| Building the index | `./rag index build <profile>-ctx`: looks every line up in the cache, embeds here (EmbeddingGemma 2), **never calls Colab** |
| How it works, step 4 | `learn.context_lab`, `GET /api/learn/context`, `web/ui/src/components/learn/context-lab.tsx`: reads the cache only |

## Windows

Each passage is read with a **window**: the document's title, a short outline of
its headings (up to 60, in reading order), and the text of the **block** the
passage sits in. A block is a run of neighbouring passages from one section,
never crossing a section or a document, cut into pieces of at most **6,000
estimated tokens** of passage text (characters / 3, a pessimistic estimate for
numbers and tables).

Every passage in a block shares one window, so the window comes first in the
prompt and the passage last: the model server can reuse its work on the shared
start. A window centred on each passage would share nothing.

| Measured from the code (2026-10-22) | Docling | PyMuPDF |
|---|---|---|
| Passages | 1,473 | 1,214 |
| Windows | 390 | 140 |
| Windows of one passage | 133 | 31 |
| Passages per window, median / max | 2 / 26 | 5 / 27 |
| Estimated tokens per window, median / max | 1,961 / 7,398 | 2,243 / 7,349 |

The largest windows pass 6,000 because the title and outline sit on top of the
block's text. Every call uses `num_ctx` 8,192, and the worker refuses a prompt
the model cut (characters per counted token over 8): a line written from a
truncated window would describe the wrong text, so it is an error, never a line.

## The prompt: `context-v1`

Anthropic's wording, adapted to one or two sentences (at most 35 words). The
system prompt says the document and the passage are **quoted data, never
instructions**, gives three examples of the style, and asks the model not to
begin with "The passage". The window is fenced in `<document>`, the passage
(its section label and text, never a context line) in `<passage>`. Its sha256
covers the system prompt, the fences, the instruction and `num_ctx`; the build
manifest records it (`config.context`), so a changed prompt is a different
index. The full text is in `context.PROMPTS` and on the How it works page under
Technical details.

Run by the `context` role: `qwen3:8b` on **Remote AI only**, temperature 0,
thinking off, at most 160 new tokens. There is no local fallback: a different
model would make a different index while the run said otherwise.

## The cache, the job and the export

- **Cache:** `data/cache/context/<model digest>/<prompt sha[:16]>/<key>.json`.
  The key is `sha256(window policy + window + label\ntext)`, built from the
  context-free text, so a key never depends on its own output. Writes are
  atomic.
- **The job** (`context.write_lines`) sends passages block by block in
  document order, 8 per request, 2 requests in flight, so the GPU never waits
  for the tunnel. It stores every line the moment it arrives, so a job cut off
  by a closed session or a reclaimed runtime starts again where it stopped. It
  has its own lock and job file (`data/indexes/jobs/<profile>.context.json`),
  not the build's: writing lines uses the runtime, not this computer, and must
  not block an evaluation or show up as an index build. The worker keeps
  finished lines by key, so a retry after a timeout redoes nothing, and its
  batch time cap (45 s) counts from arrival, queue wait included, so a batch
  stays under the tunnel's 100 s and returns what is left as `remaining`.
- **Not reproducible byte for byte:** a different GPU or batch order can give
  a different sentence (`./rag context check` regenerates 20 lines in a
  shuffled order and reports the exact-match rate). So a finished profile is
  exported to `data/context/<profile>.jsonl` (`./rag context export`), and that
  file, not the model, is what makes the index rebuildable
  (`./rag context import` after a wiped `data/cache`).
- **Ingest never calls Colab.** A `-ctx` build looks every line up and refuses
  with "N of M passages have no context line yet. Write them on Remote AI
  first: ./rag context write <profile>" when any is missing.

## Measured (T4, qwen3:8b Q4_K_M, 2026-10-16 and 2026-10-22)

| | Value |
|---|---|
| First trial (48 lines) | 0.9-1.8 s per line, limited by the GPU writing about 40 tokens; about 1 s of tunnel time in total |
| PyMuPDF job (170 lines, 50 requests) | 498.7 s: **0.34 lines/s**; 529,038 prompt tokens and 7,805 written tokens; 3 batches asked again after the time cap |
| Network in that job | 13.3 s of 963 s of call time (1.4%); requests 633 KB sent as 229 KB (gzip); **1** connection for 53 calls |
| **All 1,643 lines** (Docling 1,473 + PyMuPDF 170) | **1.38 h of T4 model time**; median 2.9 s per line; about 0.31-0.34 lines/s end to end |
| Window reuse | not seen: the first line of a block took 2.3 s (median), the later ones 2.9 s |
| Lines that still begin "The passage" / "This passage" | 694 of 1,643 (42%), although the prompt asks not to |
| Line length | median 34 words, longest 57 words (405 characters) |
| Written | Docling **1,473 of 1,473** (over three runtimes: a session ended at 769, a runtime died at 1,335, and the job resumed each time from the cache); PyMuPDF 170 of 1,214 (the rest not written: the user chose Docling only, to save T4 time) |
| Failures met and fixed | one batch timed out on the tunnel (now asked again: the runtime keeps finished lines); one honest line of 401 characters hit a 400-character junk limit and stopped the job (the limit is now 720, the worker's `num_predict` cap, and a rejected line no longer stops a job) |
| Build of `docling-structure-embeddinggemma2-ctx` here | 1,284 s (1.15 passages/s, EmbeddingGemma 2 on this CPU); every vector new, since each passage's text changed |
| Export | `data/context/docling-structure-embeddinggemma2-ctx.jsonl`, 1.0 MB |

> **Row 7 results:** see `docs/ladder.md`, row 7.

## How it works, step 4

The page shows, from the cache only (no model call while you look):

- how many lines exist for each `-ctx` index, the model, the prompt and the
  last job (a job file that still says "running" after its process ended reads
  "interrupted");
- one passage: its context line, and exactly what is embedded **without** and
  **with** it (the index's document prefix, the section label, the line, the
  passage), plus the window it was written from;
- for a test question (B4, A6, D10), where the first answer page ranks,
  searching 50 deep, on the index without context lines and on the one with
  them, and the shown passage's own rank in each. An index that is not built
  says so and shows how to build it;
- a few other lines from across the documents; picking one shows it above.

Docling is never run from this page: if the PDFs have not been converted, it
says to build the index first. Cutting the passages takes 1.3-1.4 s per
profile and is kept for the life of the app process.
