# QA: M4 Remote AI, Indexes, How it works step 5 (2026-10-15)

**Method** (as in M1-M3): Chrome DevTools MCP on `./rag web` (dev mode), the
real indexes, a real Colab T4 runtime connected by the user (with the CLI:
see M4-01), and the real models. Run while Colab builds held the runtime,
never beside a scorecard run on this computer.

- Indexes, Settings, How it works › Embed and System Health × 375 / 768 /
  1024 / 1280 / 1440 px, in dark and light, measured in a sized iframe: page
  overflow and elements wider than the viewport.
- Screenshots in dark and light, desktop (1280) and phone (375 × 812 @2x).
- The connect dialog with a real-length connect code (373 characters) and the
  longest the API accepts (4,000), at 1280 and 375.
- The embedding lab on its teaching cases: "What are the main risks to the
  global outlook?" on Docling + EmbeddingGemma 2, Matryoshka, and the mixed
  model (a nomic-embed-text question in the EmbeddingGemma 2 index).
- A build's live progress (EmbeddingGemma 2 here, Qwen3-Embedding 8B on
  Colab), the console.

## Result

| Check | Result |
|---|---|
| Horizontal overflow, 4 views × 5 widths × 2 themes | 1 of 40 (M4-03), fixed; 40/40 after |
| Console errors / warnings | 0 / 0 (over the last three navigations) |
| Settings, not connected | Local AI "In use", Remote AI "Not connected"; "Where each job runs" (Embedding per index, Answer local until M6, Fallback strict); the setup cell 22.6 KB, **no token in it** (checked: no `RAG_STUDIO_TOKEN` assignment), Ollama pinned to 0.32.6 |
| Settings, connected (CLI connect, then the page) | "Remote AI connected", Healthy; Tesla T4, 8.8 / 15.4 GB; qwen3-embedding:0.6b and :8b "loaded on the GPU"; Ollama 0.32.6, 209 ms away; last test passed |
| **Connect from the browser, real runtime** | Settings › Reconnect, the bare tunnel URL (the token remembered from the CLI connect, so no token field and no secret in the browser): "Remote AI is connected · qwen3-embedding:0.6b embedded a test sentence in 7.2 s"; the test card "Embedded a test sentence with each model in 28 s" (0.6B 7.2 s, 8B 17 s: queued behind a running build's batches) |
| System Health | Remote AI section: Healthy, "Connected to a Tesla T4 with 2 models." |
| Indexes | Row 5 as model × parser; reasons in words on every disabled action ("Built on the Colab GPU: Qwen3-Embedding 8B is too large…", "Remote AI is not connected. ./rag llm connect"); the crashed build reads "The last build stopped… it resumes"; a running build shows its stage and "1,272 of 1,428 passages (45 already embedded earlier)" with Cancel |
| Embedding lab | 1,473 points, the three documents in separate clouds (validated blue / amber / magenta, all pairs, both themes); the question lands in the GEP cloud, dashed lines to its five nearest (GEP p54, 50, 80, 36, 54); the prefix highlighted in "What the model read"; Matryoshka 768 / 512 / 256 / 128 → 5 / 5 / 4 / 3 of the top 5; the mixed model: "0 of 5 … every cosine is near 0.05" |

## Findings

Severity: P0 broken · P1 wrong or misleading · P2 confusing or ugly · P3 polish.

| ID | Sev | Where | Finding | Fix |
|---|---|---|---|---|
| M4-01 | P0 | Settings › Connect a Colab GPU | **Reported by the user**: pasting the connect code pushed "Connect and test" off screen, so connecting from the browser was impossible (the user connected with `./rag llm connect`). Measured: the 373-character code made the textarea 3,155 px wide in a 512 px dialog, the button at x = 3,393. A connect code is one unbroken base64 string; the textarea sizes to its content (`field-sizing-content`) and grid items default to `min-width: auto` | The textarea keeps a fixed size, breaks anywhere and scrolls (`field-sizing-fixed`, `break-all`, `max-h-40`); its wrapper is `min-w-0`; the dialog scrolls if taller than the window. Measured after: dialog 512 px, textarea 480 × 78, the button visible at 1280 and 375, for 373 and 4,000 characters |
| M4-02 | P3 | System Health | The section and its check were both titled "Remote AI" | The check is "Colab runtime" (the technical term; docs/product-language.md) |
| M4-03 | P2 | Indexes, 375 px | "Remote AI connected (rates-normally-response-member.trycloudflare.com)": a badge never wraps, 87 px of page overflow | The badge says "Remote AI connected"; the host is its tooltip (and on Settings) |
| M4-04 | P2 | Embedding lab | Rank labels 1 and 2 were hidden: each nearest dot was drawn with its own label, so later dots covered earlier labels | All five dots first, then all five labels |
| M4-05 | P3 | Indexes, 375 px | The build banner's spinner centred on three wrapped lines, looking stray | Aligned to the first line |
| M4-06 | P2 | Indexes | "Another index is being built. One build runs at a time." under every one of six cards | Said once, above the cards, naming the build |
| M4-07 | P1 | Indexes | "Rebuild here · about 15 min": a rebuild reuses every unchanged vector and takes seconds | Time estimates only for a first build |
| M4-09 | P3 | Connect dialog | With a bare URL: "remembers the token its last setup cell carried", but the remembered token can come from the last connection (here a CLI connect) | "will use the token it remembers, from the last connection or the last ./rag llm colab cell" |
| M4-10 | P3 | Settings, connected | "Ollama 0.32.6 · 3480 ms away": that is how long `/api/version` took while the GPU was busy with a build, not a distance | "answered in 3,480 ms" |
| M4-11 | P1 | Ask and Evaluation index pickers | A built Colab index (8B) could be chosen with no runtime connected, and the question then failed; Evaluation listed `./rag ingest --profile …` for every unbuilt index, wrong for the 8B (Colab only) and seven lines long | Both pickers disable an index the engine says cannot be asked now, with its reason ("Remote AI is not connected."); one line sends unbuilt indexes to the Indexes page |
| M4-08 | P1 | CLI and the build card | "45 from an earlier attempt": those vectors came from the other parser's build (passages with the same text), not from an attempt | "already embedded earlier" (CLI: "by this or an interrupted build") |

## Not fixed, on purpose

- **A connect code was not pasted into the browser on the real runtime**: the
  user's first connect was the CLI (M4-01), and the code is not kept anywhere
  to paste again. The browser connect was run with the URL and the remembered
  token instead (above); the dialog with real-length codes is verified by
  measurement, and `tests/test_remote.py` drives the code path through the
  real gateway.
- **Rank labels of two nearest passages can still touch** when their dots sit
  within a few pixels; the list beside the map names every rank.
- **Next's dev-mode badge** sits over content at 375 px; it does not exist in a
  production build.
