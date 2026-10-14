# QA: M1 Ask, Evaluation, Documents (2026-10-15)

**Method** (as in M0, after the trading app's audit): Chrome DevTools MCP on
`./rag web` (dev mode), an isolated browser context, the real baseline index
and the real `qwen3:1.7b`.

- Every view × 375 / 768 / 1024 / 1280 / 1440 px, measured in a sized iframe:
  page overflow, elements wider than the viewport, `<h1>` count, buttons with no
  accessible name. Views: Ask, Evaluation, Evaluation with a run open, Documents,
  System Health.
- Screenshots in dark (default) and light, desktop (1280-1440) and phone
  (375 × 812 @2x).
- A real question end to end through the Next proxy: stream timing, the
  citation chip, the page preview, the "What happened" trace.
- Keyboard and console.

## Result

| Check | Result |
|---|---|
| Horizontal overflow, 5 views × 5 widths | none (25/25) |
| Exactly one `<h1>`, no unnamed buttons | 25/25 |
| Console errors / warnings | 0 / 0 |
| Streaming through the Next rewrite | sources and steps arrive within 0.5 s, before the model starts; the answer streams in; 25 s total for A1 |
| Citation → page preview | page 5 rendered, the passage's lines outlined exactly ("…agement (LSCM)" to "…undertook a major") |
| Escape in the preview | closes it; focus returns to the citation that opened it |
| Run mode choice | native radios: arrow keys move between them |
| Scrollbars (Chrome) | computed `scrollbar-width: thin`, thumb = foreground at 22%, track transparent, both themes |
| Scrollbars (Firefox, Safari) | **not tested here** (no Firefox or Safari on this machine). Firefox uses the same standard properties; Safari gets the `::-webkit-scrollbar` fallback |

## Findings

All were fixed in this pass. Severity: P0 broken · P1 wrong or misleading ·
P2 confusing or ugly · P3 polish.

| ID | Sev | Where | Finding | Fix |
|---|---|---|---|---|
| M1-01 | P2 | Ask, sources | While the answer was still streaming, every source said "not cited", which is not yet known | Citation state is `null` until `done`; "not cited" appears only after |
| M1-02 | P3 | Evaluation, Run evaluation | Before the model list loaded, the estimate read "No earlier run with null to estimate the time from" | "Loading the installed models..." until it arrives |
| M1-03 | P2 | Evaluation, run detail | 5 tiles on a 2-column grid left a grey empty cell (the tile row's separator colour showing through) | Two tile rows, each exactly as wide as its tile count; the time moved into the run's description line |
| M1-04 | P2 | Evaluation, an opened question (reported by the user) | The answer panel was cramped: it started under the ID column, the passage list was ragged ("rank 1 Global Economic Prospects -- June 2026 · page 98 0.774"), found/missing were a run-on line | Redesigned: the panel starts under the question; answer and fact chips (✓ found, ✗ missing) on the left, an aligned rank · document · page · score table on the right with answer pages marked, a footer for time and tokens; stacks on narrow cards |
| M1-05 | P3 | Same panel (reported by the user) | The panel touched the question row; the open row was tinted by the table's `has-aria-expanded` style but the panel row was not | One shared tinted band for the open question and its panel (same `has-aria-expanded` variant, so tailwind-merge replaces it), 8 px above and 20 px below the panel, the panel on the card surface |
| M1-06 | P3 | Everywhere (requested by the user) | Default scrollbars were wide and opaque | Central rule in `globals.css`: thin, transparent track, theme-aware thumb, standard properties with a `@supports`-gated WebKit fallback; the Base UI ScrollArea thumb uses the same colour |
