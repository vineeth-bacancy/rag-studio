# QA: M8 full matrix (2026-10-23)

**Scope** (`docs/plans/m6-m8.md`, M8): every page at 375 / 768 / 1024 / 1280 /
1440 px, light and dark, keyboard, a clean console. 20 pages: Ask, Documents,
Indexes, How it works, Evaluation, Settings, System Health and the 13 How it
works steps. 200 renders per pass.

**Method:** Chrome DevTools MCP (chrome-devtools-mcp 1.11.0, Chrome 155) on
`./rag web` (dev mode), watched live by the user. A harness
(`scripts/qa/harness-src.js`, built with `scripts/qa/build.cjs`) is injected
into an open app page and loads each page in an iframe sized to the width under
test. An iframe has its own viewport, so media queries, the phone sidebar and
the `@container` queries respond as they would in a window of that width; a
CSS scale only shrinks what is shown. Same origin, so the harness measures
inside the frame:

- page overflow (`scrollWidth` vs the viewport), boxes that scroll sideways,
  elements past the edge outside any clipping box;
- exactly one `<h1>`;
- axe-core 4.14.0, WCAG 2.1 A/AA + best practice (contrast, names, ARIA,
  landmarks);
- console errors and warnings, uncaught errors, failed requests (4xx/5xx);
- the theme actually applied (`localStorage.theme`, as the app's toggle sets it).

Keyboard: a focus audit on every page (each tabbable element focused in turn,
its look compared focused vs blurred; script focus in a frame no pointer has
touched matches `:focus-visible`, checked on every element), then real key
presses through DevTools for the skip link, Ctrl+K, a Base UI select, the phone
sidebar, and the scroll boxes axe flags. Contrast of the colour chips was also
computed from the browser's own computed colours (canvas), because axe cannot
always score text on a translucent tint.

## Result (after the fixes below)

| Check | Result |
|---|---|
| Page overflow, 20 pages × 5 widths × 2 themes | **0 / 200** |
| Exactly one `<h1>` | 200 / 200 |
| Console errors or warnings, uncaught errors | **0** |
| Failed requests | **0** |
| axe violations | **0** except `scrollable-region-focusable` on 8 pages (M8-11, Safari only) |
| Focus audit, 20 pages | every focusable control (16 to 377 per page) shows a visible focus and has a name. Not focusable, correctly: disabled buttons (Previous page on page 1, Rebuild on Colab without a runtime, Restore with nothing changed) and controls inside a closed disclosure |
| Skip link | first stop in the document, visible when focused, Enter moves focus to `<main id="main">` |
| Ctrl+K | opens with focus in "Go to a page..."; 7 pages + the theme; "eval" + Enter opens Evaluation; Escape closes and returns focus |
| Base UI select (step 11's figure picker) | ArrowDown opens and moves, Enter picks (the figure switched to GEP p97), focus returns to the trigger |
| Phone sidebar (375) | Enter on the toggle opens a dialog named "Sidebar" with focus inside; Escape closes it and returns focus to the toggle |
| Scroll boxes | Tab reaches them in Chrome 130+ (Documents inspector text at 375: focused, outline shown, arrows scroll) |
| Ask with too little memory | 6.4 GB free: the composer is replaced by "Not enough free memory to answer yet" with what to do; nothing loads |
| Boxes that scroll sideways on purpose | step 12's scorecard grid (min 34rem: 576 of 337 px at 375, 458 at 768), the Evaluation comparison table at 375 (342 of 305), step 11's truth table at 375 (319 of 305) |
| Tests | `./rag test` 573 passed, 1 skipped; UI typecheck and lint clean |

## Findings

Severity: P0 broken · P1 wrong or misleading · P2 degraded · P3 polish.

| ID | Sev | Where | Finding | Status |
|---|---|---|---|---|
| M8-01 | P2 | Every page (reported by the user) | The scrollbar was 10 px wide: `scrollbar-width: thin` is Chrome's narrowest standard width on Linux | **Fixed**: Chromium and Safari get `::-webkit-scrollbar` at 6 px, a transparent track and a 4 px thumb at 20% of the text colour (40% on hover); Firefox keeps `thin`. Measured: 6 px gutter, page background behind the track |
| M8-02 | **P1** | Ask, a reopened conversation | Every citation of a saved answer was drawn as invented (red "1?", "The model cited [1], but it was only given 0 passages."), although the answer's citations were valid: `openChat` restored the citations but not the passages | **Fixed**: a saved answer's citations are neutral marks naming document and page (the passage text is not kept, so they open nothing); red only for a number outside the saved passages, counted from them |
| M8-03 | P2 | Light theme: every page ≥ 768 px (the top bar's ⌘K key), inline commands on Indexes, Settings and step 5, counts on steps 2, 12 and 13 | Muted text on `bg-muted` 4.34:1, and 4.4:1 on the red tint, below 4.5 | **Fixed**: light `--muted-foreground` oklch 0.556 → 0.54: 4.62:1 on muted, 5.04:1 on white |
| M8-04 | P2 | Steps 3, 4, 7, 8 | White on sky-600 badges 4.02:1; the chunk playground's passage chips: sky-500/white 2.71 (dark), amber-600/white 3.2, violet-500/white 4.4 (dark), fuchsia-500/white 3.54 (dark) | **Fixed**: sky-700 badges (5.86); chips sky-700 / violet-600 / amber-700 / teal-700 / fuchsia-600 with white, and the 400-500 steps with black in dark: every pair ≥ 4.66:1 |
| M8-05 | P2 | Step 5; Ask citations | Chart-blue text on a 15% chart-blue tint: step 5's prompt-prefix mark 3.68:1 (dark), 4.07 (light); Ask's citation buttons, source numbers and preview number 4.43:1 (light) | **Fixed**: the mark is foreground text on a 20% tint with a ring (14.7:1); the citation chips use a 10% tint (5.5 light, 5.9 dark) and a ring on hover |
| M8-06 | P2 | Every select, seen on Evaluation's index picker | The list was exactly as wide as its trigger: long index names were cut mid-word under the check mark, hiding why an index is greyed out ("· can't be asked now", "· not built yet") | **Fixed** in `ui/select.tsx`: the list grows to its longest item, never narrower than the trigger nor wider than the screen, and wraps only when capped (checked at 1471 and 375 px) |
| M8-07 | P3 | Step 5 | The "5 of 5" dots had `aria-label` on a span with no role (axe `aria-prohibited-attr`) | **Fixed**: `role="img"` |
| M8-08 | P3 | Ask, low memory | "loads gemma4:e4b (about 8 GB): about 8 GB in all" | **Fixed**: the total only when two parts load |
| M8-09 | P3 | Step 12 | The scorecard grid's corner header was empty (axe `empty-table-header`) | **Fixed**: "Retrieval" for screen readers |
| M8-10 | P3 | Every page ≥ 768 px | The sidebar's brand and status rows were outside any landmark (axe `region`) | **Fixed**: on a desktop the sidebar is a complementary landmark named "Sidebar" (on a phone it is already a dialog) |
| M8-11 | P3 | Documents, Evaluation at 375, steps 2, 3, 4, 6, 9, 11 | axe `scrollable-region-focusable` ("serious"): text boxes that scroll but hold nothing focusable | Open. Chrome 130+ and Firefox make such boxes keyboard-focusable themselves (verified above); Safari does not. `tabIndex={0}` plus a label on each box would cover Safari |
| M8-12 | P3 | Ctrl+K | After choosing a page, focus lands on the top bar's Toggle Sidebar rather than the new page | Open. Next's route announcer reads the new page's title |
| M8-13 | info | QA harness | Twice a DevTools call over its 180 s limit left the tab unresponsive (even screenshots failed). A fresh tab rendered every page; the same pages passed in shorter calls | Not an app defect: calls now cover two or three pages |

**Not covered:** a live answer in Ask (the M7 carry-over "Ask with a figure
source"): gemma4:e4b needs 8 GB free and 6.4 GB was free with VS Code open.
Ask's refusal was checked instead, and a saved conversation's answers and
citations were read (M8-02). A figure source's badge and page box are covered by
`tests/test_figures_api.py`. Safari and Firefox were not run.

**Open from earlier milestones (no P0-P1):** M5-06 (P3), M5-07 (P2, kept on
purpose), M6-03 (P3), M6-04 (P3), M7-04 (P3).
