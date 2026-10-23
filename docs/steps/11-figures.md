# 11. Figures

**What it does:** turns each chart and diagram into text a search can find,
so a question whose answer exists only as the size of a bar can be answered.

## Why text alone misses it

Docling already puts a chart's labels into the text (`traverse_pictures`), but
not what the chart shows. The iMac report's p2 chart prints "445 kg Baseline
emissions" and the four segment names, and nothing about their sizes; GEP's
South Asia poverty forecast for 2028 is a bar with no number on it. Set C asks
four such questions (below). None of their answers occurs anywhere in the text.

## How (M7, `rag_studio/vision.py`)

| Step | What happens |
|---|---|
| Regions | Docling's picture items of at least 0.5% of the page: 288 (iMac 17, GPO 49, GEP 222). Smaller ones are icons |
| Crop | The picture's box, widened up to a short title printed within 24 pt above it ("D. Poverty" sits outside Docling's box), padded 4 pt |
| Image | Rendered from the PDF itself (vector), longer side about 1,024 px (150-400 dpi), identified by the sha256 of the PNG |
| Description | A vision model on the Colab T4 writes Title, Kind, Axes, Series, a Data table (`~` marks a value read from the scale), Largest and smallest, Takeaway. Prompt `vision-v1`, temperature 0, `num_ctx` 8192, at most 900 tokens |
| Cache | `data/cache/figures/<model>/<digest>/<prompt>/<image sha>.json`, bad outputs included, so a rebuild replays exactly; exported to `data/figures/<profile>.jsonl` |
| Passage | The printed figure heading (found by code: "FIGURE 2.7 SAR: Outlook"), "Figure, described by gemma4:12b from the image:", then the description; `type: "figure"`, the section label of where it sits (chart-panel titles, which Docling makes headings, are dropped from it) |
| Indexed | Every figure except those the model's own Kind line calls a photo or a logo: their facts are in their captions (spec §4) |
| Sources | "Figure (described by gemma4:12b)"; the page preview outlines the picture |

**Charts versus photos.** Neither printed text nor vector paths can tell them
apart here: GEP draws about 57 of its chart panels as raster images with no
text layer at all. The model's Kind line decides, and it is recorded.

## Set C: the answers, measured from the drawing

Every key was measured from the PDF's vector paths (`page.get_drawings()`:
rectangle and line-path heights, calibrated by the axis's own tick labels),
checked against a 300 dpi crop, and searched for in the whole text of its
document (not found). A visual reading of a 90 dpi render was 0.4 points off on
the poverty bar, so no key comes from looking.

| Q | Figure | Drawn values | Key (accepted) |
|---|---|---|---|
| C1 | iMac p2, the 445 kg baseline bar (no axis; 1.11 kg per point) | materials and process 190 kg, product use 155, manufacturing 67, transport 33 | materials and process emissions |
| C2 | GEP fig 2.7.D, South Asia poverty | 2025-28: 20.78, 20.07, 19.33, 18.93% | 18.6-19.2% |
| C3 | GEP fig 3.1.C, sovereign spreads | non-investment grade 2020: 525 bp (2022: 310); investment grade 191 / 36 | 495-555 bp |
| C4 | GEP fig 3.1.D, debt-distress risk 2026 | moderate 40.0, high 34.3, in distress 13.4, low 12.1% | moderate risk |

## Choosing the vision model (2026-10-23)

Each candidate described the four Set C figures; every value was read against
the drawing, by `./rag figures check` (a cell's row and column must name the
value; a reading counts within the tolerance and within half the value) **and
by hand**, because models write tables loosely (no outer pipes, the first
column's value inside the row label, series named by colour). Error is given as
a share of the axis span (PaperUnPlot's normalised measure, VLDB 2026 TaDA).

| Model, prompt | C1 iMac (stacked, no axis) | C2 poverty | C3 spreads | C4 debt risk (stacked) | Seconds per figure |
|---|---|---|---|---|---|
| qwen3.5:4b, v1 (this CPU) | 0 of 8: put each bar's total in a segment's row | 7/7, 1.1% | 4/4, 0.3% | cut at 900 tokens: corrected itself out loud | 80-191 |
| qwen3.5:4b, v2 (this CPU) | 0 of 8 | - | - | cut at 900 tokens | 276 |
| qwen3.5:9b, v1 (T4) | cut at 900 tokens | 7/7, 0.9% | 4/4, 0.3% | a one-line table; moderate 2015 written 72 (drawn 51.4), high 2026 20 (34.3) | 24-87 |
| **gemma4:12b, v1 (T4)** | **4 of 8, 7.3%**: segments rough, but materials named largest in the baseline and product use first in the iMac bar, both right | **7/7, 0.7%** | **4/4, 0.7%** | **8/8, 1.5%** | **19-20** once loaded |
| gemma4:12b, v2 (T4) | 1 of 8, 37%: shares only, in one line | 7/7, 0.7% | same values as v1 | same values as v1 | 18-53 |

**Chosen: gemma4:12b with `vision-v1`.** It is the only model that reads
stacked bars, it never failed an output, and it is the fastest (its whole
prompt is 517-645 tokens, the Qwen models' 790-1,010 for the same text, so the
difference is the image). `vision-v2` added stacked-bar
instructions and an instruction against correcting itself: it did not help
qwen3.5:4b and made gemma4:12b worse on C1, so it is kept only as a record.

Reading values is easy for the larger models on plain bars (within 1% of the
axis) and hard on stacks: the published WB-ChartExtract results agree (wrong
values and missing points are most errors; labels are almost never wrong).

## The full run (2026-10-23)

`./rag figures describe --model gemma4:12b` on the T4: 286 of 288 figures in
50 minutes (median 21.6 s per request with two in flight; the photos first at
about 3 s, GEP's chart panels about 13 s). 237 charts and diagrams go into the
index (133 bar charts, 67 line charts, 18 stacked bar charts, 8 diagrams, 4
infographics, 3 scatter plots, …); 49 photos and logos do not. Every photo or
logo call was checked against the image's colour count: none of them is a
chart (the GEP ones are the World Bank logo, the red chapter-opener globes and
the ISBN barcode). **Two outputs were rejected**, both cut at 900 tokens on
charts with very many points (GEP p85 panel C, a two-axis country bar chart;
GEP p114 panel A, a scatter plot): they are left out and counted. No output
used thinking. The descriptions are exported to
`data/figures/docling-structure-embeddinggemma2-fig.jsonl` (303 KB).

## Measured (ladder row 11)

Retrieval: the Set C figure passage is rank 1 for all four questions on the
figures index; Sets A and D keep every answer page at the same rank (MRR
0.871 → 0.892).

Answers (gemma4:e4b, prompt v3): **Set C 0 · 3 · 1 → 4 · 0 · 0**, Sets A+D
21 · 1 · 3 both ways (no question changed), the traps T1, T4 and T5 still
refused with a figure passage in front of them. C1's pass is weak (it lists the
four parts in order instead of naming the largest). Runs, answers and timings:
`docs/ladder.md`, row 11. **Done when Set C passes on the figures index**: it
does. The figures index is the app's default since 2026-10-23 (user).
