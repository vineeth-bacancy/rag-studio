# C: Docling and LanceDB for RAG Studio (research, 2026-10-14)

Status labels: **VERIFIED-DOC** (official docs or the project's own source at the pinned tag, URL given) · **VERIFIED-LIVE** (I ran it; command and output below) · **CLAIM** (vendor paper or blog) · **UNVERIFIED**.
"src@tag" means the GitHub source file at that release tag (see Sources). Docling was **not installed** and no models were downloaded. LanceDB was tested live in `scratchpad/lancetest/` (scripts in `scripts/`, data in `data*/`).

---------------------------------------------------------------------------------------------------
## 1. Docling

### 1.1 Facts
| Fact | Status | Source |
|---|---|---|
| `docling` 2.139.0 (uploaded 2026-10-14 12:21 UTC, today), MIT, Python >=3.10,<4. Related: docling-core 2.101.1 (10-08), docling-parse 7.22.3 (10-13), docling-ibm-models 4.0.3 (09-18) | VERIFIED-LIVE | `curl -s https://pypi.org/pypi/<pkg>/json` |
| `docling` is now a meta-package: `docling-slim[standard]==2.139.0`. The `standard` extra pulls torch, torchvision, docling-ibm-models, docling-parse, pypdfium2, rapidocr, transformers (through ibm-models), docling-core[chunking] and others | VERIFIED-LIVE | PyPI `requires_dist` of docling and docling-slim |
| Model licences: docling-layout-heron Apache-2.0; docling-models (TableFormer) cdla-permissive-2.0 and apache-2.0; TableFormerV2 Apache-2.0 | VERIFIED-LIVE | HF API `https://huggingface.co/api/models/<repo>` tags |
| Dependency footprint (resolved only, nothing installed): default PyPI index gives **121 packages, about 3.2 GB of wheels**, including 19 nvidia/cuda/triton packages (torch 555 MB, cudnn 553 MB, cublas 423 MB...). With the PyTorch CPU index: **102 packages, about 424 MB of wheels** (226 MB plus torch 2.14.1+cpu at 196 MB plus torchvision 0.29.1+cpu at 1.7 MB). The installed (unpacked) size was not measured | VERIFIED-LIVE (sizes); UNVERIFIED (installed size) | `uv pip compile req.in --python-version 3.12 --python-platform x86_64-manylinux_2_28 [--extra-index-url https://download.pytorch.org/whl/cpu --index-strategy unsafe-best-match]`, then wheel sizes from PyPI JSON and an HTTP HEAD request to download.pytorch.org |
| CPU-only install is documented: `pip install docling --extra-index-url https://download.pytorch.org/whl/cpu`. In uv you add an `explicit = true` `[[tool.uv.index]]` and pin `torch`/`torchvision` to it in `[tool.uv.sources]`. The file `torch-2.14.1+cpu-cp312-cp312-manylinux_2_28_x86_64.whl` exists | VERIFIED-DOC / VERIFIED-LIVE | docling installation page; uv docs `concepts/projects/dependencies.md`; `https://download.pytorch.org/whl/cpu/torch/` |
| Default PDF backend is `ThreadedDoclingParseDocumentBackend` (`PdfBackend.THREADED_DOCLING_PARSE = "docling_parse"`). The alternative is `PYPDFIUM2`. DLPARSE_V1/V2/V4 are deprecated and map to the threaded docling-parse backend | VERIFIED-DOC | src@v2.139.0 `document_converter.py` L301-304, `pipeline_options.py` L1286-1312 |
| Body text comes from the **PDF text layer** (docling-parse cells), and the layout model only groups those cells. The default `OcrMode.DEFAULT` (= `PDF_AWARE_LAYOUT_REGIONS`) runs OCR only on layout regions that have no PDF text cells, with PDF text taking priority. `OcrMode.FULL_PAGE` (formerly `force_full_page_ocr`) discards the PDF cells and keeps only the OCR cells | VERIFIED-DOC | src@v2.139.0 `models/base_ocr_model.py` L237-264, L570-600; `pipeline_options.py` L104-120 |
| `PdfPipelineOptions` defaults: `do_ocr=True`, `do_table_structure=True`, `table_structure_options=TableStructureOptions(mode=ACCURATE, do_cell_matching=True)`, `images_scale=1.0`, `generate_page_images=False`, `generate_picture_images=False`, `generate_parsed_pages=False`, `force_backend_text=False`, `do_formula_enrichment=False`, `do_code_enrichment=False`, `do_picture_classification=False`, `do_picture_description=False`, `do_chart_extraction=False`, `use_reading_order_separators=True`, `heading_hierarchy_options.enabled=False`, `document_timeout=None`, `artifacts_path=None`, `enable_remote_services=False`, layout/ocr/table batch sizes 4, `layout_options` = `LayoutObjectDetectionOptions` preset `layout_heron_default` | VERIFIED-DOC | src@v2.139.0 `pipeline_options.py` L1375-1558, L2080-2317 |
| TableFormer has modes `FAST`/`ACCURATE`. There is also a `TableStructureV2Options` (TableFormer V2, with only `do_cell_matching`; the code comment says that `True` "can break table output if PDF cells are merged across table columns") and a Granite-Vision table option | VERIFIED-DOC | `pipeline_options.py` L122-191 |
| `AcceleratorOptions(num_threads=4, device="auto")`. Devices are auto/cpu/cuda/mps/xpu. The thread count can come from `DOCLING_NUM_THREADS` or else `OMP_NUM_THREADS` | VERIFIED-DOC | src@v2.139.0 `datamodel/accelerator_options.py` |
| OCR renders the page at 72 DPI times `scale` (default 3, i.e. 216 DPI). `OcrAutoOptions` probes for an engine (ocrmac, rapidocr+onnxruntime, easyocr, ...). The CPU dependency set includes rapidocr but **not** onnxruntime, so it is unclear which engine "auto" would pick | VERIFIED-DOC / UNVERIFIED (which engine) | `pipeline_options.py` L245-257; `models/stages/ocr/auto_ocr_model.py` |
| `converter.convert(source, page_range=(start, end), max_num_pages=..., max_file_size=..., raises_on_error=True)`. `page_range` is 1-based and inclusive, default `(1, sys.maxsize)` | VERIFIED-DOC | `document_converter.py` L632-641; `datamodel/settings.py` L21-23; `pipeline/base_pipeline.py` L353-355 |
| `ProvenanceItem.page_no` is **1-based** (`Page(page_no=i + 1)`). `PageItem` has only `size`, `image` and `page_no`, so the printed page label is **not** carried | VERIFIED-DOC | `base_pipeline.py` L355; docling-core src@v2.101.1 `types/doc/common/reference.py` L183-203 |
| `ProvenanceItem(page_no, bbox: BoundingBox, charspan: (int, int) 0-indexed)`. The PDF pipeline writes bboxes with a **BOTTOMLEFT** origin (`to_bottom_left_origin(page_height)`), so the UI needs `bbox.to_top_left_origin(h)`. Tables get `charspan=(0, 0)` | VERIFIED-DOC | docling `readingorder_model.py` L72-210; core `common/scalars.py` L10 |
| `DocItemLabel` values include: caption, chart (deprecated), footnote, formula, list_item, page_footer, page_header, picture, section_header, table, text, title, document_index, code, checkbox_*, form, key_value_region, paragraph, reference, field_*, marker, ... | VERIFIED-DOC | core `types/doc/labels.py` |
| `ContentLayer` is BODY / FURNITURE / BACKGROUND / INVISIBLE / NOTES, and `DEFAULT_CONTENT_LAYERS={BODY}`. **page_header and page_footer are put in FURNITURE**, so `iterate_items`, `export_to_markdown` and the chunkers skip them by default | VERIFIED-DOC | core `common/content_layer.py`; docling `readingorder_model.py` L120-125, L646-648 |
| `doc.iterate_items(root=None, with_groups=False, traverse_pictures=False, page_no=None, included_content_layers=None)` yields `(item, level)` in reading order | VERIFIED-DOC | core `document.py` L3458-3475 |
| `TableItem.export_to_dataframe(doc)`, `.export_to_markdown(doc)`, `.export_to_html(doc)`, `.export_to_otsl(...)`. Floating items (tables, pictures) have `captions`, `footnotes`, `references`, `caption_text(doc)` and `get_image(doc, prov_index=0)` | VERIFIED-DOC | core `items/table/table.py` L83-199, `items/node.py` L171-221 |
| `PictureItem.meta` has `.classification.predictions[*].class_name/confidence`, `.description.text`, `.tabular_chart`. The old `annotations` field is deprecated. Classifier `docling-project/DocumentFigureClassifier-v2.5` is 33 MB. The default description model is SmolVLM-256M-Instruct; its repo is 3.5 GB in total with all variants, and how much Docling actually downloads was not checked | VERIFIED-DOC / VERIFIED-LIVE (sizes) / UNVERIFIED (download subset) | core `items/picture/picture.py`, `picture/meta.py`, `common/meta.py`; HF API |
| `document_index` (a TOC) is handled as a table type: it goes through TableFormer and becomes a `TableItem` with label `document_index` | VERIFIED-DOC | docling `utils/layout_postprocessor.py` L118-126; `table_structure_model.py` L218; `readingorder_model.py` L134-152 |
| `sanitize_text` joins the lines of one cluster. A line-end hyphen attached to a word is dropped ("algo-"+"rithms"). It then normalises quotes and expands ligatures U+FB00-FB06, absorbing the stray space ("ﬁ eld" becomes "field") | VERIFIED-DOC | docling `models/stages/page_assemble/page_assemble_model.py` L115-177 |
| The "visible text" filter only drops text drawn in render modes 3 and 7 (invisible), and it is only used to compute OCR regions. There is **no occlusion test** (text painted over later). `create_orphan_clusters=True` turns text cells outside any layout box into their own clusters | VERIFIED-DOC | docling `backend/pdf_backend.py` L39-45, `backend/docling_parse_backend.py` L60-80, `pipeline_options.py` L1605-1640 |
| docling-parse, when the ToUnicode CMap has no entry for a code: for a *symbolic* font it emits `GLYPH<n>`, which the default `keep_glyphs=False` turns into a space; for a non-symbolic font it falls back to the declared encoding; a meaningful glyph name such as `/Th` is used as the text | VERIFIED-DOC | docling-parse src@v7.22.3 `tests/test_unit_tounicode_fallback.py`, `tests/test_unit_gid_glyph_names.py` |
| Heading levels: the layout model only says "section_header", so every heading is level 1 unless `HeadingHierarchyOptions(enabled=True)` is set. That option uses PDF bookmarks, then numbering, then font style (style needs `generate_parsed_pages=True`) | VERIFIED-DOC | `pipeline_options.py` L1957-2079 |
| Chunkers: `HierarchicalChunker` gives one chunk per item, with tables serialised by `TripletTableSerializer`. `HybridChunker(tokenizer=HF "sentence-transformers/all-MiniLM-L6-v2" by default, max_tokens from the tokenizer, merge_peers=True, repeat_table_header=True, omit_header_on_overflow=False, always_emit_headings=False)`. `DocChunk.meta` has `doc_items` (with prov, so pages and bboxes), `headings: list[str]` and `origin` (`captions` is deprecated). `chunker.contextualize(chunk)` gives the headings plus the text | VERIFIED-DOC | core `transforms/chunker/{hybrid_chunker,hierarchical_chunker,doc_chunk,base}.py`, `tokenizer/huggingface.py` L91-93 |
| Models: layout `docling-project/docling-layout-heron` (RT-DETR ResNet50), 171.7 MB safetensors (an ONNX variant is 171.2 MB). TableFormer `docling-project/docling-models@v2.3.0`: accurate 212.8 MB, fast 145.5 MB. TableFormerV2 209.7 MB | VERIFIED-LIVE (HF tree API) / VERIFIED-DOC (repo ids) | `stage_model_specs.py` L1011-1023; `table_structure_model.py` L36-37, L102-111 |
| Cache: models fetched on demand go through `huggingface_hub.snapshot_download` (the HF cache). `docling-tools models download` puts them in `~/.cache/docling/models`. For offline use: `PdfPipelineOptions(artifacts_path=...)` | VERIFIED-DOC | `models/utils/hf_model_download.py`; `utils/model_downloader.py` L124-195; docs advanced_options and FAQ |
| Speed, from Docling Technical Report v5 Table 1 (225 pages, OCR off, **2024 Docling v1**): with the native backend, M3 Max reached 1.27 p/s on 4 threads and 1.34 p/s on 16; Xeon E5-2690 reached 0.60 p/s on 4 and 0.92 p/s on 16. With pypdfium it reached 0.94-2.45 p/s. Peak RSS was about 6.2 GB (native) versus about 2.5 GB (pypdfium). On that basis 294 pages would take about 4-8 minutes on CPU | CLAIM | arXiv 2408.09869v5 §4 |

### 1.2 API sketch (M2)
```python
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions, TableFormerMode, HeadingHierarchyOptions
from docling.datamodel.accelerator_options import AcceleratorOptions, AcceleratorDevice
from docling_core.types.doc import DocItemLabel, TableItem, PictureItem
opts = PdfPipelineOptions(do_ocr=False, do_table_structure=True, generate_page_images=True,
                          generate_picture_images=True, images_scale=2.0, generate_parsed_pages=True,
                          heading_hierarchy_options=HeadingHierarchyOptions(enabled=True),
                          accelerator_options=AcceleratorOptions(device=AcceleratorDevice.CPU, num_threads=8))
opts.table_structure_options.mode = TableFormerMode.ACCURATE      # do_cell_matching=True is the default
conv = DocumentConverter(format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=opts)})
doc = conv.convert("gep.pdf", page_range=(1, 20)).document         # 1-based, inclusive
for item, level in doc.iterate_items():                             # BODY layer only; add ContentLayer.FURNITURE to inspect headers
    p = item.prov[0]                                                # p.page_no is 1-based; p.bbox is BOTTOMLEFT
    if isinstance(item, TableItem): df = item.export_to_dataframe(doc); cap = item.caption_text(doc)
    if isinstance(item, PictureItem): img = item.get_image(doc); cap = item.caption_text(doc)
from docling.chunking import HybridChunker                          # re-export of docling_core.transforms.chunker
for ch in HybridChunker(max_tokens=512).chunk(doc): ch.meta.headings, ch.meta.doc_items, HybridChunker().contextualize(ch)
```
(`docling.chunking` exists in src@v2.139.0. Passing `max_tokens` to the constructor follows `hybrid_chunker.py` L105-118. Neither was run. **UNVERIFIED-LIVE**)

### 1.3 The 15 PyMuPDF defects: hypotheses for Docling (to be measured in M2)
| # | Defect | Hypothesis | Reasoning (from the facts above) |
|---|---|---|---|
| 1 | Tables flattened one cell per line | **Fixed** (mostly) | TableFormer ACCURATE builds a cell grid that can be exported as a dataframe or markdown. Risks: borderless or multi-page tables, and the cell-matching caveat |
| 2 | Running headers/footers in body | **Fixed** (mostly) | They are labelled page_header/page_footer and moved to FURNITURE, which is excluded by default. Risk: a header the model misreads as body text |
| 3 | Line-break hyphens | **Partly** | Joined only *inside one cluster*, not across columns or pages. It also wrongly joins real compounds split at a line end ("post-"/"pandemic" becomes "postpandemic") |
| 4 | Ligatures | **Fixed** if the text layer gives U+FB0x characters (expanded); **not** if the ligature glyph is unmapped (same class as #7) | sanitize_text ligature map |
| 5 | Three page numbers | **Partly** | We get a 1-based physical `page_no`; index = page_no-1. There is no printed label, so we keep our own PyMuPDF page-label map |
| 6 | Headings/captions out of order | **Fixed/partly** | Layout plus reading-order model; captions are linked to their table or picture. Heading levels are flat unless heading-hierarchy is enabled |
| 7 | Dropped "Th" (AGaramondPro CMap lacks <01>) | **Not fixed by default** | Text comes from the text layer. docling-parse may give a space, the glyph name, or nothing, depending on font type, so it may differ from PyMuPDF. The fix path is `OcrMode.FULL_PAGE` on the affected pages (slow), or our own repair step |
| 8 | Charts and images | **Partly** | PictureItem with bbox, caption and image. Classification is cheap (33 MB). Description or chart extraction needs a VLM, which is slow on CPU. Text labels inside charts: UNVERIFIED |
| 9 | Footnote markers glued to words | **Not fixed** (likely) | I found no superscript handling in the paths I read. Footnote *blocks* do get the `footnote` label |
| 10 | TOC pages and dot leaders | **Partly** | The TOC becomes a `document_index` TableItem, so we can drop it by label. Dot leaders may remain if it is missed |
| 11 | Letter-spaced display text | **Unknown** | Depends on how docling-parse groups words |
| 12 | Footer printed twice | **Likely fixed** if both copies are labelled footer (FURNITURE); no de-duplication otherwise | |
| 13 | Hidden text under a later full-page fill | **Not fixed** (likely) | There is no occlusion test, and orphan cells become clusters. `create_orphan_clusters=False` might hide it, at the risk of losing real text |
| 14 | Near-empty pages | **Neutral** | They yield few or no items. We keep our own minimum-character rule |
| 15 | Different structure per PDF | **Partly** | One label set for all three PDFs, but quality will vary per PDF. Heading hierarchy works best when the PDFs have bookmarks |

### 1.4 Still to verify (M2)
1. Install `docling` with CPU torch in the project; record the installed size, the first-run model download size and the cache paths; run `HF_HUB_OFFLINE=1` with `artifacts_path`.
2. Measure pages/s and peak RSS on our 294 pages (8 threads), for ACCURATE vs FAST, with and without page images.
3. Count the 15 defects against the PyMuPDF baseline, especially #7 (count "Th"/"th" tokens per page against a page image), #13 and #3 (the false compound joins).
4. Which OCR engine "auto" picks in our dependency set, and FULL_PAGE OCR timing on the GEP pages.
5. Run the HybridChunker headings with heading-hierarchy on; check that `docling.chunking` imports; check how tables look in chunk text.

### 1.5 Recommendation (Docling)
Use Docling as the **"layout-aware" parser profile** next to the PyMuPDF baseline, not as a replacement. Settings:
- `do_ocr=False`, ACCURATE tables, `generate_page_images=True` (for the bbox overlays in the UI) and heading hierarchy on.
- Install torch from the CPU index (about 424 MB of wheels instead of about 3.2 GB).
- Store the DoclingDocument JSON (`save_as_json`) per PDF so re-chunking never re-parses.
- Keep printed page labels and the "Th" repair in our own code.
- Drop `document_index` items and FURNITURE before chunking; render them greyed out in the UI.

---------------------------------------------------------------------------------------------------
## 2. LanceDB (Python)

**Live setup:** `uv run --no-project --python 3.12 --with lancedb==0.40.0 --with pyarrow --with numpy python scripts/lance_probe{,2,3}.py <data_dir>`. Logs are in `run1.log` and `run2.log`. The machine has 8 CPU threads (`nproc`). Data: 1,000 random unit vectors (768-d) and random text.

### 2.1 Facts
| Fact | Status | Source |
|---|---|---|
| lancedb **0.40.0** (2026-10-07), Apache-2.0. Deps: pyarrow>=16, numpy, pydantic, lance-namespace. pyarrow 26.0.0 was resolved. No pandas or pylance is needed for `to_arrow`/`to_list`, FTS or hybrid | VERIFIED-LIVE | PyPI JSON; probe "versions" section |
| `lancedb.connect("<folder>")`, then `db.create_table(name, data=pa.Table, schema=pa.schema, mode="create"|"overwrite", exist_ok=False)`. A `pa.list_(pa.float32(), 768)` column is stored as `fixed_size_list<float>[768]` and found automatically as the vector column | VERIFIED-LIVE | probe |
| With no metric given (and no index), the metric is **l2** and `_distance` is the *squared* L2 (1.755 = 2x0.8775). `distance_type("cosine")` gives 1-cos (0.8775). `"dot"` gave 0.8775 on unit vectors, i.e. 1-dot | VERIFIED-LIVE | probe "exact" section |
| With no index, search is an exact flat scan: top-10 cosine over 1000x768 took a **median of 6.9 ms** (run 1) or 10.5 ms (run 2), with a cold maximum of 78 ms. `bypass_vector_index()` forces a flat scan even when an index exists | VERIFIED-LIVE / VERIFIED-DOC | probe; docs indexing/vector-index |
| Index types: `IVF_FLAT, IVF_SQ, IVF_PQ, IVF_RQ, IVF_HNSW_SQ, IVF_HNSW_PQ, IVF_HNSW_FLAT`. Parameters: `num_partitions`, `target_partition_size`, `num_sub_vectors` (default dim//16), `num_bits=8`, `max_iterations=50`, `sample_rate=256`, `m=20`, `ef_construction=300` | VERIFIED-LIVE (signature) | `inspect.signature(LanceTable.create_index)`, `lancedb.index.IvfPq/IvfHnswSq/IvfFlat` |
| `create_index(metric=..., index_type=...)` is deprecated; use `create_index("vector", config=IvfPq(distance_type="cosine"))`. `create_fts_index` has been deprecated since 0.25.0; use `create_index("text", config=FTS(...))`. `cleanup_old_versions()` raises ImportError without pylance, so use `optimize(cleanup_older_than=...)` | VERIFIED-LIVE | warnings and errors in run1/run2 |
| Minimum rows: IVF_PQ built without error on 100, 255, 256 and 300 rows (`num_partitions=1, num_sub_vectors=48`). Asking for 8 partitions on 1,000 rows logged "training 3 of the 8 requested IVF partitions". So there is no hard minimum, but the partition count is capped. The exact rule is UNVERIFIED | VERIFIED-LIVE | probe "minimum rows"; Rust WARN log |
| Search knobs: `.nprobes(n)` (docs: default 20), `.minimum_nprobes`, `.refine_factor(r)`, `.ef(n)` (HNSW), `.distance_range(lo, hi)`, `.fast_search()` (indexed rows only), `.with_row_id(True)` | VERIFIED-LIVE (methods exist) / VERIFIED-DOC (defaults) | builder `hasattr` check; docs vector-index |
| **FTS is native only:** `use_tantivy=True` gives `ValueError: Tantivy-based FTS has been removed` | VERIFIED-LIVE | probe |
| `FTS()` defaults: `base_tokenizer="simple"` (also raw, whitespace, ngram, icu, icu/split, jieba/*, lindera/*), `language="English"`, `max_token_length=40`, `lower_case=True`, `stem=True`, `remove_stop_words=True`, `ascii_folding=True`, `with_position=False`, `custom_stop_words=None`, ngram 3/3, `prefix_only=False`, `block_size=128`. The index details also show `split_on_numerics=False` and `preserve_original=False`, but these are not exposed as parameters | VERIFIED-LIVE | `inspect.signature(lancedb.index.FTS)`; `list_indices()[0].index_details` |
| FTS results carry **`_score` (BM25)**. A query in double quotes is a phrase query, and phrase queries **need `with_position=True`**; otherwise you get `ValueError ... position is not found but required for phrase queries`. With default stop-word removal, a search for "the" returns nothing. `PhraseQuery(text, column)` and `MatchQuery(text, column, fuzziness=)` objects also work | VERIFIED-LIVE | probes 1 and 2 |
| Tokenisation seen (simple, English stemming, positions, stop words kept): "2.5%" becomes 2 + 5 and also matches "2.5 per cent". `"FY 2018"` as a phrase hits only the FY 2018 text. "FY2018" is one token. "COVID-19" becomes covid + 19. "e-commerce" becomes e + commerc and also hits a stray "e". "ecommerce" gives 0 hits. growing/grow match each other. Fuzzy "inflaton" with fuzziness 1 gave 0 hits, probably because the index holds stemmed terms (UNVERIFIED cause). **"theory" gives 0 hits on the "e eory" text from defect 7** | VERIFIED-LIVE | probe 2 output |
| Freshness: rows appended after the FTS index is built are found right away (by a flat scan of unindexed rows), and `index_stats("text_idx")` shows `num_unindexed_rows=5`. Rows updated by `merge_insert` drop out of the index (950 indexed, 50 unindexed) but are still found. `optimize()` brings this back to 1000/0, so **the index does not have to be rebuilt after appends** | VERIFIED-LIVE / VERIFIED-DOC | probes 1 and 2; docs indexing/reindexing |
| Hybrid: `tbl.search(query_type="hybrid").vector(v).text(q).rerank(RRFReranker(K=60, return_score="relevance"|"all")).limit(k)`. By default only `_relevance_score` is returned. `"all"` adds `_distance` and `_score`, set to None for rows that were missing from that list. The fused list is cut to `limit`, so the two separate input lists are **not** returned | VERIFIED-LIVE | probes 1 and 2; docs reranking/rrf |
| RRF score = Σ 1/(rank + K) with **1-based** rank: a row ranked first in both lists got 0.032787 = 2/61. My own RRF (dense top-5 + BM25 top-5) gave the same order | VERIFIED-LIVE | probe "hybrid" (`inspect.getsource(RRFReranker)`) |
| Latency at 1k rows: FTS median 4.6 ms; hybrid median 13.7 ms; `create_fts_index` 291 ms | VERIFIED-LIVE | probe 1 |
| Filters: `.where("page <= 10")` prefilters by default (5/5 results satisfied it). `where(..., prefilter=False)` returned **0 rows** for top-5, because postfiltering trims after top-k. `where` also works on FTS | VERIFIED-LIVE | probe "filters" |
| Versions: create, add, index build and merge each make a new version. `list_versions()`, `checkout(int or tag)`, `checkout_latest()`, `restore(v)` (this writes a new version), and `tags.create/list/get_version/update/delete`. **Cleanup refuses to remove tagged versions**, with an error suggesting `error_if_tagged_old_versions=false`. After deleting the tag, `optimize(cleanup_older_than=timedelta(0))` cut versions [1..5] down to [7, 8] (with a warning that concurrent readers may fail) | VERIFIED-LIVE | probes 1 and 3 |
| `merge_insert("chunk_id").when_matched_update_all(where="target.content_hash != source.content_hash").when_not_matched_insert_all().when_not_matched_by_source_delete(condition=None)` returns `MergeResult(version, num_updated_rows, num_inserted_rows, num_deleted_rows, ...)`. The full-sync variant deleted 105 rows | VERIFIED-LIVE | probe "merge_insert"; signatures |
| `db.rename_table` raises **`NotImplementedError: rename_table is not supported in LanceDB OSS`**. `drop_table` works | VERIFIED-LIVE | probes 1 and 3 |
| `create_table(..., mode="overwrite")` or `add(..., mode="overwrite")` makes a new version **with no indices**. A reader that checked out the old version still reads the old rows and the old FTS index; a new `open_table` sees the new data | VERIFIED-LIVE | probes 1 and 3 |

**Recall@10 against exact search** (cosine; 20 queries; 1,000 x 768). This is random data, which is close to the worst case for ANN, so the numbers are pessimistic for real embeddings (UNVERIFIED for those).

| Index (build time) | default | +refine 5/10 | nprobes 64 + refine 10 |
|---|---|---|---|
| IVF_PQ auto (0.47 s), random queries / near-duplicate queries | 0.32 / 0.36 | 0.675 (r5) / 0.835 (r10) | 0.85 / 0.835 |
| IVF_PQ 8→3 partitions, 96 sub-vectors (1.4 s), random | 0.435 | 0.855 (r5) | 0.97 |
| IVF_PQ 2 partitions, 96 sub-vectors, near-duplicate | 0.525 | 0.965 (r10) | 0.965 |
| IVF_HNSW_SQ (0.2-0.3 s), random / near-duplicate | 0.635 / **0.985** | 0.965 / 1.0 | 1.0 / 1.0 |
| IVF_FLAT 8 or 4 partitions (0.05 s) | **1.0** (0.62 with nprobes=4 of 8) | 1.0 | 1.0 |

Latency for every index was about 4-15 ms, which is no faster than the exact scan at this size.

### 2.2 API sketch (M3)
```python
import lancedb, pyarrow as pa
from lancedb.index import FTS, IvfHnswSq
db = lancedb.connect("data/lancedb")
schema = pa.schema([("chunk_id", pa.string()), ("doc_id", pa.string()), ("page_no", pa.int32()), ("page_label", pa.string()),
                    ("section", pa.list_(pa.string())), ("label", pa.string()), ("text", pa.string()), ("content_hash", pa.string()),
                    ("vector", pa.list_(pa.float32(), 768))])
t = db.create_table(f"chunks__{profile}__{build_id}", data=rows, schema=schema)
t.create_index("text", config=FTS(with_position=True, remove_stop_words=False, stem=True), replace=True)
t.wait_for_index(["text_idx"])
dense = t.search(qvec).distance_type("cosine").limit(50).select(["chunk_id", "_distance"]).to_list()
bm25  = t.search(qtext, query_type="fts").limit(50).select(["chunk_id", "_score"]).to_list()
t.tags.create(f"build-{build_id}", t.version)
```

### 2.3 Still to verify
- **M3:** exact-scan latency at 10k rows (I expect about 10x the 1k figure; UNVERIFIED). Whether two FTS indexes on different columns of one table work together with `fts_columns=`. Whether `select([... "_distance"])` or `disable_scoring_autoprojection` silences the `_distance` autoprojection warning. The behaviour of `when_not_matched_by_source_delete("doc_id = 'X'")` with a scoped condition.
- **M4:** concurrent reader and writer while FastAPI is serving (snapshot semantics were only tested within one process). How big the tables get on disk (1k rows took 19.6 MB, including many versions).

---------------------------------------------------------------------------------------------------
## 3. Recommendation for RAG Studio

1. **Table layout:**
   - One LanceDB folder (`data/lancedb/`). One table per index profile *and* build: `chunks__{parser}-{chunker}-{embedder}__{build_id}`.
   - Columns as in §2.2, plus `token_count` and `bbox` (`list<float32>`, top-left origin after converting Docling's coordinates).
   - `chunk_id` = hash(doc_id, profile, ordinal). `content_hash` = sha256(text + embedder id).
   - A small registry file (`active.json`, written atomically with temp file + `os.replace`) maps profile to the active table. **We have to do this ourselves because `rename_table` is not available in OSS.**
2. **Vector search:**
   - Use **exact flat search** for 1-10k chunks (about 7-10 ms per query at 1k). Always set `distance_type("cosine")` explicitly, because the default is l2.
   - Build ANN indexes only as a teaching toggle: IVF_HNSW_SQ or IVF_FLAT. Show "recall@10 vs `bypass_vector_index()`" in the UI. Avoid IVF_PQ at this scale (recall 0.3-0.5 without refine).
3. **FTS for English reports:**
   - Use `FTS(base_tokenizer="simple", language="English", stem=True, remove_stop_words=False, with_position=True, lower_case=True, ascii_folding=True, max_token_length=40)`.
   - Keeping stop words and positions makes phrase queries work (`"FY 2018"`, `"rate of growth"`). Stemming helps report prose.
   - Numbers: "FY2018" stays one token, while "2.5%" splits into "2" + "5". A quoted phrase query still works, but it also matches "2.5 per cent".
   - If exact numeric matching matters in evaluation, add a second column `text_exact` with a `whitespace`, no-stem FTS index. This needs verifying in M3.
   - Show the tokens in the UI, since the defect-7 "e eory" problem is directly visible as BM25 misses.
4. **Hybrid: fuse in our own code.**
   - Run the dense query (top 50) and the BM25 query (top 50) separately, display both lists, then apply RRF ourselves with K=60 and 1-based ranks. This reproduces LanceDB's `RRFReranker` order (verified).
   - LanceDB's own hybrid hides the input lists and truncates to `limit`. Keep it only as a test oracle (`return_score="all"`).
5. **Rebuild and swap:**
   - (a) Create a new build table.
   - (b) Add all rows.
   - (c) Create the FTS (and optional vector) index, then `wait_for_index`.
   - (d) Run smoke queries and check `index_stats(...).num_unindexed_rows == 0`.
   - (e) Tag the version `build-{id}`.
   - (f) Swap the registry pointer.
   - (g) Keep the previous table for rollback and drop older ones.
   - Do **not** swap with `mode="overwrite"` on the live table: the indices disappear until they are rebuilt.
6. **Incremental re-ingest of a changed PDF** (same profile):
   - Run `merge_insert("chunk_id")` with update-on-hash-change, insert-new, and delete-by-source scoped to that `doc_id`.
   - Then run `optimize()` to fold the changed rows back into the FTS index. They stay searchable in the meantime.
   - Before running cleanup with a short retention, delete the tags you no longer want.

## Sources (all accessed 2026-10-14)
- PyPI JSON: https://pypi.org/pypi/docling/json, .../docling-slim/json, .../docling-core/json, .../docling-parse/json, .../docling-ibm-models/json, .../lancedb/json, .../torch/2.14.1/json
- Docling source @v2.139.0: https://github.com/docling-project/docling/tree/v2.139.0 (`docling/datamodel/pipeline_options.py`, `accelerator_options.py`, `settings.py`, `stage_model_specs.py`, `document_converter.py`, `pipeline/base_pipeline.py`, `models/base_ocr_model.py`, `models/stages/ocr/auto_ocr_model.py`, `models/stages/page_assemble/page_assemble_model.py`, `models/stages/reading_order/readingorder_model.py`, `models/stages/table_structure/table_structure_model{,_v2}.py`, `backend/pdf_backend.py`, `backend/docling_parse_backend.py`, `utils/layout_postprocessor.py`, `utils/model_downloader.py`, `models/utils/hf_model_download.py`)
- docling-core source @v2.101.1: https://github.com/docling-project/docling-core/tree/v2.101.1 (`docling_core/types/doc/{labels,document}.py`, `common/{reference,content_layer,scalars,meta}.py`, `items/{node,text}.py`, `items/table/table.py`, `items/picture/{picture,meta}.py`, `transforms/chunker/{base,doc_chunk,hierarchical_chunker,hybrid_chunker}.py`, `tokenizer/huggingface.py`)
- docling-parse tests @v7.22.3: https://github.com/docling-project/docling-parse/blob/v7.22.3/tests/test_unit_tounicode_fallback.py, .../tests/test_unit_gid_glyph_names.py
- Docling docs (via ctx7 `/docling-project/docling`, `/websites/docling-project_github_io_docling`): https://docling-project.github.io/docling/getting_started/installation, /usage/advanced_options, /faq, /usage/gpu, /usage/model_catalog
- Docling Technical Report v5: https://arxiv.org/abs/2408.09869 (PDF v5, §4 Table 1)
- Hugging Face model trees: https://huggingface.co/api/models/docling-project/docling-layout-heron/tree/main, .../docling-project/docling-models/tree/v2.3.0, .../docling-project/TableFormerV2/tree/main, .../docling-project/DocumentFigureClassifier-v2.5/tree/main, .../HuggingFaceTB/SmolVLM-256M-Instruct/tree/main
- PyTorch CPU wheels: https://download.pytorch.org/whl/cpu/torch/ ; uv index docs: https://github.com/astral-sh/uv/blob/main/docs/concepts/projects/dependencies.md (via ctx7 `/astral-sh/uv`)
- LanceDB docs (ctx7 `/websites/lancedb` and the `.md` pages): https://docs.lancedb.com/search/full-text-search, /indexing/fts-index, /indexing/vector-index, /search/vector-search, /search/hybrid-search, /reranking/rrf, /indexing/reindexing, /tables/versioning, /tables/update, /performance
- Live probes: `scratchpad/lancetest/scripts/lance_probe.py`, `lance_probe2.py`, `lance_probe3.py`, `wheel_sizes.py`; logs `scratchpad/lancetest/run1.log`, `run2.log`
