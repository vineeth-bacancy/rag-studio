"""Figures as text (M7): regions, the description cache and job, figure passages, Set C, the truth check."""
from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import pytest

from rag_studio import eval as evaluation
from rag_studio import profiles, vision
from rag_studio.clean import CleanPage

FIG = "docling-structure-embeddinggemma2-fig"

CHART = ("Title: D. Poverty\nKind: bar chart\nAxes: y percent of population 17-21\nSeries: bars (navy)\n"
         "Data:\nYear | Percent of population\n2025e | ~20.8\n2026f | ~20.1\n2027f | ~19.3\n2028f | ~18.9\n"
         "Largest and smallest: 2025e largest, 2028f smallest\nTakeaway: poverty falls slowly.")
PHOTO = "Kind: photo\nA pressman beside a printing press."


def page(items, text, *, sections=None, width=612.0, height=792.0):
    return CleanPage(doc_id="doc", page=3, width=width, height=height, text_raw=text, text=text, marks=[],
                     raw_marks=[], label="75", items=items, sections=sections or [{"start": 0, "path": []}],
                     kinds=[{"start": 0, "kind": "body"}])


def item(label, box, text, at):
    """A text item placed at `at` in the page text (start/end), or a picture (no text)."""
    if label == "picture":
        return {"label": label, "box": box, "start": None, "end": None}
    return {"label": label, "box": box, "start": at, "end": at + len(text)}


# --------------------------------------------------------------------------- regions

def test_regions_take_a_short_title_above_and_the_figure_heading_but_not_a_paragraph():
    heading, panel, para = "FIGURE 2.7 SAR: Outlook", "D. Poverty", "Body text " * 20
    text = f"{heading}\n{para}\n{panel}\n21"
    a, b, c = 0, len(heading) + 1, len(heading) + 1 + len(para) + 1
    items = [item("section_header", [300, 60, 560, 72], heading, a),
             item("text", [40, 200, 280, 330], para, b),               # left column, beside the chart: not a title
             item("section_header", [447, 324, 520, 331], panel, c),   # 6 pt above the picture
             item("picture", [447, 337, 573, 434], "", 0),
             item("text", [449, 346, 455, 352], "21", c + len(panel) + 1),   # a tick label inside the chart
             item("picture", [10, 10, 14, 14], "", 0)]                  # an icon: under 0.5% of the page
    sections = [{"start": 0, "path": ["Ch. 2", "South Asia"]}, {"start": c, "path": ["Ch. 2", "South Asia", panel]}]
    regs = vision.page_regions(page(items, text, sections=sections))
    assert [r.figure_id for r in regs] == ["doc:p3:f0"]       # the icon (f1) is skipped, not renumbered
    r = regs[0]
    assert r.title_above == panel and r.heading == heading
    assert r.crop[1] == pytest.approx(324 - vision.PAD)        # widened up to the panel title
    assert r.text_inside == 1
    assert r.section_path == ["Ch. 2", "South Asia"]           # the panel title is not the figure's section


def test_a_long_text_just_above_a_picture_is_not_its_title():
    para = "We've reduced emissions for iMac " * 8
    items = [item("text", [222, 200, 517, 240], para, 0), item("picture", [222, 247, 517, 717], "", 0)]
    r = vision.page_regions(page(items, para))[0]
    assert r.title_above is None and r.crop[1] == pytest.approx(247 - vision.PAD)


def test_dpi_keeps_small_panels_sharp_and_large_pictures_bounded():
    assert vision.dpi_for([0, 0, 126, 97]) == vision.DPI_RANGE[1]       # a GEP panel: 400 dpi
    assert vision.dpi_for([0, 0, 612, 577]) == vision.DPI_RANGE[0]      # a full-page photo: 150 dpi
    assert 150 <= vision.dpi_for([0, 0, 295, 470]) <= 400


# --------------------------------------------------------------------------- one description

def test_clean_description_refuses_junk():
    assert vision.clean_description("  Kind:   bar chart \n\n Data:  x ") == "Kind: bar chart\nData: x"
    for raw in (None, "", "   ", "<think>hm</think> Kind: photo", "x" * (vision.MAX_CHARS + 1)):
        with pytest.raises(vision.FigureBadOutput):
            vision.clean_description(raw)


class FakeVision:
    """Ollama's /api/chat with an image, as the runtime answers it."""

    def __init__(self, answers=None, digest="sha256:abc123def4567890"):
        self.answers = answers or {}
        self.calls: list[dict] = []
        self.digest = digest
        self.remote = True

    def model_info(self, model):
        return SimpleNamespace(digest=self.digest)

    def post_json(self, path, body):
        assert path == "/api/chat"
        msg = body["messages"][-1]
        assert body["think"] is False and len(msg["images"]) == 1
        assert body["options"]["temperature"] == 0.0 and body["options"]["num_ctx"] == vision.NUM_CTX
        self.calls.append(body)
        out = self.answers.get(msg["images"][0], CHART)
        if isinstance(out, dict):
            return out
        return {"message": {"content": out}, "prompt_eval_count": 700, "eval_count": 120, "done_reason": "stop"}


def test_ask_marks_a_cut_answer_as_bad():
    client = FakeVision(answers={"aW1n": {"message": {"content": "Kind: bar"}, "done_reason": "length"}})
    out = vision.ask(client, "m", b"img")
    assert out["bad"]["error"].startswith("The description was cut")
    ok = vision.ask(FakeVision(), "m", b"other")
    assert ok["description"].startswith("Title: D. Poverty") and ok["prompt_tokens"] == 700


# --------------------------------------------------------------------------- the job and the build

def fake_regions(n=3):
    return [vision.Region(figure_id=f"doc:p{i + 1}:f0", doc_id="doc", page=i + 1, box=[0, 0, 100, 100],
                          crop=[0, 0, 100, 100], area=0.02, text_inside=1, title_above=None,
                          heading="FIGURE 1.1 Test" if i == 0 else None, section_path=["Ch. 1"])
            for i in range(n)]


@pytest.fixture
def figures(tmp_data, monkeypatch):
    regs = fake_regions()
    monkeypatch.setattr(vision, "regions", lambda doc_ids=None: regs)
    monkeypatch.setattr(vision, "image_for", lambda r: (r.figure_id.encode(), f"sha-{r.figure_id}"))
    return regs


def test_descriptions_are_cached_and_a_second_job_asks_nothing(figures):
    import base64
    photo = base64.b64encode(b"doc:p2:f0").decode()
    bad = base64.b64encode(b"doc:p3:f0").decode()
    client = FakeVision(answers={photo: PHOTO, bad: {"message": {"content": ""}, "done_reason": "stop"}})
    s = vision.write_descriptions("m:1", client=client, in_flight=1)
    assert (s["described"], s["bad"], s["already"]) == (2, 1, 0)
    assert len(client.calls) == 3
    again = vision.write_descriptions("m:1", client=client)
    assert (again["described"], again["bad"], again["already"]) == (0, 0, 3)   # a bad output replays, never re-asked
    assert len(client.calls) == 3
    st = vision.status("m:1")
    assert (st["described"], st["indexed"], st["pictures"], st["bad"], st["missing"]) == (2, 1, 1, 1, 0)
    assert vision.last_job("m:1")["status"] == "done"


def test_figure_passages_come_from_the_cache_and_say_who_wrote_them(figures):
    prof = replace(profiles.get(FIG), figures="m:1")
    with pytest.raises(vision.FiguresMissing, match=r"3 of 3 figures .* ./rag figures describe --model m:1"):
        vision.passages(prof, {"doc": "Doc"})
    import base64
    client = FakeVision(answers={base64.b64encode(b"doc:p2:f0").decode(): PHOTO})
    vision.write_descriptions("m:1", client=client)
    out, meta, summary = vision.passages(prof, {"doc": "Doc"})
    assert [p.chunk_id for p in out] == ["doc:p1:f0", "doc:p3:f0"]          # the photo is not indexed
    assert summary == {"model": "m:1", "digest": "abc123def4567890", "regions": 3, "passages": 2,
                       "pictures": 1, "bad": 0}
    p = out[0]
    assert p.text.startswith("FIGURE 1.1 Test\nFigure, described by m:1 from the image:\nTitle: D. Poverty")
    assert p.label == "Document: Doc\nSection: Ch. 1"
    assert meta[p.chunk_id]["type"] == "figure" and meta[p.chunk_id]["section_path"] == "Ch. 1"
    assert (p.char_start, p.char_end) == (0, 0)


def test_export_and_import_round_trip(figures):
    prof = replace(profiles.get(FIG), figures="m:1")
    vision.write_descriptions("m:1", client=FakeVision())
    r = vision.export(prof)
    assert r["figures"] == 3
    import shutil
    shutil.move(str(vision.cache_root() / vision.slug("m:1")), str(vision.cache_root() / "moved"))
    assert vision.status("m:1")["described"] == 0
    with pytest.raises(vision.FiguresMissing, match=rf"./rag figures import {prof.id}"):
        vision.passages(prof, {"doc": "Doc"})                # a wiped cache points at the saved export
    assert vision.import_export(prof.id) == 3
    assert vision.status("m:1")["described"] == 3


# --------------------------------------------------------------------------- profiles

def test_only_the_figures_profile_records_figures():
    for p in profiles.PROFILES.values():
        assert ("figures" in p.config()) == bool(p.figures), p.id
    cfg = profiles.get(FIG).config()["figures"]
    assert cfg["model"] == vision.VISION_MODEL and cfg["prompt_sha256"] == vision.prompt_sha256()
    assert profiles.get(FIG).embedding == profiles.get("docling-structure-embeddinggemma2").embedding


# --------------------------------------------------------------------------- Set C and the truth check

@pytest.mark.parametrize("qid,text,want", [
    ("C1", "The largest part is materials and process emissions, about 190 kg.", "PASS"),
    ("C1", "Electricity for product use is the largest.", "WRONG"),
    ("C2", "About 18.9 percent in 2028.", "PASS"),
    ("C2", "Roughly 19% of the population.", "PASS"),
    ("C2", "19.3% in 2028.", "WRONG"),
    ("C2", "about 18.5 percent", "WRONG"),
    ("C3", "They rose by about 525 basis points.", "PASS"),
    ("C3", "about 190 basis points", "WRONG"),
    ("C3", "about 600 basis points", "WRONG"),
    ("C4", "Moderate risk had the largest share, about 40 percent.", "PASS"),
    ("C4", "High risk, about 34 percent.", "WRONG"),
])
def test_set_c_keys(qid, text, want):
    q = next(q for q in evaluation.load_sets()["set-c"]["questions"] if q["id"] == qid)
    assert evaluation.score_answer(q["expect"], text)["outcome"] == want


def test_every_set_c_question_names_its_figure_and_measured_values():
    for t in vision.truth_figures():
        assert t["region"].count(":") == 2 and t["span"] > 0 and t["tolerance"] > 0 and t["values"]


def test_truth_check_reads_values_from_the_lines_that_name_them():
    truth = next(t for t in vision.truth_figures() if t["qid"] == "C2")
    chk = vision.truth_check(CHART, truth)
    by = {v["bar"]: v for v in chk["values"]}
    assert by["2028f"]["reading"] == 18.9 and by["2028f"]["read"]
    assert by["2025e"]["error_share"] == pytest.approx(0.02 / 4, abs=1e-3)
    assert by["2027f"]["read"]
    # The January diamonds are not in this description's lines with numbers of their own: the closest
    # number on the line that names the year is used, and the report shows it beside the drawing.
    assert chk["of"] == 7 and chk["found"] == 7
    none = vision.truth_check("Kind: bar chart\nTakeaway: poverty falls.", truth)
    assert none["found"] == 0 and none["mean_error_share"] is None
    assert all(v["reading"] is None and not v["read"] for v in none["values"])


def test_truth_check_reads_table_cells_by_row_and_column():
    c3 = next(t for t in vision.truth_figures() if t["qid"] == "C3")
    desc = ("Kind: bar chart\nData:\n| Category | 2020 Pandemic (bp) | 2022 Fed Tightening Cycle (bp) |\n|---|---|---|\n"
            "| Investment Grade | ~190 | ~35 |\n| Non-Investment Grade | ~520 | ~310 |")
    by = {(v["bar"], v["series"]): v["reading"] for v in vision.truth_check(desc, c3)["values"]}
    assert by[("Investment grade", "2020 pandemic")] == 190          # not 520: "Non-" counts against that row
    assert by[("Non-investment grade", "2020 pandemic")] == 520
    assert by[("Investment grade", "2022 Federal Reserve tightening cycle")] == 35


def test_a_zero_is_not_a_reading_of_a_small_segment():
    c1 = next(t for t in vision.truth_figures() if t["qid"] == "C1")
    desc = ("Data:\n| Series | Baseline Emissions | iMac emissions |\n|---|---|---|\n"
            "| Materials and process emissions | 445 kg | 346 kg |\n| Transportation | ~99 kg | 0 kg |")
    by = {(v["bar"], v["segment"]): v for v in vision.truth_check(desc, c1)["values"]}
    assert by[("Baseline (445 kg)", "Materials and process emissions")]["reading"] == 445   # the total, misplaced
    t = by[("iMac (346 kg)", "Transportation")]
    assert t["reading"] == 0 and t["error"] < c1["tolerance"] and not t["read"]


def test_prompt_versions_keep_separate_caches(figures):
    vision.write_descriptions("m:1", client=FakeVision(), prompt_version="vision-v1")
    assert vision.status("m:1", "vision-v1")["described"] == 3
    assert vision.status("m:1", "vision-v2")["missing"] == 3
    client = FakeVision()
    vision.write_descriptions("m:1", client=client, prompt_version="vision-v2")
    assert len(client.calls) == 3
    assert "stacked bar" in client.calls[0]["messages"][-1]["content"]
