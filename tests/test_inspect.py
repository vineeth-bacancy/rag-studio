"""The page inspector's API and the passage outline for cleaned indexes."""
from __future__ import annotations

from rag_studio import parse, paths

IMAC = "apple_imac_per_2024_10"


def test_inspect_returns_both_views_of_a_page(client):
    r = client.get(f"/api/documents/{IMAC}/pages/2/inspect")
    assert r.status_code == 200
    d = r.json()
    assert d["page"] == 2 and d["page_count"] == 12 and d["label"] == "2"
    py = d["pymupdf"]
    assert py["text_raw"] and py["text"] and py["text"] != py["text_raw"]
    assert py["counts"].get("footnote", 0) >= 5                 # "iMac emissions10" and the chart legend
    assert all({"start", "end", "rule", "before"} <= set(m) for m in py["marks"])
    assert [r["id"] for r in d["rules"]][:3] == ["hidden", "nfkc", "th"]


def test_inspect_refuses_a_page_past_the_end(client):
    assert client.get(f"/api/documents/{IMAC}/pages/99/inspect").status_code == 404
    assert client.get("/api/documents/nope/pages/1/inspect").status_code == 404


def test_cleaning_summary_lists_pages_where_rules_acted(client):
    d = client.get("/api/documents/gpo_2018_annual_report/cleaning").json()
    pages = {p["page"] for p in d["pymupdf"]["pages"]}
    assert {29, 33} <= pages                                     # hidden footer; dot leaders
    assert d["pymupdf"]["counts"]["hidden_spans"] >= 37


def test_a_cleaned_passage_is_found_on_its_page_by_its_words():
    text = ("The global economy is facing another major shock. The conflict in the Middle East has "
            "triggered sharp increases in energy prices")
    r = parse.aligned_boxes(paths.PDFS / "wb_gep_2026_06.pdf", 19, text)
    assert r["aligned"] >= 0.9 and r["boxes"]                     # "The" was "Te" in the raw words
    assert all(0 <= b["x0"] < b["x1"] <= 1 and 0 <= b["y0"] < b["y1"] <= 1 for b in r["boxes"])
