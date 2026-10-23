"""How it works 11 and the figure routes (M7): shapes and input checks. The figure regions are the real
Docling ones (read from the committed PDFs' cached conversion); images and descriptions are stand-ins,
so nothing is drawn into or read from the real description cache."""
from __future__ import annotations

import pytest

from rag_studio import vision

C1 = "apple_imac_per_2024_10:p2:f0"
DESC = ("Title: none\nKind: stacked bar chart\nAxes: none\nSeries: Materials and process emissions (pink)\n"
        "Data: Baseline emissions | iMac emissions\nMaterials and process emissions (pink): ~190 kg | ~130 kg\n"
        "Takeaway: emissions fall.")
PNG = b"\x89PNG\r\n\x1a\nfake"


@pytest.fixture
def figures(monkeypatch):
    from rag_studio import ingest, parse_docling
    if not all(parse_docling.is_converted(p) for p in ingest.pdf_files()):
        pytest.skip("Docling has not converted the PDFs here (./rag ingest --profile docling-clean)")
    monkeypatch.setattr(vision, "image_for", lambda r: (PNG, f"sha-{r.figure_id}"))
    monkeypatch.setattr(vision, "lookup", lambda model, shas, version=vision.PROMPT_VERSION: (
        {s: {"description": DESC, "ms": 1000.0} for s in shas if s == f"sha-{C1}"}, "digest0"))


def test_the_figures_lab_shows_before_and_after(client, figures):
    r = client.get("/api/learn/figures")
    assert r.status_code == 200
    body = r.json()
    assert body["figure"]["figure_id"] == C1 and body["figure"]["image"] == f"/api/figures/{C1}/image"
    assert "Materials" in body["before"]["inside"]                       # the labels the text already had
    after = body["after"]
    assert after["description"] == DESC and after["kind"] == "stacked bar chart" and after["indexed"]
    assert after["passage"].startswith("Figure, described by ")
    assert body["question"]["id"] == "C1" and body["question"]["key"] == "materials and process emissions"
    reads = {(v["bar"], v["segment"]): v["reading"] for v in body["truth"]["check"]["values"]}
    assert reads[("Baseline (445 kg)", "Materials and process emissions")] == 190
    assert {e["figure_id"] for e in body["examples"]} >= {C1, "wb_gep_2026_06:p97:f3"}
    assert body["status"]["figures"] == len(vision.regions())


def test_a_figure_without_a_description_says_so(client, figures):
    body = client.get("/api/learn/figures", params={"figure": "wb_gep_2026_06:p97:f3"}).json()
    assert body["after"]["description"] is None and body["after"]["indexed"] is False
    assert body["truth"]["check"] is None                                # nothing to check: "-", not a guess


@pytest.mark.parametrize("figure,status", [("C1", 422), ("apple_imac_per_2024_10:p2:c0", 422),
                                           ("apple_imac_per_2024_10:p99:f0", 404)])
def test_the_figures_lab_checks_its_input(client, figures, figure, status):
    assert client.get("/api/learn/figures", params={"figure": figure}).status_code == status


def test_the_figure_image_route(client, figures):
    r = client.get(f"/api/figures/{C1}/image")
    assert r.status_code == 200 and r.headers["content-type"] == "image/png" and r.content == PNG
    assert client.get("/api/figures/apple_imac_per_2024_10:p99:f0/image").status_code == 404
    assert client.get("/api/figures/not-a-figure/image").status_code == 400


def test_a_figure_passage_is_outlined_by_its_picture(client, figures):
    r = client.get(f"/api/documents/apple_imac_per_2024_10/pages/2/boxes",
                   params={"start": 0, "end": 0, "profile": "docling-structure-embeddinggemma2-fig", "chunk": C1})
    assert r.status_code == 200
    body = r.json()
    assert body["figure"] is True and len(body["boxes"]) == 1
    b = body["boxes"][0]
    assert 0 < b["x0"] < b["x1"] <= 1 and 0 < b["y0"] < b["y1"] <= 1
    # A text passage still needs a real span.
    bad = client.get(f"/api/documents/apple_imac_per_2024_10/pages/2/boxes",
                     params={"start": 0, "end": 0, "profile": "docling-structure-embeddinggemma2"})
    assert bad.status_code == 400
