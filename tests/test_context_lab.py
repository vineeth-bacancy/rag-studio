"""How it works step 4 (M5): a passage, its context line, the exact embedded
text with and without it, and the answer's rank on the plain and the -ctx
index. Reads the context-line cache only; no model writes a line."""
from __future__ import annotations

import os
import subprocess
from dataclasses import replace

import pytest

from rag_studio import context, ingest, learn, paths, profiles, retrieve
from rag_studio.files import write_json_atomic
from tests.fakes import FakeOllama
from tests.test_context import FakeRuntime

IMAC = "apple_imac_per_2024_10"
# Named like the real row-7 profiles, so its plain index is pymupdf-structure.
CTX = replace(profiles.PYMUPDF_STRUCTURE, id="pymupdf-structure-ctx", label="PyMuPDF structure, with context lines",
              contextual=True)


@pytest.fixture
def imac_ctx(tmp_data, monkeypatch):
    monkeypatch.setattr(ingest, "pdf_files", lambda: [paths.PDFS / f"{IMAC}.pdf"])
    monkeypatch.setitem(profiles.PROFILES, CTX.id, CTX)
    monkeypatch.setattr(learn, "_cuts", {})

    def no_search(profile_id):
        raise AssertionError("nothing here should open an index")

    monkeypatch.setattr(learn, "retriever", no_search)
    return tmp_data


def test_it_shows_the_line_and_exactly_what_is_embedded_with_and_without_it(imac_ctx):
    rt = FakeRuntime()
    context.write_lines(CTX, client=rt, batch_size=8)
    calls = len(rt.calls)
    out = learn.context_lab(CTX.id)
    assert len(rt.calls) == calls                                    # looking writes nothing
    p = out["passage"]
    prefix = CTX.embedding.doc_prefix
    assert p["context"].startswith("From the iMac report")
    assert p["embedded_after"] == f"{prefix}{p['label']}\n{p['context']}\n{p['text']}"
    assert p["embedded_before"] == f"{prefix}{p['label']}\n{p['text']}" and p["context"] not in p["embedded_before"]
    assert p["window"]["passages"] >= 1 and 1 <= p["window"]["position"] <= p["window"]["passages"]
    assert p["window"]["head"].startswith("Document: ")
    cov = out["coverage"]
    assert cov["written"] == cov["passages"] > 0 and cov["built"] is False and cov["digest"] == "sha256-fake8b"   # the cache folder's name
    assert 0 < len(out["samples"]) <= learn.CONTEXT_SAMPLES and all(s["context"] for s in out["samples"])
    assert out["describe"]["prompt_version"] == "context-v1" and out["ranking"] is None
    assert out["plain_profile"] == "pymupdf-structure"
    assert any(x["id"] == CTX.id for x in out["profiles"])


def test_a_passage_without_a_line_is_shown_plainly(imac_ctx):
    out = learn.context_lab(CTX.id, chunk_id=f"{IMAC}:p1:c0")
    p = out["passage"]
    assert p["chunk_id"] == f"{IMAC}:p1:c0" and p["context"] is None and p["embedded_after"] is None
    assert out["coverage"]["written"] == 0 and out["samples"] == []


def test_a_test_question_is_ranked_on_the_plain_and_the_ctx_index(imac_ctx, monkeypatch):
    context.write_lines(CTX, client=FakeRuntime(), batch_size=8)
    fake = FakeOllama()
    ingest.build(profiles.PYMUPDF_STRUCTURE, client=fake)
    ingest.build(CTX, client=fake)
    opened = {pid: retrieve.Retriever(pid, client=fake) for pid in ("pymupdf-structure", CTX.id)}
    monkeypatch.setattr(learn, "retriever", lambda pid: opened[pid])
    out = learn.context_lab(CTX.id, question_id="B5")              # iMac recycled content
    r = out["ranking"]
    assert r["id"] == "B5" and r["targets"][0]["doc_id"] == IMAC
    assert r["plain"]["profile"] == "pymupdf-structure" and r["ctx"]["profile"] == CTX.id
    for side in ("plain", "ctx"):
        assert len(r[side]["top"]) == 5
        rank = r[side]["first_answer_rank"]
        assert rank is None or 1 <= rank <= 50
    p = out["passage"]
    assert p["page"] in r["targets"][0]["pages"]                    # the question's answer passage is shown
    assert set(p["ranks"]) == {"plain", "ctx"}
    assert out["coverage"]["built"] is True and out["coverage"]["plain_built"] is True


def test_an_unbuilt_index_has_no_rank_and_opens_nothing(imac_ctx):
    out = learn.context_lab(CTX.id, question_id="B5")
    assert out["ranking"]["plain"] is None and out["ranking"]["ctx"] is None
    assert out["passage"]["page"] in out["ranking"]["targets"][0]["pages"]


@pytest.mark.parametrize("kwargs, error", [
    ({"profile_id": "pymupdf-structure"}, ValueError),            # no context lines
    ({"question_id": "Z9"}, LookupError),                          # no such question
    ({"question_id": "E1"}, LookupError),                          # a follow-up: not a question by itself
    ({"chunk_id": f"{IMAC}:p999:c0"}, LookupError),                 # no such passage
])
def test_it_refuses_what_it_cannot_show(imac_ctx, kwargs, error):
    with pytest.raises(error):
        learn.context_lab(**({"profile_id": CTX.id} | kwargs))


def test_docling_is_never_run_from_here(tmp_data, monkeypatch):
    from rag_studio import parse_docling

    monkeypatch.setattr(ingest, "pdf_files", lambda: [paths.PDFS / f"{IMAC}.pdf"])
    monkeypatch.setattr(learn, "_cuts", {})
    monkeypatch.setattr(parse_docling, "convert", lambda *a, **k: pytest.fail("Docling must not run"))
    with pytest.raises(LookupError, match="Docling has not converted"):
        learn.context_lab("docling-structure-embeddinggemma2-ctx")


def test_a_job_whose_process_is_gone_reads_interrupted(imac_ctx):
    dead = subprocess.Popen(["true"])
    dead.wait()
    write_json_atomic(context.job_path(CTX.id), {"status": "running", "pid": dead.pid, "done": 3, "total": 9})
    assert learn._job_brief(CTX.id)["status"] == "interrupted"
    write_json_atomic(context.job_path(CTX.id), {"status": "running", "pid": os.getpid(), "done": 3, "total": 9})
    assert learn._job_brief(CTX.id)["status"] == "running"


def test_the_route_answers_and_validates(client, imac_ctx):
    context.write_lines(CTX, client=FakeRuntime(), batch_size=8)
    r = client.get("/api/learn/context", params={"profile": CTX.id})
    assert r.status_code == 200, r.text
    assert r.json()["passage"]["context"]
    assert client.get("/api/learn/context", params={"profile": CTX.id, "question": "b4"}).status_code == 422
    assert client.get("/api/learn/context", params={"profile": CTX.id, "chunk": "../etc/passwd"}).status_code == 422
    assert client.get("/api/learn/context", params={"profile": "pymupdf-structure"}).status_code == 400
    assert client.get("/api/learn/context", params={"profile": "nope"}).status_code == 400
    assert client.get("/api/learn/context", params={"profile": CTX.id, "question": "Z9"}).status_code == 404
