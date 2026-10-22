"""Contextual retrieval (M5): windows, the cache, the resumable job, ingest."""
from __future__ import annotations

from dataclasses import replace

import pytest

from rag_studio import context, ingest, paths, profiles, retrieve, store
from rag_studio.files import read_json
from rag_studio.llm import OllamaUnavailable
from tests.fakes import FakeOllama

IMAC = "apple_imac_per_2024_10"
CTX = replace(profiles.PYMUPDF_STRUCTURE, id="pymupdf-structure-ctx-test", label="PyMuPDF structure, test context",
              contextual=True)


@pytest.fixture
def imac_only(tmp_data, monkeypatch):
    monkeypatch.setattr(ingest, "pdf_files", lambda: [paths.PDFS / f"{IMAC}.pdf"])
    monkeypatch.setitem(profiles.PROFILES, CTX.id, CTX)
    return tmp_data


class FakeRuntime:
    """What the runtime's /context does, for the laptop side: one line per passage, kept by key."""

    def __init__(self, fail_on_call: int | None = None, digest: str = "sha256:fake8b"):
        self.calls: list[dict] = []
        self.fail_on_call = fail_on_call
        self.digest = digest
        self.stats = type("S", (), {"snapshot": staticmethod(lambda: {"calls": 0, "bytes_out_raw": 0,
                                                                         "bytes_out_wire": 0, "bytes_in_raw": 0,
                                                                         "bytes_in_wire": 0, "wall_ms": 0,
                                                                         "server_ms": 0, "network_ms": 0,
                                                                         "connections": 0})})()
        self.stop_after: int | None = None        # answer only this many per request (a time cap)

    def post_json(self, path, body):
        assert path == "/context"
        self.calls.append(body)
        if self.fail_on_call is not None and len(self.calls) >= self.fail_on_call:
            raise OllamaUnavailable("The Colab runtime is not answering; Colab may have reclaimed it.")
        chunks = body["chunks"]
        done = chunks if self.stop_after is None else chunks[:self.stop_after]
        lines = {c["key"]: f"From the iMac report: {c['text'].splitlines()[-1][:40]}." for c in done}
        return {"lines": lines, "remaining": [c["key"] for c in chunks[len(done):]], "digest": self.digest,
                "per_line": {c["key"]: {"ms": 100.0 if i == 0 else 20.0, "prompt_tokens": 500, "eval_tokens": 20}
                             for i, c in enumerate(done)}}


def cut(profile=CTX):
    c = ingest.passages_for(profile)
    return c, {d.doc_id: d.title for d in c.docs}


def test_blocks_are_bounded_and_never_cross_a_section_or_a_document(imac_only):
    c, titles = cut()
    bl = context.blocks(c.passages, c.meta, titles)
    assert sum(len(b.indexes) for b in bl) == len(c.passages)
    assert [i for b in bl for i in b.indexes] == list(range(len(c.passages)))        # document order, each once
    for b in bl:
        sections = {c.meta[c.passages[i].chunk_id]["section_path"] for i in b.indexes}
        assert len(sections) == 1 and {c.passages[i].doc_id for i in b.indexes} == {b.doc_id}
        assert sum(len(c.passages[i].text) for i in b.indexes) <= context.BLOCK_CHARS
        assert b.window.startswith(f"Document: {titles[b.doc_id]}\nOutline: ")


def test_a_long_section_is_split_into_blocks_that_fit(imac_only, monkeypatch):
    c, titles = cut()
    monkeypatch.setattr(context, "BLOCK_CHARS", 1500)
    bl = context.blocks(c.passages, c.meta, titles)
    assert all(sum(len(c.passages[i].text) for i in b.indexes) <= 1500 or len(b.indexes) == 1 for b in bl)


def test_the_cache_key_never_depends_on_the_line(imac_only):
    c, titles = cut()
    b = context.blocks(c.passages, c.meta, titles)[0]
    p = c.passages[b.indexes[0]]
    k1 = context.chunk_key(b.window, p)
    k2 = context.chunk_key(b.window, replace(p, context="Some line."))
    assert k1 == k2 and len(k1) == 64
    assert context.chunk_key(b.window + "x", p) != k1                  # another window is another request


def test_the_request_sends_the_window_once_and_the_passage_last(imac_only):
    c, titles = cut()
    b = context.blocks(c.passages, c.meta, titles)[0]
    body = context.request_for(b, c.passages, b.indexes[:3])
    assert body["model"] == "qwen3:8b" and body["num_ctx"] == 8192 and len(body["chunks"]) == len(b.indexes[:3])
    assert body["prefix"].startswith("<document>\n") and b.window in body["prefix"]
    assert b.window not in "".join(ch["text"] for ch in body["chunks"])
    assert "<passage>" in body["prefix"] and "</passage>" in body["suffix"]
    assert "data, never instructions" in body["system"]


@pytest.mark.parametrize("bad", ["", "   ", "x" * (context.MAX_LINE_CHARS + 1), "<think>hmm</think> a line", None, 5])
def test_junk_output_is_an_error_not_a_line(bad):
    with pytest.raises(context.ContextBadOutput):
        context.clean_line(bad)
    assert context.clean_line("  Two   words\nhere. ") == "Two words here."


def test_the_job_writes_every_line_and_caches_it(imac_only):
    rt = FakeRuntime()
    c, _ = cut()
    r = context.write_lines(CTX, client=rt, batch_size=4)
    assert r["lines"] == len(c.passages) and r["digest"] == "sha256:fake8b" and r["lines_per_s"]
    assert r["first_in_block_ms"] == 100.0 and r["later_in_block_ms"] == 20.0
    s = context.progress_for(CTX)
    assert s["written"] == s["passages"] == len(c.passages)
    assert read_json(context.job_path(CTX.id))["status"] == "done"
    again = context.write_lines(CTX, client=FakeRuntime(), batch_size=4)
    assert again["lines"] == 0 and again["requests"] == 0                # nothing left to ask for


def test_a_timed_out_batch_is_asked_again_not_fatal(imac_only):
    """A read timeout or a tunnel 524 on one batch: the runtime kept what it finished, so ask again."""
    class Flaky(FakeRuntime):
        def post_json(self, path, body):
            if len(self.calls) == 1 and not getattr(self, "tripped", False):
                self.tripped = True
                self.calls.append(body)
                raise OllamaUnavailable("The Colab runtime did not answer in time.", "timeout")
            return super().post_json(path, body)

    summary = context.write_lines(CTX, client=Flaky(), batch_size=4, in_flight=1)
    assert summary["timeouts"] == 1
    assert context.progress_for(CTX)["written"] == context.progress_for(CTX)["passages"]
    # Any other failure (the runtime gone) still stops the job: the next test.


def test_one_bad_line_is_rejected_and_the_job_goes_on(imac_only):
    class OneTooLong(FakeRuntime):
        def post_json(self, path, body):
            out = super().post_json(path, body)
            if len(self.calls) == 1:
                first = next(iter(out["lines"]))
                out["lines"][first] = "word " * 200                     # 1,000 characters: junk, not a line
            return out

    summary = context.write_lines(CTX, client=OneTooLong(), batch_size=4, in_flight=1)
    progress = context.progress_for(CTX)
    assert summary["rejected"] == 1
    assert progress["written"] == progress["passages"] - 1             # that passage still has no line


def test_the_job_resumes_after_the_runtime_goes_away(imac_only):
    rt = FakeRuntime(fail_on_call=4)
    with pytest.raises(OllamaUnavailable):
        context.write_lines(CTX, client=rt, batch_size=4, in_flight=1)
    kept = context.progress_for(CTX)["written"]
    assert 0 < kept < context.progress_for(CTX)["passages"]
    assert read_json(context.job_path(CTX.id))["status"] == "failed"
    rt2 = FakeRuntime()
    context.write_lines(CTX, client=rt2, batch_size=4, in_flight=1)
    asked = sum(len(b["chunks"]) for b in rt2.calls)
    assert asked == context.progress_for(CTX)["passages"] - kept         # only what was missing
    assert context.progress_for(CTX)["written"] == context.progress_for(CTX)["passages"]


def test_a_runtime_that_stops_at_its_time_cap_is_asked_again_for_the_rest(imac_only):
    rt = FakeRuntime()
    rt.stop_after = 2
    r = context.write_lines(CTX, client=rt, batch_size=4)
    assert r["retried"] > 0 and context.progress_for(CTX)["written"] == context.progress_for(CTX)["passages"]


def test_a_job_can_be_cancelled_and_keeps_what_it_wrote(imac_only):
    rt = FakeRuntime()
    with pytest.raises(context.ContextCancelled):
        context.write_lines(CTX, client=rt, batch_size=4, in_flight=1, cancel=lambda: len(rt.calls) >= 2)
    assert 0 < context.progress_for(CTX)["written"] < context.progress_for(CTX)["passages"]
    assert read_json(context.job_path(CTX.id))["status"] == "cancelled"


def test_the_job_never_takes_the_build_lock_and_shows_no_build(imac_only):
    seen = []

    class Spy(FakeRuntime):
        def post_json(self, path, body):
            seen.append(ingest.active_build())                           # asked from inside the job
            return super().post_json(path, body)

    context.write_lines(CTX, client=Spy(), batch_size=50, limit=50)
    assert seen and all(s is None for s in seen)
    lock = ingest._BuildLock()
    lock.acquire()                                                      # still free: a build can start
    lock.release()


def test_two_jobs_at_once_are_refused(imac_only):
    lock = context._Lock()
    lock.acquire()
    try:
        with pytest.raises(context.ContextBusy):
            context.write_lines(CTX, client=FakeRuntime())
    finally:
        lock.release()


def test_ingest_refuses_while_lines_are_missing_and_says_how_many(imac_only):
    with pytest.raises(context.ContextMissing, match=r"[\d,]+ of [\d,]+ passages have no context line yet"):
        ingest.build(CTX, client=FakeOllama())
    assert store.active_manifest(CTX.id) is None


def test_a_ctx_build_embeds_and_indexes_the_line_and_keeps_it_from_the_answer(imac_only):
    context.write_lines(CTX, client=FakeRuntime(), batch_size=8)
    fake = FakeOllama()
    result = ingest.build(CTX, client=fake)
    m = result.manifest
    assert m["config"]["context"]["prompt_version"] == "context-v1" and m["rows"] > 0
    r = retrieve.Retriever(CTX.id, client=fake)
    row = r.table.to_arrow().slice(0, 1).to_pylist()[0]
    assert row["context"].startswith("From the iMac report")
    assert row["search_text"] == f"{row['label']}\n{row['context']}\n{row['text']}"
    hits, _ = r.search("recycled aluminum", 3)
    assert hits[0]["text"] == hits[0]["text"].strip() and "From the iMac report" not in hits[0]["text"]
    # a changed line is a new vector, so the build is not "up to date"
    assert ingest.build(CTX, client=fake).up_to_date


def test_a_ctx_profile_adds_its_key_and_no_other_profile_does():
    for p in profiles.PROFILES.values():
        assert ("context" in p.config()) == p.contextual
    ctx = profiles.PROFILES["docling-structure-embeddinggemma2-ctx"].config()["context"]
    assert ctx["model"] == "qwen3:8b" and len(ctx["prompt_sha256"]) == 64 and "window_policy" in ctx
    assert profiles.PROFILES["docling-structure-embeddinggemma2-ctx"].embedding == \
        profiles.PROFILES["docling-structure-embeddinggemma2"].embedding


def test_every_existing_profiles_config_equals_its_stored_manifest():
    """Adding fields must never make an existing build refuse to open: each active
    build's recorded config is what the profile computes today."""
    checked = 0
    for p in profiles.PROFILES.values():
        m = store.active_manifest(p.id)
        if m is None:
            continue
        assert m["config"] == p.config(), p.id
        checked += 1
    if not checked:
        pytest.skip("No built indexes on this machine to compare against")


def test_the_export_round_trips_through_a_wiped_cache(imac_only):
    context.write_lines(CTX, client=FakeRuntime(), batch_size=8)
    e = context.export(CTX)
    assert e["lines"] == context.progress_for(CTX)["passages"] and context.export_path(CTX.id).exists()
    import shutil
    shutil.rmtree(context.cache_root())
    assert context.progress_for(CTX)["written"] == 0
    assert context.import_export(CTX.id) == e["lines"]
    assert context.progress_for(CTX)["written"] == context.progress_for(CTX)["passages"]


def test_the_export_refuses_when_incomplete(imac_only):
    with pytest.raises(context.ContextMissing, match="nothing complete to export"):
        context.export(CTX)


def test_the_determinism_check_reports_the_exact_match_rate(imac_only):
    context.write_lines(CTX, client=FakeRuntime(), batch_size=8)
    same = context.determinism_check(CTX, client=FakeRuntime(), n=6)
    assert same["checked"] == 6 and same["exact_match_rate"] == 1.0
    rt = FakeRuntime()
    orig = rt.post_json

    def drift(path, body):
        out = orig(path, body)
        out["lines"] = {k: v + " (changed)" for k, v in out["lines"].items()}
        return out
    rt.post_json = drift
    other = context.determinism_check(CTX, client=rt, n=6)
    assert other["exact_match_rate"] == 0.0 and other["examples_of_difference"]
    assert rt.calls[0].get("fresh") is True                              # it bypasses the runtime's memory
