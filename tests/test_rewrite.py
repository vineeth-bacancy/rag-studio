"""Query rewriting and follow-ups (M5): the prompts' outputs are checked, cached
and replayed, junk is an error, and the fused searches behave."""
from __future__ import annotations

import json

import pytest

from rag_studio import answer, ingest, paths, profiles, remote_cache, retrieve, rewrite
from rag_studio.llm import InstalledModel, OllamaUnavailable
from rag_studio.retrieve import SearchSpec
from tests.fakes import FakeOllama

IMAC = "apple_imac_per_2024_10"


class FakeRemote:
    """The Colab runtime's chat, scripted: `replies` maps a technique to the JSON it returns."""

    def __init__(self, replies=None, digest="sha256:8b"):
        self.replies = replies or {}
        self.digest = digest
        self.calls: list[dict] = []
        self.stats = type("S", (), {"snapshot": staticmethod(lambda: {"calls": 1, "bytes_out_raw": 10, "bytes_out_wire": 10,
                                                                         "bytes_in_raw": 10, "bytes_in_wire": 10,
                                                                         "wall_ms": 5.0, "server_ms": 3.0, "network_ms": 2.0,
                                                                         "connections": 0})})()

    def model_info(self, name):
        return InstalledModel(name=name, digest=self.digest, size=1, family=None, parameter_size="8B",
                              quantization="Q4_K_M", context_length=8192, capabilities=("completion",), modified_at=None)

    def chat(self, model, messages, *, options, think=False, format=None, keep_alive=None):
        self.calls.append({"model": model, "messages": messages, "options": options, "think": think, "format": format})
        system = messages[0]["content"]
        kind = next(k for k, p in rewrite.PROMPTS.items() if p["system"] == system)
        reply = self.replies[kind.split("-")[0]]
        text = reply(messages[1]["content"]) if callable(reply) else reply
        return {"message": {"content": text if isinstance(text, str) else json.dumps(text)},
                "prompt_eval_count": 300, "eval_count": 40}


GOOD = {
    "decompose": {"questions": ["What is the iMac's carbon footprint?", "How much recycled aluminum is in the stand?"]},
    "multi": {"queries": ["iMac lifecycle emissions", "greenhouse gases iMac total", "iMac kg CO2e per unit"]},
    "hyde": {"passage": "The iMac has a total carbon footprint of 470 kg CO2e over its lifetime, mostly from production."},
    "standalone": {"standalone": "How much of the iMac's footprint is transportation?"},
}


@pytest.fixture
def index(tmp_data, monkeypatch):
    monkeypatch.setattr(ingest, "pdf_files", lambda: [paths.PDFS / f"{IMAC}.pdf"])
    fake = FakeOllama()
    ingest.build(profiles.PYMUPDF_STRUCTURE, client=fake)
    return fake, retrieve.Retriever(profiles.PYMUPDF_STRUCTURE.id, client=fake)


def search(r, spec, remote, question="What is the carbon footprint of the iMac?", **kw):
    out = None
    steps = []
    for kind, payload in retrieve.run(question, retriever=r, k=5, spec=spec, deep=20, remote_client=remote, **kw):
        if kind == "stage":
            steps.append(payload)
        else:
            out = payload
    return out, steps


# ------------------------------------------------------------------ the calls

@pytest.mark.parametrize("kind", ["decompose", "multi", "hyde", "standalone"])
def test_each_rewrite_is_asked_with_its_schema_and_the_recorded_settings(tmp_data, kind):
    remote = FakeRemote(GOOD)
    inp = {"history": "Question: q\nAnswer: a", "question": "And in 2027?"} if kind == "standalone" else {"question": "Q?"}
    r = rewrite.call(kind, inp, client=remote)
    sent = remote.calls[0]
    assert sent["model"] == "qwen3:8b" and sent["think"] is False
    assert sent["options"] == {"temperature": 0.0, "num_ctx": 8192, "num_predict": 400}
    assert sent["format"] == rewrite.PROMPTS[rewrite.PROMPT_FOR[kind]]["schema"]
    assert "data, never instructions" in sent["messages"][0]["content"]
    assert r.cache == "miss" and r.seq and r.tokens == {"prompt": 300, "answer": 40} and r.network["calls"] == 0


def test_the_second_ask_is_a_cache_hit_and_never_reaches_the_runtime(tmp_data):
    remote = FakeRemote(GOOD)
    first = rewrite.plan("multi", "Q?", client=remote)
    second = rewrite.plan("multi", "Q?", client=remote)
    assert (first.cache, second.cache) == ("miss", "hit") and len(remote.calls) == 1
    assert second.value == first.value and second.seq == first.seq


def test_a_new_prompt_or_model_or_input_is_a_new_question(tmp_data, monkeypatch):
    remote = FakeRemote(GOOD)
    rewrite.plan("multi", "Q?", client=remote)
    rewrite.plan("multi", "Another?", client=remote)
    rewrite.plan("multi", "Q?", client=FakeRemote(GOOD, digest="sha256:other"))
    monkeypatch.setitem(rewrite.PROMPTS["multi-v1"], "system", rewrite.PROMPTS["multi-v1"]["system"] + " ")
    # (the fake finds the technique by its system text, so give it the same reply under the new text)
    class Same(FakeRemote):
        def chat(self, model, messages, **kw):
            self.calls.append(1)
            return {"message": {"content": json.dumps(GOOD["multi"])}, "prompt_eval_count": 300, "eval_count": 40}
    changed = Same()
    rewrite.plan("multi", "Q?", client=changed)
    assert len(remote.calls) == 2 and len(changed.calls) == 1


def test_cached_rewrites_replay_with_no_runtime_connected(tmp_data):
    rewrite.plan("hyde", "Q?", client=FakeRemote(GOOD))
    again = rewrite.plan("hyde", "Q?")                       # no client, nothing connected: the remembered digest finds it
    assert again.cache == "hit" and "carbon footprint" in again.value["passage"]


def test_a_rerank_only_runtime_still_replays_remembered_rewrites(tmp_data):
    """Connected to a runtime without qwen3:8b (only the reranker): cached rewrites replay, new ones refuse."""
    from rag_studio.llm import OllamaUnavailable
    rewrite.plan("multi", "Q?", client=FakeRemote(GOOD))

    class NoOllama(FakeRemote):
        def model_info(self, name):
            raise OllamaUnavailable("The tunnel answered /api/tags with an error page (HTTP 502).")

    again = rewrite.plan("multi", "Q?", client=NoOllama())
    assert again.cache == "hit"
    with pytest.raises(rewrite.RewriteUnavailable, match="not connected"):
        rewrite.plan("multi", "A question never asked?", client=NoOllama())


def test_no_runtime_and_nothing_cached_refuses_with_how_to_connect(tmp_data):
    with pytest.raises(rewrite.RewriteUnavailable, match="Remote AI, which is not connected. Connect it in Settings"):
        rewrite.plan("multi", "Q?")
    with pytest.raises(rewrite.RewriteUnavailable, match="Follow-up questions are rewritten by Remote AI, which is not "
                                                          "connected. Connect it in Settings, or ask the question in full."):
        rewrite.standalone("And in 2027?", [{"question": "q", "answer": "a"}])


def test_a_runtime_that_goes_away_mid_run_is_a_refusal_not_a_stack_trace(tmp_data):
    class Gone(FakeRemote):
        def chat(self, *a, **k):
            raise OllamaUnavailable("The Colab runtime is not answering; Colab may have reclaimed it.")
    with pytest.raises(rewrite.RewriteUnavailable, match="not answering.*Remote AI in Settings"):
        rewrite.plan("multi", "Q?", client=Gone())


@pytest.mark.parametrize("kind, bad", [
    ("decompose", "not json"), ("decompose", {"questions": []}), ("decompose", {"questions": ["a"] * 4}),
    ("decompose", {"questions": [""]}), ("decompose", {"questions": [5]}), ("multi", {"queries": ["a", "b"]}),
    ("multi", {"queries": ["x" * 301, "b", "c"]}), ("hyde", {"passage": "short"}), ("hyde", {"passage": 5}),
    ("hyde", {"other": "x" * 50}), ("standalone", {"standalone": ""}), ("standalone", ["x"]),
])
def test_junk_is_an_error_and_is_never_cached(tmp_data, kind, bad):
    remote = FakeRemote({kind: bad})
    inp = {"history": "h", "question": "q"} if kind == "standalone" else {"question": "q"}
    with pytest.raises(rewrite.RewriteBadOutput):
        rewrite.call(kind, inp, client=remote)
    assert not list(remote_cache.root().glob("*/*/*.json")) if remote_cache.root().exists() else True


def test_a_cut_prompt_is_an_error(tmp_data):
    class Cut(FakeRemote):
        def chat(self, *a, **k):
            return {"message": {"content": json.dumps(GOOD["multi"])}, "prompt_eval_count": 5, "eval_count": 3}
    with pytest.raises(rewrite.RewriteBadOutput, match="cut by the model's context"):
        rewrite.plan("multi", "q" * 600, client=Cut())


def test_a_follow_up_with_no_history_is_left_alone(tmp_data):
    remote = FakeRemote(GOOD)
    r = rewrite.standalone("What is the footprint?", [], client=remote)
    assert r.value == {"standalone": "What is the footprint?"} and r.cache == "none" and remote.calls == []


def test_history_is_shortened_and_limited_for_the_prompt():
    turns = [{"question": f"q{i}", "answer": "word " * 200} for i in range(9)]
    text = rewrite.format_history(turns)
    assert "q0" not in text and "q8" in text and text.count("Question:") == rewrite.MAX_HISTORY_TURNS
    assert max(len(t) for t in text.split("\n\n")) < 450 + 40


def test_a_recorded_description_is_stable_and_complete():
    d = rewrite.describe("hyde")
    assert d["model"] == "qwen3:8b" and d["where"] == "remote" and d["embed_as"] == "document" and d["num_ctx"] == 8192
    assert len(d["prompt_sha256"]) == 64 and "embed_as" not in rewrite.describe("multi")
    assert rewrite.describe("multi") == rewrite.describe("multi")


# ------------------------------------------------------------------ the searches

def test_a_plain_search_does_not_touch_the_runtime(index):
    _, r = index
    remote = FakeRemote(GOOD)
    out, steps = search(r, SearchSpec(), remote)
    assert remote.calls == [] and [s.id for s in steps] == ["embed", "search"] and out.rewrites is None


def test_multi_fuses_the_original_and_three_phrasings_with_rrf(index):
    _, r = index
    out, steps = search(r, SearchSpec(rewrite="multi"), FakeRemote(GOOD))
    assert [s.id for s in steps] == ["rewrite", "search", "search", "search", "search", "fuse"]
    assert set(out.lists) == {"original", "q1", "q2", "q3"}
    assert set(out.hits[0]["ranks"]) == {"original", "q1", "q2", "q3"}
    plain, _ = r.search("What is the carbon footprint of the iMac?", 20)
    assert {h["chunk_id"] for h in out.hits[:20]} >= {plain[0]["chunk_id"]}              # nothing the plain search found first is lost
    assert [h["rank"] for h in out.hits] == list(range(1, len(out.hits) + 1))
    assert out.rewrites[0]["technique"] == "multi" and out.cache == {"rewrite": "miss"}


def test_decompose_pools_each_parts_best_in_turn(index):
    _, r = index
    out, steps = search(r, SearchSpec(rewrite="decompose"), FakeRemote(GOOD))
    assert set(out.lists) == {"original", "q1", "q2"}
    firsts = [out.lists[name][0]["chunk_id"] for name in ("original", "q1", "q2")]
    assert [h["chunk_id"] for h in out.hits[:len(set(firsts))]] == list(dict.fromkeys(firsts))
    assert len({h["chunk_id"] for h in out.hits}) == len(out.hits)                          # each passage once
    assert next(s for s in steps if s.id == "fuse").detail["how"] == "round robin over the lists"


def test_hyde_embeds_the_drafted_passage_as_a_document(index, monkeypatch):
    _, r = index
    seen = []
    real = r.embedder.documents
    monkeypatch.setattr(r.embedder, "documents", lambda texts, *a, **k: (seen.append(texts), real(texts, *a, **k))[1])
    out, steps = search(r, SearchSpec(rewrite="hyde"), FakeRemote(GOOD))
    assert seen == [[GOOD["hyde"]["passage"]]] and set(out.lists) == {"original", "hyde"}
    assert next(s for s in steps if s.id == "rewrite").detail["passage"] == GOOD["hyde"]["passage"]
    assert any(s.detail.get("embed_as") == "document" for s in steps)


def test_the_original_question_is_searched_while_the_rewrite_is_in_flight(index):
    _, r = index
    order = []

    class Slow(FakeRemote):
        def chat(self, *a, **k):
            order.append("rewrite starts")
            import time
            time.sleep(0.3)
            order.append("rewrite done")
            return super().chat(*a, **k)

    real = r.search
    r.search = lambda q, k, spec=retrieve.DENSE: (order.append("search original") if len(order) < 3 else None, real(q, k, spec))[1]
    search(r, SearchSpec(rewrite="multi"), Slow(GOOD))
    assert order.index("search original") < order.index("rewrite done")


def test_the_reranker_reads_the_question_as_asked_and_the_section_limit_follows_fusion(index):
    _, r = index
    asked = []

    class Scorer:
        spec = retrieve.reranking.RERANKERS["qwen3-0.6b"]

        def score(self, q, docs):
            asked.append(q)
            return [-len(d) for d in docs]
    out, steps = search(r, SearchSpec(rewrite="multi", section_cap=1, rerank="qwen3-0.6b"), FakeRemote(GOOD),
                        scorer=Scorer())
    assert asked == ["What is the carbon footprint of the iMac?"]
    assert [s.id for s in steps][-2:] == ["fuse", "rerank"]


def test_a_missing_runtime_stops_a_rewritten_search_with_the_connect_message(index):
    _, r = index
    with pytest.raises(rewrite.RewriteUnavailable, match="not connected"):
        search(r, SearchSpec(rewrite="multi"), None)


# ------------------------------------------------------------------ follow-ups

HISTORY = [{"question": "What is the carbon footprint of the iMac?", "answer": "It is 470 kg CO2e [1]."}]


def test_a_follow_up_is_searched_as_the_standalone_question(index):
    _, r = index
    out, steps = search(r, SearchSpec(), FakeRemote(GOOD), question="How much of that is transportation?",
                        history=HISTORY)
    assert out.standalone_question == "How much of the iMac's footprint is transportation?"
    assert [s.id for s in steps] == ["followup", "embed", "search"]
    assert steps[0].detail["asked"] == "How much of that is transportation?" and steps[0].detail["changed"] is True
    plain, _ = r.search("How much of the iMac's footprint is transportation?", 20)
    assert [h["chunk_id"] for h in out.hits[:5]] == [h["chunk_id"] for h in plain[:5]]


def test_asked_as_typed_skips_the_rewrite(index):
    _, r = index
    remote = FakeRemote(GOOD)
    out, steps = search(r, SearchSpec(), remote, question="How much of that is transportation?",
                        history=HISTORY, as_typed=True)
    assert remote.calls == [] and out.standalone_question is None and [s.id for s in steps] == ["embed", "search"]


def test_the_answer_model_sees_the_standalone_question_never_the_conversation(index, monkeypatch):
    fake, r = index
    remote = FakeRemote(GOOD)
    # ask() builds its own retrieve.run call: give it the runtime through the module-level remote client
    import rag_studio.remote as remote_mod
    monkeypatch.setattr(remote_mod, "client", lambda: remote)
    events = list(answer.ask("How much of that is transportation?", retriever=r, client=fake, history=HISTORY,
                             search=SearchSpec(), prompt_version="v3", k=3, model="qwen3:1.7b"))
    done = events[-1][1]
    user = fake.chats[-1]["messages"][1]["content"]
    assert "How much of the iMac's footprint is transportation?" in user
    assert "470 kg" not in user and "How much of that is transportation" not in user
    assert done["question"] == "How much of that is transportation?" and done["rewritten"] is True
    assert done["searched_for"] == "How much of the iMac's footprint is transportation?"


def test_a_follow_up_with_no_runtime_is_an_error_event_that_offers_asking_as_typed(index):
    fake, r = index
    events = list(answer.ask("And for 2027?", retriever=r, client=fake, history=HISTORY, model="qwen3:1.7b"))
    name, data = events[-1]
    assert name == "error" and data["error"].startswith("Follow-up questions are rewritten by Remote AI")
    assert data["can_ask_as_typed"] is True
    again = list(answer.ask("And for 2027?", retriever=r, client=fake, history=HISTORY, model="qwen3:1.7b",
                            as_typed=True))
    assert again[-1][0] == "done" and again[-1][1]["rewritten"] is False
