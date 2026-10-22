"""Conversations (M5): ids, atomic saves, history for the follow-up rewrite, and the API."""
from __future__ import annotations

import json

import pytest

from rag_studio import answer, chat, paths, rewrite

DONE = {"question": "What is the footprint?", "standalone_question": None, "rewritten": None,
        "answer": "It is 470 kg [1].", "refused": False,
        "citations": {"map": [{"n": 1, "chunk_id": "d:p1:c0", "doc_id": "d", "page": 1}], "numbers": [1], "valid": [1],
                      "invalid": []},
        "sources": [{"n": 1, "chunk_id": "d:p1:c0", "doc_id": "d", "page": 1, "text": "SECRET PASSAGE TEXT"}],
        "trace": [{"id": "embed", "ms": 1.0, "detail": {}}], "total_ms": 5.0}


@pytest.mark.parametrize("bad", ["", "abc", "../../etc/passwd", "a" * 31, "a" * 33, "G" * 32, "a" * 32 + "/x",
                                 "a" * 31 + ".", "..%2f" + "a" * 26, None, 7])
def test_only_32_hex_characters_name_a_conversation(tmp_data, bad):
    with pytest.raises(chat.BadChatId):
        chat.load(bad)
    with pytest.raises(chat.BadChatId):
        chat.delete(bad)
    assert not list(paths.CHAT.glob("*")) if paths.CHAT.exists() else True


def test_a_turn_keeps_what_the_thread_needs_and_not_the_passages(tmp_data):
    cid = chat.new_id()
    convo = chat.add_turn(cid, DONE | {"standalone_question": "Q2?", "rewritten": True}, {"profile": "p"})
    t = convo["turns"][0]
    assert t["question"] == "What is the footprint?" and t["standalone_question"] == "Q2?" and t["rewritten"] is True
    assert t["citations"] == DONE["citations"]["map"] and t["sources"] == [
        {"n": 1, "chunk_id": "d:p1:c0", "doc_id": "d", "page": 1}]
    assert t["setup"] == {"profile": "p"} and t["trace"] == [{"id": "embed", "ms": 1.0}]
    assert "SECRET PASSAGE TEXT" not in json.dumps(chat.load(cid))           # sources are references, not copies
    assert convo["title"] == "What is the footprint?" and convo["id"] == cid


def test_history_is_what_was_asked_and_answered_and_skips_unanswered_turns(tmp_data):
    cid = chat.new_id()
    chat.add_turn(cid, DONE, {})
    chat.add_turn(cid, DONE | {"question": "And in 2027?", "answer": "I could not find this in the documents.",
                               "refused": True}, {})
    assert chat.history(cid) == [{"question": "What is the footprint?", "answer": "It is 470 kg [1]."},
                                 {"question": "And in 2027?", "answer": "I could not find this in the documents."}]
    assert chat.history(chat.new_id()) == []


def test_writes_are_atomic_and_leave_no_temporary_files(tmp_data, monkeypatch):
    cid = chat.new_id()
    chat.add_turn(cid, DONE, {})
    first = (paths.CHAT / f"{cid}.json").read_text()
    # a failure while writing must leave the earlier file whole
    import rag_studio.files as files
    monkeypatch.setattr(files.os, "replace", lambda *a: (_ for _ in ()).throw(OSError("disk full")))
    with pytest.raises(OSError):
        chat.add_turn(cid, DONE | {"question": "Q2?"}, {})
    assert (paths.CHAT / f"{cid}.json").read_text() == first
    assert [p.name for p in paths.CHAT.iterdir()] == [f"{cid}.json"]


def test_list_is_newest_first_and_delete_removes(tmp_data):
    a, b = chat.new_id(), chat.new_id()
    chat.add_turn(a, DONE, {})
    chat.add_turn(b, DONE | {"question": "Second thread"}, {})
    chat.add_turn(a, DONE | {"question": "Back to the first"}, {})
    assert [c["id"] for c in chat.list_chats()] == [a, b] and chat.list_chats()[0]["turns"] == 2
    assert chat.delete(a) is True and chat.delete(a) is False and [c["id"] for c in chat.list_chats()] == [b]


def test_a_thread_has_a_limit(tmp_data, monkeypatch):
    monkeypatch.setattr(chat, "MAX_TURNS", 2)
    cid = chat.new_id()
    chat.add_turn(cid, DONE, {})
    chat.add_turn(cid, DONE, {})
    with pytest.raises(ValueError, match="Start a new one"):
        chat.add_turn(cid, DONE, {})


def test_the_module_logs_nothing(tmp_data, capsys, caplog):
    cid = chat.new_id()
    chat.add_turn(cid, DONE, {})
    chat.list_chats()
    chat.delete(cid)
    assert capsys.readouterr().out == "" and caplog.text == ""


# ------------------------------------------------------------------ the API

def fake_ask(seen):
    def ask(question, k=5, **kw):
        seen.append({"question": question, **kw})
        yield "ready", {"sources": []}
        yield "done", DONE | {"question": question}
    return ask


def frames(body):
    out = []
    for f in body.split("\n\n"):
        if f.strip():
            name, data = f.split("\n", 1)
            out.append((name.removeprefix("event: "), json.loads(data.removeprefix("data: "))))
    return out


def test_the_first_question_starts_a_conversation_and_the_next_continues_it(client, tmp_data, monkeypatch):
    seen = []
    monkeypatch.setattr(answer, "ask", fake_ask(seen))
    with client.stream("POST", "/api/ask", json={"question": "First?"}) as r:
        first = frames(r.read().decode())
    cid = first[-1][1]["chat_id"]
    assert chat.ID.match(cid) and first[0][1]["chat_id"] == cid and seen[0]["history"] is None
    with client.stream("POST", "/api/ask", json={"question": "And then?", "chat_id": cid}) as r:
        frames(r.read().decode())
    assert seen[1]["history"] == [{"question": "First?", "answer": "It is 470 kg [1]."}]
    assert len(client.get(f"/api/chats/{cid}").json()["turns"]) == 2
    assert client.get("/api/chats").json()["chats"][0]["id"] == cid


def test_as_typed_reaches_the_engine(client, tmp_data, monkeypatch):
    seen = []
    monkeypatch.setattr(answer, "ask", fake_ask(seen))
    cid = chat.new_id()
    chat.add_turn(cid, DONE, {})
    with client.stream("POST", "/api/ask", json={"question": "And 2027?", "chat_id": cid, "as_typed": True}) as r:
        r.read()
    assert seen[0]["as_typed"] is True and seen[0]["history"]


def test_a_bad_or_unknown_conversation_is_refused(client, tmp_data, monkeypatch):
    monkeypatch.setattr(answer, "ask", fake_ask([]))
    assert client.post("/api/ask", json={"question": "Q?", "chat_id": "../../x"}).status_code == 400
    assert client.post("/api/ask", json={"question": "Q?", "chat_id": "a" * 32}).status_code == 404
    assert client.get("/api/chats/" + "a" * 32).status_code == 404
    assert client.get("/api/chats/..%2F..%2Fsecret").status_code == 404
    assert client.delete("/api/chats/" + "b" * 32).status_code == 404


def test_deleting_a_conversation_removes_the_file(client, tmp_data):
    cid = chat.new_id()
    chat.add_turn(cid, DONE, {})
    assert client.delete(f"/api/chats/{cid}").json() == {"id": cid, "deleted": True}
    assert client.get(f"/api/chats/{cid}").status_code == 404


def test_the_switches_are_validated_before_anything_runs(client, tmp_data):
    assert client.post("/api/ask", json={"question": "Q?", "rewrite": "magic"}).status_code == 400
    assert client.post("/api/ask", json={"question": "Q?", "rerank": "gpt"}).status_code == 400
    assert client.post("/api/ask", json={"question": "Q?", "agent_rounds": 1}).status_code == 422


def test_an_answer_survives_a_failure_to_save_its_thread(client, tmp_data, monkeypatch):
    monkeypatch.setattr(answer, "ask", fake_ask([]))
    monkeypatch.setattr(chat, "add_turn", lambda *a, **k: (_ for _ in ()).throw(OSError("read-only")))
    with client.stream("POST", "/api/ask", json={"question": "Q?"}) as r:
        out = frames(r.read().decode())
    assert out[-1][0] == "done" and "could not be saved" in out[-1][1]["chat_warning"]


def test_no_conversation_content_reaches_the_log(client, tmp_data, monkeypatch, capsys, caplog):
    monkeypatch.setattr(answer, "ask", fake_ask([]))
    with client.stream("POST", "/api/ask", json={"question": "SECRET QUESTION?"}) as r:
        r.read()
    assert "SECRET" not in capsys.readouterr().out + caplog.text


def test_the_strict_remote_down_message_is_the_one_a_person_can_act_on():
    assert rewrite.FOLLOWUP_UNAVAILABLE == ("Follow-up questions are rewritten by Remote AI, which is not connected. "
                                            "Connect it in Settings, or ask the question in full.")
