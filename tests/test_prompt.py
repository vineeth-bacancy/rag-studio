"""The prompt is versioned to the byte: whitespace changed a small model's answers."""
from __future__ import annotations

import hashlib

from rag_studio import prompt


def test_v2_is_the_tested_text():
    text = prompt.system_prompt("v2")
    assert text.startswith("You answer questions about a set of PDF reports")
    assert "\n\nRules:\n1. Use ONLY" in text          # exactly one blank line
    assert "\n\n\n" not in text and not text.endswith("\n")
    assert prompt.REFUSAL + "\n3. After each fact" in text
    # Pinned: any edit, even whitespace, is a new version with a new hash.
    assert hashlib.sha256(text.encode()).hexdigest() == V2_SHA256
    assert prompt.prompt_sha256("v2") == V2_SHA256


# sha256 of the spec §6 text as typed into prompt.py on 2026-10-15.
V2_SHA256 = "d30ea3a10dfbdebb6fdcb8bde0fea60f4e97afd3a0d1c83c19294f8d4b997c56"


def test_messages_use_the_baseline_layout():
    msgs = prompt.build_messages("Q?", [{"chunk_id": "d:p5:c0", "text": "alpha"},
                                         {"chunk_id": "d:p6:c1", "text": "beta"}], version="v2")
    assert msgs[0]["role"] == "system" and msgs[1]["role"] == "user"
    assert msgs[1]["content"] == "Context:\n[1] d:p5:c0\nalpha\n\n[2] d:p6:c1\nbeta\n\nQuestion: Q?"


def test_truncation_is_detected_from_the_token_count():
    msgs = [{"role": "user", "content": "x" * 7000}]
    assert prompt.looks_truncated(msgs, 1750) is False   # 4 chars per token: read in full
    assert prompt.looks_truncated(msgs, 258) is True     # the measured overflow case
    assert prompt.looks_truncated(msgs, None) is None


HANDBUILT = __import__("pathlib").Path.home() / "Dev/RAG/advanced-rag/answer.py"


def test_handbuilt_variant_is_byte_identical_to_answer_py():
    import ast

    import pytest

    if not HANDBUILT.is_file():
        pytest.skip("advanced-rag not on this machine")
    tree = ast.parse(HANDBUILT.read_text())
    text = next(n.value.value for n in tree.body
                if isinstance(n, ast.Assign) and getattr(n.targets[0], "id", "") == "SYSTEM")
    assert prompt.system_prompt("v2-handbuilt") == text
    assert prompt.prompt_sha256("v2-handbuilt") == HANDBUILT_SHA256


HANDBUILT_SHA256 = "04280fede52408e1a6cb0c54c32a041c717370f1d75892799d5eeda31e40f6f2"


def test_v4_changes_only_rule_4_of_v3():
    v3, v4 = prompt.system_prompt("v3").splitlines(), prompt.system_prompt("v4").splitlines()
    changed = [(a, b) for a, b in zip(v3, v4) if a != b]
    assert len(v3) == len(v4) and len(changed) == 1
    assert changed[0][0].startswith("4. Never say which section")
    assert changed[0][1] == prompt.RULE_4_V4


def test_a_passage_shows_its_section_label_only_when_it_has_one():
    msgs = prompt.build_messages("Q?", [
        {"chunk_id": "d:p19:c2", "text": "Risks...", "label": "Document: GEP\nSection: Executive Summary"},
        {"chunk_id": "d:p6:c1", "text": "beta", "label": None}], "v4")
    user = msgs[1]["content"]
    assert "[1] d:p19:c2\nDocument: GEP\nSection: Executive Summary\nRisks..." in user
    assert "[2] d:p6:c1\nbeta" in user
    assert user.startswith(f"Context:\n{prompt.FENCE_OPEN}\n")
