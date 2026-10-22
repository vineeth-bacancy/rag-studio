"""Does a model really answer with thinking off? One real RAG prompt, streamed.

Usage: .venv/bin/python scripts/think_check.py <model> [think]
Prints only counts and timings, plus the first 300 chars of the answer.
"""
import json
import sys
import time

import httpx

from rag_studio import profiles
from rag_studio.prompt import build_messages
from rag_studio.retrieve import retriever

model = sys.argv[1]
think = (sys.argv[2] == "true") if len(sys.argv) > 2 else False
q = "What replaced FDsys, and when?"
r = retriever(profiles.DEFAULT)
hits, _ = r.search(q, k=5)
sources = [{"chunk_id": h["chunk_id"], "label": h.get("label"), "text": h["text"]} for h in hits]
messages = build_messages(q, sources, "v3")
body = {"model": model, "messages": messages, "stream": True, "think": think,
        "options": {"temperature": 0, "num_ctx": 8192}}
t0 = time.perf_counter()
content, thinking, first = [], [], None
final = {}
with httpx.stream("POST", "http://127.0.0.1:11434/api/chat", json=body, timeout=900) as resp:
    for line in resp.iter_lines():
        if not line.strip():
            continue
        d = json.loads(line)
        if d.get("error"):
            print("ERROR", d["error"]); sys.exit(1)
        m = d.get("message") or {}
        if m.get("thinking"):
            thinking.append(m["thinking"])
        if m.get("content"):
            if first is None:
                first = time.perf_counter() - t0
            content.append(m["content"])
        if d.get("done"):
            final = d
text = "".join(content)
print(json.dumps({
    "model": model, "think_sent": think, "wall_s": round(time.perf_counter() - t0, 1),
    "first_content_s": round(first, 1) if first else None,
    "thinking_chars": len("".join(thinking)), "content_chars": len(text),
    "think_tag_in_content": "<think>" in text or "</think>" in text,
    "prompt_tokens": final.get("prompt_eval_count"), "answer_tokens": final.get("eval_count"),
    "load_s": round((final.get("load_duration") or 0) / 1e9, 1),
    "prompt_s": round((final.get("prompt_eval_duration") or 0) / 1e9, 1),
    "answer_s": round((final.get("eval_duration") or 0) / 1e9, 1),
}, indent=1))
print("ANSWER:", text[:300])
