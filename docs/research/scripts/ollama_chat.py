"""Minimal Ollama /api/chat client, same shape as the Colab gateway (bearer token optional)."""
import os
import httpx

BASE = os.environ.get("JUDGE_BASE", "http://127.0.0.1:11434")
MODEL = os.environ.get("JUDGE_MODEL", "qwen3:1.7b")
TOKEN = os.environ.get("JUDGE_TOKEN")  # gateway bearer token, unused locally
SEED = int(os.environ.get("JUDGE_SEED", "42"))
HEADERS = {"Authorization": f"Bearer {TOKEN}"} if TOKEN else {}
STATS = {"calls": 0, "json_fail": 0}


def _payload(prompt, schema):
    p = {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
        "think": False,
        "options": {"temperature": 0, "seed": SEED, "num_ctx": 8192},
    }
    if schema is not None:
        p["format"] = schema.model_json_schema()
    return p


def chat(prompt, schema=None):
    STATS["calls"] += 1
    r = httpx.post(f"{BASE}/api/chat", json=_payload(prompt, schema), headers=HEADERS, timeout=600)
    r.raise_for_status()
    return r.json()["message"]["content"]


async def achat(prompt, schema=None):
    STATS["calls"] += 1
    async with httpx.AsyncClient(timeout=600, headers=HEADERS) as c:
        r = await c.post(f"{BASE}/api/chat", json=_payload(prompt, schema))
        r.raise_for_status()
        return r.json()["message"]["content"]


def embed(texts):
    r = httpx.post(f"{BASE}/api/embed", json={"model": "all-minilm", "input": texts}, headers=HEADERS, timeout=120)
    r.raise_for_status()
    return r.json()["embeddings"]


def parse(schema, content):
    try:
        return schema.model_validate_json(content)
    except Exception:
        STATS["json_fail"] += 1
        raise
