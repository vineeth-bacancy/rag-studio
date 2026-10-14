import netguard  # noqa: F401  (must be first)
import asyncio
import json
import os
import time

from case import ANSWER, CONTEXTS, QUESTION
from openai import AsyncOpenAI
from ragas.llms import llm_factory
from ragas.metrics.collections import Faithfulness

MODEL = os.environ.get("JUDGE_MODEL", "qwen3:1.7b")


async def main():
    client = AsyncOpenAI(base_url="http://127.0.0.1:11434/v1", api_key="ollama")
    llm = llm_factory(MODEL, provider="openai", client=client, temperature=0, max_tokens=4096)
    t0 = time.time()
    try:
        r = await Faithfulness(llm=llm).ascore(user_input=QUESTION, response=ANSWER, retrieved_contexts=CONTEXTS)
        rec = {"score": float(r.value)}
    except Exception as e:
        rec = {"error": repr(e)[:400]}
    rec["secs"] = round(time.time() - t0, 1)
    print(json.dumps({"model": MODEL, "RAGAS_DO_NOT_TRACK": os.environ.get("RAGAS_DO_NOT_TRACK"), **rec}))


asyncio.run(main())
