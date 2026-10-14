import netguard  # noqa: F401  (must be first)
import asyncio
import json
import sys
import time

import ollama_chat as oc
from case import ANSWER, CONTEXTS, QUESTION, REFERENCE

from ragas.llms.base import InstructorBaseRagasLLM
from ragas.embeddings.base import BaseRagasEmbedding
from ragas.metrics.collections import (
    AnswerRelevancy,
    ContextPrecisionWithReference,
    ContextRecall,
    Faithfulness,
)


class GatewayJudge(InstructorBaseRagasLLM):
    """Custom judge: Ollama /api/chat with JSON-schema constrained output."""

    def generate(self, prompt, response_model):
        return oc.parse(response_model, oc.chat(prompt, response_model))

    async def agenerate(self, prompt, response_model):
        return oc.parse(response_model, await oc.achat(prompt, response_model))


class LocalEmb(BaseRagasEmbedding):
    def embed_text(self, text, **kw):
        return oc.embed([text])[0]

    async def aembed_text(self, text, **kw):
        return oc.embed([text])[0]

    def embed_texts(self, texts, **kw):
        return oc.embed(list(texts))

    async def aembed_texts(self, texts, **kw):
        return oc.embed(list(texts))


async def main(which):
    llm = GatewayJudge()
    out = {}
    for name in which:
        t0 = time.time()
        try:
            if name == "faith":
                m = Faithfulness(llm=llm)
                stmts = await m._create_statements(QUESTION, ANSWER)
                verdicts = await m._create_verdicts(stmts, "\n".join(CONTEXTS))
                r = await m.ascore(user_input=QUESTION, response=ANSWER, retrieved_contexts=CONTEXTS)
                rec = {"score": float(r.value), "reason": r.reason,
                       "debug_statements(private API)": stmts,
                       "debug_verdicts(private API)": verdicts.model_dump()}
            elif name == "relevancy":
                m = AnswerRelevancy(llm=llm, embeddings=LocalEmb())
                r = await m.ascore(user_input=QUESTION, response=ANSWER)
                rec = {"score": float(r.value), "reason": r.reason}
            elif name == "precision":
                m = ContextPrecisionWithReference(llm=llm)
                r = await m.ascore(user_input=QUESTION, reference=REFERENCE, retrieved_contexts=CONTEXTS)
                rec = {"score": float(r.value), "reason": r.reason}
            elif name == "recall":
                m = ContextRecall(llm=llm)
                r = await m.ascore(user_input=QUESTION, retrieved_contexts=CONTEXTS, reference=REFERENCE)
                rec = {"score": float(r.value), "reason": r.reason}
        except Exception as e:
            rec = {"error": repr(e)[:400]}
        rec["secs"] = round(time.time() - t0, 1)
        out[name] = rec
    print(json.dumps({"model": oc.MODEL, "stats": oc.STATS, "results": out}, indent=1, default=str))


asyncio.run(main(sys.argv[1:] or ["faith", "relevancy", "precision", "recall"]))
