import netguard  # noqa: F401  (must be first)
import json
import os
import sys
import time

import ollama_chat as oc
from case import ANSWER, CONTEXTS, QUESTION, REFERENCE

from deepeval.models import DeepEvalBaseLLM
from deepeval.test_case import LLMTestCase
from deepeval.metrics import (
    AnswerRelevancyMetric,
    ContextualPrecisionMetric,
    ContextualRecallMetric,
    FaithfulnessMetric,
)


class GatewayJudge(DeepEvalBaseLLM):
    """Custom judge: Ollama /api/chat (optionally behind a bearer-token gateway)."""

    def __init__(self):
        super().__init__(oc.MODEL)

    def load_model(self):
        return self

    def generate(self, prompt: str, schema=None):
        out = oc.chat(prompt, schema)
        return oc.parse(schema, out) if schema else out

    async def a_generate(self, prompt: str, schema=None):
        out = await oc.achat(prompt, schema)
        return oc.parse(schema, out) if schema else out

    def get_model_name(self):
        return f"{oc.MODEL} via /api/chat"


judge = GatewayJudge()
tc = LLMTestCase(input=QUESTION, actual_output=ANSWER, retrieval_context=CONTEXTS, expected_output=REFERENCE)
which = sys.argv[1:] or ["faith", "relevancy", "precision", "recall"]
out = {}
for name in which:
    cls = {"faith": FaithfulnessMetric, "relevancy": AnswerRelevancyMetric,
           "precision": ContextualPrecisionMetric, "recall": ContextualRecallMetric}[name]
    m = cls(model=judge, eval_mode="llm", async_mode=False, include_reason=True)
    t0 = time.time()
    try:
        m.measure(tc, _show_indicator=False)
        rec = {"score": m.score, "secs": round(time.time() - t0, 1), "reason": m.reason}
        if name == "faith":
            rec["claims"] = m.claims
            rec["verdicts"] = [v.model_dump() for v in m.verdicts]
        elif name == "relevancy":
            rec["statements"] = m.statements
            rec["verdicts"] = [v.model_dump() for v in m.verdicts]
        else:
            rec["verdicts"] = [v.model_dump() for v in m.verdicts]
    except Exception as e:
        rec = {"error": repr(e)[:300], "secs": round(time.time() - t0, 1)}
    out[name] = rec
print(json.dumps({"model": oc.MODEL, "stats": oc.STATS, "results": out}, indent=1, default=str))
