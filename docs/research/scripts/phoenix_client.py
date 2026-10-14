import netguard  # noqa: F401
import json
import time
import urllib.request

from openinference.semconv.trace import (
    DocumentAttributes,
    OpenInferenceSpanKindValues,
    SpanAttributes,
)
from phoenix.otel import register

tp = register(project_name="rag-studio-test", endpoint="http://127.0.0.1:6006/v1/traces",
              batch=False, set_global_tracer_provider=False, verbose=False)
tracer = tp.get_tracer("rag-studio")

with tracer.start_as_current_span("rag.query") as root:
    root.set_attribute(SpanAttributes.OPENINFERENCE_SPAN_KIND, OpenInferenceSpanKindValues.CHAIN.value)
    root.set_attribute(SpanAttributes.INPUT_VALUE, "When was the Eiffel Tower completed?")
    with tracer.start_as_current_span("retrieve.hybrid") as s:
        s.set_attribute(SpanAttributes.OPENINFERENCE_SPAN_KIND, OpenInferenceSpanKindValues.RETRIEVER.value)
        for i, (doc_id, score) in enumerate([("c1", 0.82), ("c7", 0.61)]):
            p = f"{SpanAttributes.RETRIEVAL_DOCUMENTS}.{i}."
            s.set_attribute(p + DocumentAttributes.DOCUMENT_ID, doc_id)
            s.set_attribute(p + DocumentAttributes.DOCUMENT_SCORE, score)
            s.set_attribute(p + DocumentAttributes.DOCUMENT_CONTENT, f"chunk {doc_id} text")
        s.set_attribute("rag.bm25_k", 20)
    with tracer.start_as_current_span("rerank") as s:
        s.set_attribute(SpanAttributes.OPENINFERENCE_SPAN_KIND, OpenInferenceSpanKindValues.RERANKER.value)
    with tracer.start_as_current_span("llm.ollama.chat") as s:
        s.set_attribute(SpanAttributes.OPENINFERENCE_SPAN_KIND, OpenInferenceSpanKindValues.LLM.value)
        s.set_attribute(SpanAttributes.LLM_MODEL_NAME, "qwen3:1.7b")
        s.set_attribute(SpanAttributes.LLM_TOKEN_COUNT_PROMPT, 512)
        s.set_attribute(SpanAttributes.LLM_TOKEN_COUNT_COMPLETION, 64)
    root.set_attribute(SpanAttributes.OUTPUT_VALUE, "1889")
tp.force_flush()
time.sleep(3)
with urllib.request.urlopen("http://127.0.0.1:6006/v1/projects/rag-studio-test/spans?limit=20") as r:
    data = json.load(r)
print("spans stored:", [(d["name"], d.get("span_kind")) for d in data.get("data", [])])
