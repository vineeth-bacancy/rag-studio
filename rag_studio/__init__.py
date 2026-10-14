"""RAG Studio engine.

Every RAG step lives in this package: parse, clean, chunk, embed, store,
retrieve, rerank, prompt, answer, evaluate. The API in `web/api/` calls it and
the UI calls the API, so a number cannot be computed two different ways.
"""

__version__ = "0.1.0"
