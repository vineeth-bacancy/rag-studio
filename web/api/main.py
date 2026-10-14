"""HTTP routes only: validate, call the engine, return. No RAG logic lives here.

Run with `./rag web`, which binds 127.0.0.1 and exports the ports the request
guard trusts. Interactive schema at http://127.0.0.1:8775/api/_docs.
"""
from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from guard import RequestGuard, allowed_origins
from rag_studio import __version__, health

app = FastAPI(
    title="RAG Studio API",
    version=__version__,
    docs_url="/api/_docs",
    redoc_url=None,
    openapi_url="/api/openapi.json",
)

# The browser never makes a cross-origin call in practice: next.config.ts
# rewrites /api/* to this process, so everything is same-origin. CORS is the
# fallback for opening the UI directly against this port.
app.add_middleware(CORSMiddleware, allow_origins=allowed_origins(),
                   allow_credentials=False, allow_methods=["*"], allow_headers=["*"])
# Added last, so it runs first: a refused request never reaches CORS or a route.
app.add_middleware(RequestGuard)


@app.get("/api/status")
def get_status() -> dict:
    """What the sidebar and top bar poll: is the AI up, which model answers."""
    return health.status()


@app.get("/api/health")
def get_health() -> dict:
    """The full System Health report. Cheap enough to compute on every request."""
    return health.report()


@app.get("/api/documents")
def get_documents() -> dict:
    """The PDF library. Page counts and parser views arrive with ingestion (M1-M2)."""
    return {"documents": health.documents()}
