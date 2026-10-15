# The Colab runtime

These two files are everything a Colab runtime needs to serve models to RAG
Studio. You never upload them by hand: `./rag llm colab --copy` (or Settings ›
Model backend › Set up a Colab GPU runtime) packs both into one setup cell,
checked by sha256, built from this folder every time. This folder is the only
copy.

| File | Runs on | What it does |
|---|---|---|
| `server.py` | the runtime | The gateway in front of Ollama: an allowlist (404 before the token for anything else), a bearer token, an 8 MB body cap, no CORS, a content-free log. Standard library only. |
| `start.sh` | the runtime | GPU → zstd → Ollama (pinned to the laptop's version) → models (on the GPU?) → token → gateway (lock proved on loopback) → cloudflared tunnel (lock proved on the public URL). Prints one line to run on the laptop. |

Change either file and the next cell carries the change; the tests in
`tests/test_colab_gateway.py` and `tests/test_remote.py` run both (the gateway
over real HTTP, `start.sh`'s helpers by sourcing it with `RAG_COLAB_LIB=1`).

The full story, the security model and the failure table:
[`docs/02-remote-gpu.md`](../../docs/02-remote-gpu.md).
