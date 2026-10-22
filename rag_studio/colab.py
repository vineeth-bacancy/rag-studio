"""The Colab setup cell: one paste that brings up a GPU runtime for RAG Studio.

The cell carries `tools/colab/server.py`, `worker.py` and `start.sh` (gzip + base64, each
checked against a sha256 taken here), writes them out only when every file
survived the paste, and runs `start.sh` with this job's models. The files in
`tools/colab/` stay the only copy: the cell is rebuilt from them every time,
so there is no second version to drift.

Two variants (spec §13; the trading app's design):

* **Tokenless** (the browser's, and the default): the runtime makes its own
  token and hands it back inside the connect code. No HTTP response from this
  app ever carries a credential.
* **With a token** (`./rag llm colab`, a terminal): the cell embeds a token
  this laptop remembers, so after Colab reclaims the runtime, re-running the
  same cell and pasting only the new URL is the whole reconnect. The cost: the
  token sits in the notebook, which Colab saves to Drive.
"""
from __future__ import annotations

import base64
import gzip
import hashlib
import re
import textwrap
from pathlib import Path

from rag_studio import paths

COLAB_DIR = paths.ROOT / "tools" / "colab"
PAYLOAD_FILES = ("server.py", "worker.py", "start.sh")
PORT = 8000
# The M4 job: the 8B builds and serves its index's questions; the 0.6B is
# there for the local-vs-Colab vector check (the laptop has it too).
DEFAULT_MODELS = ("qwen3-embedding:8b", "qwen3-embedding:0.6b")
WORKER_JOBS = ("qwen3-0.6b-gpu", "qwen3-4b", "qwen3-8b", "context")   # rerankers the worker serves (worker.py MODELS), and the context job
MODEL_NAME = re.compile(r"^[A-Za-z0-9._:/-]{1,80}$")
_WRAP = 92   # Colab's editor does not wrap; a cell is read before it is run


class MissingPayload(FileNotFoundError):
    """A file the cell is built from is missing."""


def _pack(path: Path) -> tuple[str, str]:
    """(base64 of the gzipped bytes, sha256 of the original bytes). mtime=0 so
    the same source always makes the same cell."""
    raw = path.read_bytes()
    return (base64.b64encode(gzip.compress(raw, compresslevel=9, mtime=0)).decode(),
            hashlib.sha256(raw).hexdigest())


def payload(colab_dir: Path | None = None) -> dict[str, dict[str, str]]:
    folder = colab_dir or COLAB_DIR
    out: dict[str, dict[str, str]] = {}
    for name in PAYLOAD_FILES:
        path = folder / name
        if not path.is_file():
            raise MissingPayload(f"{path} is missing, so the Colab cell cannot be built. "
                                 "Restore it from git.")
        blob, digest = _pack(path)
        out[name] = {"b64": blob, "sha256": digest}
    return out


def gateway_version(colab_dir: Path | None = None) -> str:
    text = ((colab_dir or COLAB_DIR) / "server.py").read_text(encoding="utf-8")
    m = re.search(r'^VERSION = "(.*)"', text, re.MULTILINE)
    return m.group(1) if m else ""


def _literal(name: str, blob: str) -> str:
    body = "\n".join(f'        "{line}"' for line in textwrap.wrap(blob, _WRAP))
    return f'    "{name}": (\n{body}\n    ),'


def check_worker(worker: list[str] | tuple[str, ...]) -> list[str]:
    out = [w.strip() for w in worker if w.strip()]
    for w in out:
        if w not in WORKER_JOBS:
            raise ValueError(f"{w[:40]!r} is not a reranker the worker serves. Known: {', '.join(WORKER_JOBS)}.")
    return out


def library_pins() -> str:
    """This laptop's sentence-transformers and transformers, so the worker
    scores with the same code (the numbers a run records must come from one
    implementation)."""
    from importlib.metadata import PackageNotFoundError, version
    out = []
    for name in ("sentence-transformers", "transformers"):
        try:
            out.append(f"{name}=={version(name)}")
        except PackageNotFoundError:
            pass
    return " ".join(out)


def check_models(models: list[str] | tuple[str, ...], *, allow_none: bool = False) -> list[str]:
    out = [m.strip() for m in models if m.strip()]
    if not out and not allow_none:
        raise ValueError("Name at least one model for the runtime to serve.")
    for m in out:
        if not MODEL_NAME.match(m):
            raise ValueError(f"{m[:40]!r} is not a model name.")
    return out


def build_cell(*, models: list[str] | tuple[str, ...] = DEFAULT_MODELS, token: str | None = None,
               ollama_version: str | None = None, port: int = PORT,
               worker: list[str] | tuple[str, ...] = (), pins: str | None = None,
               keep_alive: bool = True, colab_dir: Path | None = None) -> str:
    """The whole cell, as text to paste into one Colab code cell. `worker`
    names the rerankers the GPU worker downloads and serves (no models and a
    worker is a rerank-only runtime: no Ollama is installed)."""
    worker = check_worker(worker)
    models = check_models(models, allow_none=bool(worker))
    if "context" in worker and "qwen3:8b" not in models:
        models = [*models, "qwen3:8b"]        # the context job writes its lines with qwen3:8b through Ollama
    pins = library_pins() if pins is None else pins
    if not re.fullmatch(r"[A-Za-z0-9_.=<>, -]{0,200}", pins):
        raise ValueError("Those library pins are not valid.")
    if ollama_version is not None and not re.fullmatch(r"[0-9][0-9A-Za-z.\-]{0,30}", ollama_version):
        raise ValueError("That is not an Ollama version.")
    files = payload(colab_dir)
    literals = "\n".join(_literal(n, d["b64"]) for n, d in files.items())
    digests = "\n".join(f'    "{n}": "{d["sha256"]}",' for n, d in files.items())
    token_line = (f'os.environ["RAG_STUDIO_TOKEN"] = "{token}"\n'
                  'print("token    supplied by the laptop that made this cell")'
                  if token else 'print("token    this runtime will make its own")')
    pin = ollama_version or ""
    return f'''# ============================================================================
# RAG Studio: a GPU model server on this Colab runtime.
#
# First: Runtime > Change runtime type > T4 GPU. Colab hands out CPU-only
# runtimes without saying so.
#
# This cell writes three files (a gateway, a GPU worker and a start script) and runs them.
# Nothing is uploaded from the laptop: no PDFs, no index, no repository. It
# downloads cloudflared and the models named below (Ollama too, only when it
# serves models; a rerank-only runtime installs none), nothing else.
#
# Made by RAG Studio from tools/colab/. After Colab reclaims the runtime, run
# this same cell again; only the tunnel URL changes.
# ============================================================================
import base64, gzip, hashlib, os, pathlib, subprocess, sys

MODELS = {" ".join(models)!r}
WORKER = {",".join(worker)!r}     # rerankers the GPU worker serves ("" = none)
PINS = {pins!r}                   # the laptop's library versions, so scores match
OLLAMA_VERSION = {pin!r}   # the laptop's, so both sides embed with the same runtime
PORT = {port}
KEEP_ALIVE = {keep_alive!r}   # keep this cell running so Colab does not reclaim an idle runtime

_FILES = {{
{literals}
}}

_SHA256 = {{
{digests}
}}

# A truncated paste must fail before anything runs: half a shell script does
# not fail, it does something else. Damage early in a blob breaks gzip; later
# damage fails the digest. Both say the same thing.
_DECODED = {{}}
for _name, _blob in _FILES.items():
    try:
        _raw = gzip.decompress(base64.b64decode(_blob))
        _got = hashlib.sha256(_raw).hexdigest()
    except Exception as _exc:
        raise SystemExit(f"{{_name}} did not survive the paste ({{type(_exc).__name__}}). "
                         "Copy the whole cell again, including the last line.") from None
    if _got != _SHA256[_name]:
        raise SystemExit(f"{{_name}} did not survive the paste (sha256 {{_got[:12]}} is not "
                         f"{{_SHA256[_name][:12]}}). Copy the whole cell again.")
    _DECODED[_name] = (_raw, _got)

# Written only after every file checked out.
for _name, (_raw, _got) in _DECODED.items():
    pathlib.Path(_name).write_bytes(_raw)
    print(f"wrote    {{_name}}  {{len(_raw)}} bytes  sha256 {{_got[:12]}}")

{token_line}
print(f"models   {{MODELS or 'none (no Ollama)'}}")
print(f"worker   {{WORKER or 'none'}}")
print(f"ollama   {{OLLAMA_VERSION or 'latest'}}")
print("-" * 78)

# Streamed, so the seven checks appear as they happen.
_args = ["bash", "start.sh", "--models", MODELS, "--port", str(PORT)]
if WORKER:
    _args += ["--worker", WORKER, "--pins", PINS]
if OLLAMA_VERSION:
    _args += ["--ollama-version", OLLAMA_VERSION]
_proc = subprocess.Popen(_args, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
for _line in _proc.stdout:
    sys.stdout.write(_line)
    sys.stdout.flush()
_proc.wait()
if _proc.returncode != 0:
    print(f"\\nstart.sh stopped with exit code {{_proc.returncode}}. Read the failure above.")


# The connect code carries this runtime's token, and Colab saves this cell's
# output to Drive with the notebook. So start.sh leaves it in .rag_code, and
# this shows it in an output of its own that is blanked when you press Enter.
def show_connect_code():
    """Show this runtime's connect code once, then blank it from the output."""
    code_file = pathlib.Path(".rag_code")
    if not code_file.exists():
        print("No connect code on this runtime: the laptop already holds the token, "
              "so the URL above is all it needs.")
        return
    code = code_file.read_text().strip()
    text = (f"On the laptop:  ./rag llm connect {{code}}\\n"
            f"Or in the app (Settings > Model backend > Connect):  {{code}}")
    try:
        from IPython.display import Pretty, display
        handle = display(Pretty(text), display_id=True)
    except Exception:
        handle = None
    if handle is None:
        print(text)          # not a notebook, so nothing saves it
        return
    try:
        input("Copy it, then press Enter to blank it from this notebook: ")
    except Exception:
        pass                 # cannot wait: blank it rather than keep it
    handle.update(Pretty("Connect code blanked. show_connect_code() shows it again."))


if _proc.returncode == 0:
    show_connect_code()
{KEEPALIVE_CODE}
# Keeps this cell running (and the GPU touched) so Colab does not reclaim an idle
# runtime. Press the stop button to end it; the runtime keeps running.
if _proc.returncode == 0 and KEEP_ALIVE:
    keep_alive()
'''


# The keep-alive loop, one source for the standalone cell and the end of the
# setup cell. Colab reclaims a runtime that sits idle (no running cell, no GPU
# use); this keeps a cell running and touches the GPU every few minutes by
# scoring two short passages with the model the worker already holds (no extra
# GPU memory, and it checks the worker is alive). Stop it with the cell's stop
# button; the runtime and the worker keep running after that.
KEEPALIVE_CODE = '''
def keep_alive(interval=240, port=8001):
    """Keep this runtime busy until you press the cell's stop button."""
    import json, pathlib, time, urllib.request
    # Say where to connect again, and never clear the output: an earlier
    # version cleared it every tick and erased the URL.
    url_file = pathlib.Path(".rag_url")
    if url_file.exists():
        print(f"Connect from the laptop:  ./rag llm connect {url_file.read_text().strip()}")
    try:
        from IPython.display import Pretty, display
        handle = display(Pretty("keep-alive starting"), display_id=True)
    except Exception:
        handle = None
    started, ticks = time.time(), 0
    while True:
        what = "waiting"
        try:
            info = json.load(urllib.request.urlopen(f"http://127.0.0.1:{port}/worker", timeout=20))
            body = {"model": info["loaded"], "instruction": "Find the passage that answers the question",
                    "query": "keep alive", "documents": ["keep alive", "idle"]}
            req = urllib.request.Request(f"http://127.0.0.1:{port}/rerank", data=json.dumps(body).encode(),
                                         headers={"Content-Type": "application/json"})
            out = json.load(urllib.request.urlopen(req, timeout=120))
            what = f"worker scored 2 passages with {info['loaded']} in {out['compute_ms']:.0f} ms"
        except Exception:
            try:
                import torch
                if torch.cuda.is_available():
                    x = torch.randn(2048, 2048, device="cuda")
                    (x @ x).sum().item()
                    what = "GPU matrix multiply"
            except Exception:
                what = "no GPU work available; the loop alone keeps the cell running"
        ticks += 1
        mins = int((time.time() - started) / 60)
        text = (f"keep-alive: running for {mins} min, {ticks} ticks, last: {what} ({time.strftime('%H:%M:%S')})\\n"
                "Press the stop button to end it. The runtime and the worker keep running.")
        if handle is not None:
            handle.update(Pretty(text))     # one line, updated in place; the rest of the output stays
        else:
            print(text)
        time.sleep(interval)
'''


def build_keepalive_cell(interval: int = 240) -> str:
    """A small cell to paste into a runtime that is already up: it runs until stopped."""
    if not 30 <= interval <= 900:
        raise ValueError("The keep-alive interval is 30 to 900 seconds.")
    return ("# RAG Studio: keep this Colab runtime busy so it is not reclaimed for sitting idle.\n"
            "# Run it in a cell of its own; stop it with the stop button when you are done.\n"
            + KEEPALIVE_CODE + f"\nkeep_alive(interval={interval})\n")


def cell_stats(cell: str) -> dict[str, float | int]:
    return {"characters": len(cell), "lines": cell.count("\n") + 1,
            "kilobytes": round(len(cell.encode()) / 1024, 1)}
