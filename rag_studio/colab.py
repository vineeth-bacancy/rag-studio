"""The Colab setup cell: one paste that brings up a GPU runtime for RAG Studio.

The cell carries `tools/colab/server.py` and `start.sh` (gzip + base64, each
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
PAYLOAD_FILES = ("server.py", "start.sh")
PORT = 8000
# The M4 job: the 8B builds and serves its index's questions; the 0.6B is
# there for the local-vs-Colab vector check (the laptop has it too).
DEFAULT_MODELS = ("qwen3-embedding:8b", "qwen3-embedding:0.6b")
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


def check_models(models: list[str] | tuple[str, ...]) -> list[str]:
    out = [m.strip() for m in models if m.strip()]
    if not out:
        raise ValueError("Name at least one model for the runtime to serve.")
    for m in out:
        if not MODEL_NAME.match(m):
            raise ValueError(f"{m[:40]!r} is not a model name.")
    return out


def build_cell(*, models: list[str] | tuple[str, ...] = DEFAULT_MODELS, token: str | None = None,
               ollama_version: str | None = None, port: int = PORT,
               colab_dir: Path | None = None) -> str:
    """The whole cell, as text to paste into one Colab code cell."""
    models = check_models(models)
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
# This cell writes two files (a gateway and its start script) and runs them.
# Nothing is uploaded from the laptop: no PDFs, no index, no repository. It
# downloads Ollama, cloudflared and the models below, nothing else.
#
# Made by RAG Studio from tools/colab/. After Colab reclaims the runtime, run
# this same cell again; only the tunnel URL changes.
# ============================================================================
import base64, gzip, hashlib, os, pathlib, subprocess, sys

MODELS = {" ".join(models)!r}
OLLAMA_VERSION = {pin!r}   # the laptop's, so both sides embed with the same runtime
PORT = {port}

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
print(f"models   {{MODELS}}")
print(f"ollama   {{OLLAMA_VERSION or 'latest'}}")
print("-" * 78)

# Streamed, so the seven checks appear as they happen.
_args = ["bash", "start.sh", "--models", MODELS, "--port", str(PORT)]
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
'''


def cell_stats(cell: str) -> dict[str, float | int]:
    return {"characters": len(cell), "lines": cell.count("\n") + 1,
            "kilobytes": round(len(cell.encode()) / 1024, 1)}
