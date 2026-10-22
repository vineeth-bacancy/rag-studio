"""System Health: what is working, what is not, and the one fix for each.

Every check reports one status from a single vocabulary (docs/product-language.md):
healthy, running, warning, needs_attention, unavailable, plus `info` for a
state that is fine but worth knowing ("Remote AI is not connected").

A check never raises. A failure to measure something is itself a result and
is reported as `unavailable`, so one broken probe cannot blank the page.
"""
from __future__ import annotations

import os
import platform
import shutil
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from rag_studio import __version__, library, paths, profiles, store
from rag_studio.config import OLLAMA_URL, ROLES
from rag_studio.llm import OllamaClient, OllamaUnavailable, same_model

Status = Literal["healthy", "running", "warning", "needs_attention", "unavailable", "info"]

GB = 1024**3
# Below this, loading a 5 GB model pushes the laptop into swap and an answer
# that takes 25 s can take minutes. Measured context: 15.7 GB total, shared
# with an editor, a browser and the trading app's qwen3:8b.
LOW_MEMORY = 3 * GB
LOW_DISK = 10 * GB


@dataclass
class Check:
    id: str
    status: Status
    label: str                       # what is being checked, e.g. "Local AI"
    message: str                     # short, plain explanation
    fix: str | None = None           # one command to copy, when there is one
    technical: dict[str, Any] = field(default_factory=dict)  # for "Technical details"


@dataclass
class Section:
    id: str
    name: str
    checks: list[Check]


def _plural(n: int, word: str) -> str:
    return f"{n} {word}" if n == 1 else f"{n} {word}s"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _meminfo() -> dict[str, int] | None:
    """Total and available RAM from /proc/meminfo (Linux), in bytes."""
    try:
        values = {}
        for line in Path("/proc/meminfo").read_text().splitlines():
            key, rest = line.split(":", 1)
            if key in ("MemTotal", "MemAvailable", "SwapTotal", "SwapFree"):
                values[key] = int(rest.split()[0]) * 1024
        return values or None
    except (OSError, ValueError):
        return None


def _dir_size(path: Path) -> int:
    total = 0
    for root, _dirs, files in os.walk(path):
        for name in files:
            try:
                total += (Path(root) / name).stat().st_size
            except OSError:
                pass
    return total


def documents() -> list[dict[str, Any]]:
    """The PDFs on disk, with what the active index holds of each."""
    return library.documents()


def ollama_snapshot(client: OllamaClient | None = None) -> dict[str, Any]:
    """Everything Health and the sidebar need to know about Ollama, in one pass."""
    c = client or OllamaClient()
    snap: dict[str, Any] = {"up": False, "host": c.base_url, "version": None,
                            "installed": [], "resident": [], "error": None}
    try:
        snap["version"] = c.version()
        snap["installed"] = [asdict(m) for m in c.installed()]
        snap["resident"] = [asdict(m) for m in c.resident()]
        snap["up"] = True
    except OllamaUnavailable as e:
        snap["error"] = e.reason
    return snap


def role_rows(snap: dict[str, Any]) -> list[dict[str, Any]]:
    """Each role, the model it names, and whether that model is installed."""
    installed = [m["name"] for m in snap["installed"]]
    resident = [m["name"] for m in snap["resident"]]
    rows = []
    for role in ROLES:
        row = {"role": role.key, "label": role.label, "purpose": role.purpose,
               "model": role.model, "runs_on": role.runs_on, "arrives": role.arrives, "where": role.where,
               "installed": None, "loaded": None}
        if role.where == "remote":
            # Checked against what the connected runtime serves, never against the local Ollama,
            # and never with a `ollama pull` suggestion: it does not run here.
            from rag_studio import remote
            conn = remote.connection()
            row["connected"] = bool(conn["connected"])
            row["installed"] = (any(same_model(role.model, m) for m in conn.get("models") or [])
                                if conn["connected"] and role.model else None)
        elif role.model and role.runs_on == "ollama" and snap["up"]:
            row["installed"] = any(same_model(role.model, n) for n in installed)
            row["loaded"] = any(same_model(role.model, n) for n in resident)
        elif role.runs_on == "python":
            # Runs inside this app (sentence-transformers): "installed" is downloaded at the
            # pinned revision, "loaded" is held in this process. Never a download or a load here.
            row["installed"], row["loaded"] = _python_model_state(role.key)
        rows.append(row)
    return rows


def _python_model_state(role_key: str) -> tuple[bool | None, bool | None]:
    try:
        if role_key == "embed":
            from rag_studio.embed import STModel, st_snapshot
            spec = profiles.get(profiles.DEFAULT).embedding
            return st_snapshot(spec) is not None, STModel.is_loaded(spec)
        if role_key == "rerank":
            from rag_studio import rerank
            from rag_studio.config import role_model
            spec = next(s for s in rerank.RERANKERS.values() if s.where == "local" and s.model == role_model("rerank"))
            return rerank.snapshot(spec) is not None, rerank.LocalReranker.is_loaded(spec)
    except (ImportError, KeyError, OSError):
        pass
    return None, None


def technique_defaults() -> list[dict[str, Any]]:
    """What Ask and an evaluation use unless told otherwise (spec §9.6), read
    from the engine's own defaults. Every technique is a switch; the scorecard
    decides which are on, and the user changes a default, not this list."""
    from rag_studio import answer, prompt, retrieve
    d = retrieve.DENSE
    off = "Off"
    return [
        {"key": "index", "label": "Index", "value": profiles.get(profiles.DEFAULT).label, "id": profiles.DEFAULT},
        {"key": "prompt", "label": "Prompt version", "value": prompt.DEFAULT_VERSION},
        {"key": "search", "label": "Search", "value": d.label},
        {"key": "k", "label": "Passages per answer", "value": str(answer.DEFAULT_K)},
        {"key": "section_cap", "label": "Limit per section", "value": off if d.section_cap is None else str(d.section_cap)},
        {"key": "rerank", "label": "Reranker", "value": off if d.rerank is None else d.rerank},
        {"key": "rewrite", "label": "Query rewriting", "value": off if d.rewrite is None else d.rewrite},
        {"key": "agent", "label": "Multi-step search", "value": off if d.agent_rounds is None else f"{d.agent_rounds} rounds"},
        {"key": "context", "label": "Context lines", "value": "In the index" if profiles.get(profiles.DEFAULT).contextual else off},
    ]


def _our_models() -> set[str]:
    """The models this app itself loads into the local Ollama: local roles and the indexes' embedders.
    Remote roles are not here: they run on the Colab runtime."""
    from rag_studio.config import ollama_roles
    mine = {r.model for r in ollama_roles()}
    mine |= {p.embedding.model for p in profiles.PROFILES.values() if p.embedding.backend == "ollama"}
    return mine


def _ai_checks(snap: dict[str, Any], roles: list[dict[str, Any]]) -> list[Check]:
    if not snap["up"]:
        return [Check("ollama", "unavailable", "Local AI",
                      f"AI unavailable. {snap['error']}",
                      fix="systemctl status ollama",
                      technical={"Endpoint": snap["host"]})]

    checks = [Check("ollama", "healthy", "Local AI", "Ollama is running.",
                    technical={"Endpoint": snap["host"], "Version": snap["version"],
                               "Installed models": len(snap["installed"])})]

    local = [r for r in roles if r.get("where", "local") == "local"]
    missing = [r for r in local if r["installed"] is False]
    if missing:
        names = ", ".join(sorted({r["model"] for r in missing}))
        checks.append(Check(
            "roles", "needs_attention", "Models per role",
            f"{_plural(len(missing), 'role')} {'names' if len(missing) == 1 else 'name'} a model that is not installed: {names}.",
            fix=" && ".join(f"ollama pull {m}" for m in sorted({r['model'] for r in missing})),
        ))
    else:
        ready = [r for r in local if r["installed"]]
        checks.append(Check("roles", "healthy", "Models per role",
                            f"All {len(ready)} active roles have their model installed."))

    ours = _our_models()
    resident = snap["resident"]
    foreign = [m for m in resident if not any(same_model(m["name"], o) for o in ours)]
    if foreign:
        names = ", ".join(m["name"] for m in foreign)
        checks.append(Check(
            "busy", "warning", "AI busy",
            f"Another app has {names} loaded. Answers will be slower until it unloads.",
            technical={m["name"]: f"{m['size'] / GB:.1f} GB, unloads {m['expires_at'] or '-'}"
                       for m in foreign},
        ))
    elif resident:
        checks.append(Check(
            "busy", "running", "Loaded models",
            f"{_plural(len(resident), 'model')} loaded and ready: "
            + ", ".join(m["name"] for m in resident) + ".",
        ))
    else:
        checks.append(Check("busy", "info", "Loaded models",
                            "No model is loaded. The first answer loads one, which takes a few seconds."))
    return checks


def _machine_checks() -> list[Check]:
    checks = []
    mem = _meminfo()
    if mem and "MemAvailable" in mem:
        avail, total = mem["MemAvailable"], mem["MemTotal"]
        status: Status = "warning" if avail < LOW_MEMORY else "healthy"
        msg = (f"{avail / GB:.1f} GB of {total / GB:.1f} GB memory is free."
               + (" Large models will load slowly or swap." if status == "warning" else ""))
        checks.append(Check("memory", status, "Memory", msg,
                            technical={"Swap free": f"{mem.get('SwapFree', 0) / GB:.1f} GB"}))
    else:
        checks.append(Check("memory", "unavailable", "Memory", "Couldn't read memory usage."))

    try:
        usage = shutil.disk_usage(paths.DATA if paths.DATA.exists() else paths.ROOT)
        status = "warning" if usage.free < LOW_DISK else "healthy"
        msg = (f"{usage.free / GB:.0f} GB free on the data disk."
               + (" Model downloads and indexes need more room." if status == "warning" else ""))
        checks.append(Check("disk", status, "Disk", msg,
                            technical={"Data folder": str(paths.DATA),
                                       "Data size": f"{_dir_size(paths.DATA) / 1024**2:.0f} MB"}))
    except OSError:
        checks.append(Check("disk", "unavailable", "Disk", "Couldn't read disk usage."))
    return checks


def _library_checks() -> list[Check]:
    docs = documents()
    if docs:
        doc_check = Check("documents", "healthy", "Documents",
                          f"{_plural(len(docs), 'PDF')} in the library.",
                          technical={d["doc_id"]: f"{d['bytes'] / 1024**2:.1f} MB" for d in docs})
    else:
        doc_check = Check("documents", "needs_attention", "Documents",
                          "No PDFs found. Add PDFs to the library to ask questions.",
                          technical={"Folder": str(paths.PDFS)})

    index_check = _index_check(profiles.DEFAULT)
    return [doc_check, index_check]


def _index_check(profile_id: str) -> Check:
    """The active build opens, and holds as many passages as its manifest says."""
    manifest = store.active_manifest(profile_id)
    if manifest is None:
        return Check("indexes", "info", "Indexes",
                     "No search index yet. Build one to start asking questions.",
                     fix=profiles.build_command(profile_id))
    tech = {"Profile": manifest["profile"], "Table": manifest["table"],
            "Built": manifest["built_at"], "Embedding": manifest["embedding"]["model"]}
    try:
        rows = store.open_table(manifest["table"]).count_rows()
    except Exception as e:  # noqa: BLE001 - any failure to open is the finding
        return Check("indexes", "needs_attention", "Indexes",
                     "The search index could not be opened. Rebuild it.",
                     fix=profiles.build_command(profile_id) + " --force", technical=tech | {"Error": type(e).__name__})
    if rows != manifest["rows"]:
        return Check("indexes", "needs_attention", "Indexes",
                     f"The search index holds {rows:,} passages but should hold {manifest['rows']:,}. Rebuild it.",
                     fix=profiles.build_command(profile_id) + " --force", technical=tech)
    docs = len(manifest["documents"])
    return Check("indexes", "healthy", "Indexes",
                 f"{manifest['label']} index: {rows:,} passages from {_plural(docs, 'document')}.",
                 technical=tech | {"Rows": rows})


def _remote_checks() -> list[Check]:
    """The Colab runtime, probed live when one is connected (one request
    through the tunnel per check; the sidebar's status poll never does this)."""
    from rag_studio import remote
    conn = remote.connection()
    if not conn["connected"]:
        return [Check("colab", "info", "Colab runtime",
                      "Not connected. Everything runs on this computer.",
                      fix="./rag llm colab --copy",
                      technical={"Setup": "Settings > Model backend", "Token remembered":
                                 "yes" if conn["token_remembered"] else "no"})]
    info = remote.probe()
    tech: dict[str, Any] = {"Endpoint": conn["host"], "Connected": conn.get("connected_at") or "-"}
    if not info["reachable"]:
        return [Check("colab", "unavailable", "Colab runtime",
                      f"{info['error']} Re-run the setup cell in Colab and connect again.",
                      fix="./rag llm connect", technical=tech | {"Error kind": info.get("error_kind")})]
    gpus = (info.get("gpu") or {}).get("gpus") or []
    tech |= {"Ollama": info["ollama_version"], "Latency": f"{info['latency_ms']} ms",
             "GPU": ", ".join(f"{g['name']} ({g['memory_used_mb']:,} / {g['memory_total_mb']:,} MB)" for g in gpus) or "none",
             "Models": ", ".join(m["name"] for m in info["models"]) or "none"}
    worker = info.get("worker")
    if worker:
        tech["Worker"] = (f"serves {', '.join(worker.get('served') or []) or 'no reranker'}"
                          f"{' and context lines' if worker.get('context') else ''}; "
                          f"loaded {worker.get('loaded') or 'nothing'}"
                          + (f", peak {worker['vram']['peak_mb']:,} MB" if (worker.get("vram") or {}).get("peak_mb") else ""))
    on_cpu = [m["name"] for m in info["loaded"] if not m["on_gpu"]]
    if not gpus:
        return [Check("colab", "warning", "Colab runtime",
                      "Connected, but the runtime has no GPU, so it is no faster than this computer. "
                      "In Colab choose a T4 GPU, then run the setup cell again.", technical=tech)]
    if on_cpu:
        return [Check("colab", "warning", "Colab runtime",
                      f"Connected, but {', '.join(on_cpu)} runs on the runtime's CPU. Run the setup cell again.",
                      technical=tech)]
    if info["missing"]:
        return [Check("colab", "warning", "Colab runtime",
                      f"Connected, but the runtime does not serve {', '.join(info['missing'])}, which the "
                      "Colab indexes need. Run the setup cell with those models.", technical=tech)]
    return [Check("colab", "healthy", "Colab runtime",
                  f"Connected to a {gpus[0]['name']} with {_plural(len(info['models']), 'model')}.",
                  technical=tech)]


def report() -> dict[str, Any]:
    snap = ollama_snapshot()
    roles = role_rows(snap)
    sections = [
        Section("ai", "AI models", _ai_checks(snap, roles)),
        Section("remote", "Remote AI", _remote_checks()),
        Section("library", "Documents and indexes", _library_checks()),
        Section("machine", "This computer", _machine_checks()),
    ]
    statuses = [c.status for s in sections for c in s.checks]
    overall: Status = (
        "needs_attention" if any(s in ("needs_attention", "unavailable") for s in statuses)
        else "warning" if "warning" in statuses
        else "healthy"
    )
    return {
        "checked_at": _now(),
        "overall": overall,
        "counts": {s: statuses.count(s) for s in
                   ("healthy", "running", "warning", "needs_attention", "unavailable", "info")},
        "sections": [asdict(s) for s in sections],
        "ollama": snap,
        "roles": roles,
        "defaults": technique_defaults(),
        "documents": documents(),
        "app": {"version": __version__, "python": sys.version.split()[0],
                "platform": platform.platform(terse=True), "ollama_url": OLLAMA_URL},
    }


def status() -> dict[str, Any]:
    """The small version the sidebar polls: is the AI up, and which model answers."""
    snap = ollama_snapshot()
    answer = next(r for r in role_rows(snap) if r["role"] == "answer")
    ours = _our_models()
    busy = [m["name"] for m in snap["resident"]
            if not any(same_model(m["name"], o) for o in ours)]
    manifest = store.active_manifest(profiles.DEFAULT)
    from rag_studio import embed, ingest, remote
    # Asking loads the default index's embedder in this process the first time.
    spec = profiles.get(profiles.DEFAULT).embedding
    need = embed.ST_LOAD_GB if spec.backend == embed.ST and not embed.STModel.is_loaded(spec) else None
    free = ingest.available_ram_gb()
    conn = remote.connection()
    return {
        "remote": {"connected": conn["connected"], "host": conn.get("host"),
                   "last_error": conn.get("last_error")},
        "ai": {"up": snap["up"], "backend": "local", "answer_model": answer["model"],
               "answer_model_installed": answer["installed"], "busy_with": busy,
               "error": snap["error"]},
        "documents": len(library.pdf_files()),
        "default_profile": profiles.DEFAULT,
        "build_command": profiles.build_command(profiles.DEFAULT),
        # Memory the first question needs and what is free now; Ask explains a
        # shortfall instead of failing on the first question.
        "memory": {"needed_gb": need, "free_gb": round(free, 1) if free is not None else None,
                   "short": bool(need and free is not None and free < need)},
        "indexed": manifest is not None,
        "index": ({"profile": manifest["profile"], "label": manifest["label"],
                   "rows": manifest["rows"], "built_at": manifest["built_at"]}
                  if manifest else None),
        "version": __version__,
    }
