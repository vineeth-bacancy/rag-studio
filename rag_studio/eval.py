"""Evaluate: ask every test question with one setup, and score what comes back.

A **run** is a set of test questions × one configuration. It records the
configuration in full (index build, embedding digest, answer model and digest,
options, prompt version and hash), every question's passages, ranks, answer
and outcome, and how long each step took. Two runs are comparable only when
their recorded configurations say what changed (rule 2).

**The scorecard** (spec §7), unchanged from the hand-built project:

* PASS: every expected fact is in the answer, or a trap was refused;
* REFUSED: an answerable question got the exact refusal (a safe failure);
* WRONG: a fact is missing or wrong, or a trap was answered (the dangerous one).

Facts are matched as case-insensitive substrings, and the refusal must be the
whole answer. Keyword scoring checks that facts are present, not that every
claim is supported; the faithfulness judge arrives in M6 (spec §5.8).

Retrieval is scored beside the outcome: **Hit@N** when the first passage from
an answer page is at rank N within the top k, otherwise **Miss**. The run also
records where the first answer page actually ranked (searching 50 deep), which
is how "global growth" is known to sit at rank 11.
"""
from __future__ import annotations

import fcntl
import os
import re
import secrets
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable

import pymupdf

from rag_studio import __version__, paths, profiles
from rag_studio.answer import DEFAULT_K, GenerationSpec, ask, is_refusal
from rag_studio.config import role_model
from rag_studio.files import read_json, write_json_atomic
from rag_studio.llm import OllamaClient
from rag_studio.prompt import DEFAULT_VERSION, prompt_sha256
from rag_studio.retrieve import Retriever, SearchSpec, retriever as active_retriever

DEEP_K = 50          # how far down to look for the first answer page
OUTCOMES = ("PASS", "REFUSED", "WRONG")


# --------------------------------------------------------------------------- test questions

class GoldenError(ValueError):
    """A test-question file is malformed. The message names the file and the question."""


def load_sets() -> dict[str, dict[str, Any]]:
    """Every test-question set in data/golden/, validated, by id."""
    sets: dict[str, dict[str, Any]] = {}
    for path in sorted(paths.GOLDEN.glob("*.json")):
        data = read_json(path)
        _validate(path.name, data)
        sets[data["id"]] = data
    return sets


def targets(q: dict[str, Any]) -> list[dict[str, Any]]:
    """Where a question's answer is: one entry per document that must be
    retrieved. Most questions name one `doc_id` with its `answer_pages`; a
    question that needs two documents lists them under `answers`."""
    if q.get("answers"):
        return [{"doc_id": a["doc_id"], "pages": list(a["pages"])} for a in q["answers"]]
    if q.get("doc_id"):
        return [{"doc_id": q["doc_id"], "pages": list(q.get("answer_pages") or [])}]
    return []


def _valid_fact(f: Any) -> bool:
    if isinstance(f, str):
        return bool(f)
    if isinstance(f, list):
        return bool(f) and all(isinstance(x, str) and x for x in f)
    if isinstance(f, dict) and isinstance(f.get("pattern"), str) and isinstance(f.get("label"), str):
        try:
            re.compile(f["pattern"])
        except re.error:
            return False
        return True
    return False


def _validate(name: str, data: Any) -> None:
    if not isinstance(data, dict) or not isinstance(data.get("questions"), list):
        raise GoldenError(f"{name}: expected an object with a 'questions' list.")
    for key in ("id", "name", "description", "scored"):
        if key not in data:
            raise GoldenError(f"{name}: missing '{key}'.")
    seen = set()
    for q in data["questions"]:
        qid = q.get("id")
        if not qid or qid in seen:
            raise GoldenError(f"{name}: question ids must be present and unique ({qid!r}).")
        seen.add(qid)
        if not str(q.get("question", "")).strip():
            raise GoldenError(f"{name} {qid}: empty question.")
        trap = q.get("expect") == "REFUSE"
        where = targets(q)
        if trap and where:
            raise GoldenError(f"{name} {qid}: a trap has no document or answer pages.")
        if not trap:
            if not where or any(not t["pages"] for t in where):
                raise GoldenError(f"{name} {qid}: an answerable question needs a document and its answer pages.")
            if any(not isinstance(p, int) or p < 1 for t in where for p in t["pages"]):
                raise GoldenError(f"{name} {qid}: answer pages are 1-based PDF pages.")
        exp = q.get("expect")
        if not (exp is None or exp == "REFUSE" or (isinstance(exp, list) and exp
                                                   and all(_valid_fact(f) for f in exp))):
            raise GoldenError(f"{name} {qid}: 'expect' is a list of facts, \"REFUSE\" or null. "
                              "A fact is a string, a list of alternatives, or {pattern, label}.")
        if data["scored"] and exp is None:
            raise GoldenError(f"{name} {qid}: a scored set needs expected facts for every question.")


def check_pages(sets: dict[str, dict[str, Any]]) -> list[str]:
    """Problems with answer pages against the PDFs on disk (a page past the end,
    an unknown document). Empty when everything checks out."""
    counts: dict[str, int] = {}
    for p in paths.PDFS.glob("*.pdf"):
        with pymupdf.open(p) as d:
            counts[p.stem] = d.page_count
    problems = []
    for s in sets.values():
        for q in s["questions"]:
            for t in targets(q):
                doc = t["doc_id"]
                if doc not in counts:
                    problems.append(f"{s['id']} {q['id']}: no document {doc!r}.")
                elif any(p > counts[doc] for p in t["pages"]):
                    problems.append(f"{s['id']} {q['id']}: a page is past the end of {doc} ({counts[doc]} pages).")
    return problems


# --------------------------------------------------------------------------- scoring

def first_answer_rank(hits: list[dict[str, Any]], where: list[dict[str, Any]]) -> int | None:
    """The rank at which every target document has a passage from one of its
    answer pages. One document: the first such rank. Two documents: the rank
    where the second of them first appears. None when any is missing."""
    if not where:
        return None
    ranks = []
    for t in where:
        r = next((h["rank"] for h in hits if h["doc_id"] == t["doc_id"] and h["page"] in t["pages"]), None)
        if r is None:
            return None
        ranks.append(r)
    return max(ranks)


def fact_label(fact: Any) -> str:
    if isinstance(fact, str):
        return fact
    if isinstance(fact, list):
        return " or ".join(fact)
    return fact["label"]


def fact_found(fact: Any, text: str) -> bool:
    """A string is a case-insensitive substring (the baseline rule, kept for
    Set A); a list is any of its strings; {pattern} is a case-insensitive
    regex, for numbers that need a boundary ('5%' must not match '45%')."""
    if isinstance(fact, str):
        return fact.lower() in text.lower()
    if isinstance(fact, list):
        return any(f.lower() in text.lower() for f in fact)
    return re.search(fact["pattern"], text, re.I) is not None


def score_answer(expect: list[Any] | str | None, text: str | None) -> dict[str, Any]:
    """The scorecard outcome for one answer. `expect` None means the question is
    scored for retrieval only, so there is no outcome."""
    if expect is None or text is None:
        return {"outcome": None, "found": [], "missing": [], "refused": None}
    refused = is_refusal(text)
    if expect == "REFUSE":
        return {"outcome": "PASS" if refused else "WRONG", "found": [], "missing": [],
                "refused": refused}
    found = [fact_label(f) for f in expect if fact_found(f, text)]
    missing = [fact_label(f) for f in expect if not fact_found(f, text)]
    outcome = "REFUSED" if refused else "PASS" if not missing else "WRONG"
    return {"outcome": outcome, "found": found, "missing": missing, "refused": refused}


# --------------------------------------------------------------------------- runs

@dataclass
class RunSpec:
    sets: list[str]
    mode: str = "answer"              # "answer" (full scorecard) or "retrieval" (no model)
    k: int = DEFAULT_K
    model: str | None = None          # answer model; the Answer role's model when None
    profile: str = profiles.DEFAULT
    prompt_version: str = DEFAULT_VERSION
    label: str | None = None
    ladder: dict[str, Any] | None = field(default=None)   # {"row": 1, "label": "Baseline"}
    search: dict[str, Any] = field(default_factory=lambda: {"method": "dense", "section_cap": None})

    @property
    def search_spec(self) -> SearchSpec:
        return SearchSpec(**self.search)


class RunBusy(RuntimeError):
    """Another run holds the lock."""


class _RunLock:
    """One run at a time, across processes (the CLI and the API). Two runs on
    this 4-core CPU would each take twice as long and make every timing in both
    meaningless."""

    def __init__(self) -> None:
        paths.RUNS.mkdir(parents=True, exist_ok=True)
        self._fd = os.open(paths.RUNS / ".lock", os.O_RDWR | os.O_CREAT, 0o644)

    def acquire(self) -> None:
        try:
            fcntl.flock(self._fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            os.close(self._fd)
            raise RunBusy("An evaluation run is already in progress. Wait for it to finish, or cancel it.") from None

    def release(self) -> None:
        fcntl.flock(self._fd, fcntl.LOCK_UN)
        os.close(self._fd)


def run_is_active() -> bool:
    lock = _RunLock()
    try:
        lock.acquire()
    except RunBusy:
        return True
    lock.release()
    return False


def new_run_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S-") + secrets.token_hex(2)


def run_path(run_id: str):
    if not run_id.replace("-", "").isalnum():
        raise ValueError("Not a run id.")
    return paths.RUNS / f"{run_id}.json"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _config(spec: RunSpec, r: Retriever, client: OllamaClient) -> dict[str, Any]:
    search = spec.search_spec
    cfg: dict[str, Any] = {
        "mode": spec.mode, "k": spec.k, "deep_k": DEEP_K, "index": r.index_info(),
        "retrieval": search.describe(),
        "answer": None, "app_version": __version__, "ollama_version": client.version(),
    }
    if search.method != "dense":
        cfg["retrieval"]["word_index"] = r.word_index_info()
    if spec.mode == "answer":
        gen = GenerationSpec(model=spec.model or role_model("answer"))
        info = client.model_info(gen.model)
        if info is None:
            raise RuntimeError(f"The answer model {gen.model} is not installed. "
                               f"Install it with: ollama pull {gen.model}")
        cfg["answer"] = {**asdict(gen), "digest": info.digest, "quantization": info.quantization,
                         "parameter_size": info.parameter_size,
                         "prompt_version": spec.prompt_version,
                         "prompt_sha256": prompt_sha256(spec.prompt_version)}
    return cfg


def summarize(results: list[dict[str, Any]], sets: list[str]) -> dict[str, Any]:
    recall = {}
    for sid in sets:
        rows = [x for x in results if x["set"] == sid and x["targets"]]
        recall[sid] = {"hits": sum(1 for x in rows if x["hit_rank"] is not None), "total": len(rows)}
    scored = [x for x in results if x["outcome"] is not None]
    return {
        "recall": recall,
        "outcomes": {o: sum(1 for x in scored if x["outcome"] == o) for o in OUTCOMES} if scored else None,
        "scored": len(scored),
        "questions": len(results),
        "errors": sum(1 for x in results if x.get("error")),
        "answer_seconds": round(sum(x["timings"].get("generate_ms") or 0 for x in results) / 1000, 1),
    }


def execute(spec: RunSpec, *, run_id: str | None = None, cancel: Callable[[], bool] | None = None,
            on_start: Callable[[], None] | None = None,
            on_result: Callable[[dict[str, Any], dict[str, Any]], None] | None = None,
            client: OllamaClient | None = None) -> dict[str, Any]:
    """Run every question in order, saving the run after each one, so a page
    reload or a crash leaves the finished questions on disk."""
    if spec.mode not in ("answer", "retrieval"):
        raise ValueError("mode is 'answer' or 'retrieval'.")
    all_sets = load_sets()
    unknown = [s for s in spec.sets if s not in all_sets]
    if unknown or not spec.sets:
        raise ValueError(f"Unknown test-question sets: {', '.join(unknown) or '(none chosen)'}.")
    lock = _RunLock()
    lock.acquire()
    try:
        client = client or OllamaClient()
        r = active_retriever(spec.profile)
        if spec.search_spec.method != "dense" and not r.has_word_index():
            raise RuntimeError(f"The {r.profile.label} index has no word index yet, so BM25 and hybrid "
                               f"search are unavailable. Add it with ./rag ingest --profile {spec.profile}.")
        questions = [(sid, q) for sid in spec.sets for q in all_sets[sid]["questions"]]
        run = {
            "id": run_id or new_run_id(), "status": "running", "started_at": _now(),
            "finished_at": None, "label": spec.label, "ladder": spec.ladder,
            "spec": asdict(spec), "config": _config(spec, r, client),
            "sets": {sid: {"name": all_sets[sid]["name"], "scored": all_sets[sid]["scored"]}
                     for sid in spec.sets},
            "progress": {"done": 0, "total": len(questions)},
            "results": [], "summary": None, "error": None, "pid": os.getpid(),
        }
        path = run_path(run["id"])
        write_json_atomic(path, run)
        if on_start:
            on_start()
        t_run = time.perf_counter()
        for sid, q in questions:
            if cancel and cancel():
                run["status"] = "cancelled"
                break
            res = _one(sid, q, all_sets[sid]["scored"], spec, r, client)
            run["results"].append(res)
            run["progress"]["done"] += 1
            run["summary"] = summarize(run["results"], spec.sets)
            write_json_atomic(path, run)
            if on_result:
                on_result(run, res)
        else:
            run["status"] = "done"
        run["finished_at"] = _now()
        run["seconds"] = round(time.perf_counter() - t_run, 1)
        run["summary"] = summarize(run["results"], spec.sets)
        write_json_atomic(path, run)
        return run
    finally:
        lock.release()


def _one(sid: str, q: dict[str, Any], scored: bool, spec: RunSpec, r: Retriever,
         client: OllamaClient) -> dict[str, Any]:
    hits, found = r.search(q["question"], DEEP_K, spec.search_spec)
    t = found["timings"]
    top = hits[:spec.k]
    where = targets(q)
    res: dict[str, Any] = {
        "set": sid, "id": q["id"], "category": q.get("category"), "question": q["question"],
        "targets": where, "expect": q.get("expect"),
        "expect_labels": ([fact_label(f) for f in q["expect"]] if isinstance(q.get("expect"), list)
                          else q.get("expect")),
        "hits": [{k: h.get(k) for k in ("rank", "chunk_id", "doc_id", "page", "score", "ranks", "capped")}
                 for h in top],
        "hit_rank": first_answer_rank(top, where),
        "best_rank": first_answer_rank(hits, where),
        "answer": None, "outcome": None, "found": [], "missing": [], "refused": None,
        "citations": None, "usage": None, "error": None,
        "timings": {"embed_ms": round(t["embed_ms"], 1) if "embed_ms" in t else None,
                    "search_ms": round(t.get("search_ms", 0) + t.get("bm25_ms", 0) + t.get("fuse_ms", 0), 1),
                    "generate_ms": None},
    }
    if spec.mode == "answer" and scored:
        done = None
        for event, data in ask(q["question"], k=spec.k, model=spec.model, profile_id=spec.profile,
                               prompt_version=spec.prompt_version, search=spec.search_spec,
                               retriever=r, client=client):
            if event in ("done", "error"):
                done = (event, data)
        assert done is not None
        event, data = done
        if event == "error":
            res["error"] = data["error"]
        else:
            # The answer path searched again with k; it must have seen the same
            # passages this row reports, or the row would describe one search
            # and score another.
            seen = [s["chunk_id"] for s in data["sources"]]
            if seen != [h["chunk_id"] for h in top]:
                res["error"] = "The answer used different passages from the scored search."
            res["answer"] = data["answer"]
            res["citations"] = data["citations"]
            res["usage"] = data["usage"]
            res["timings"]["generate_ms"] = next(
                (s["ms"] for s in data["trace"] if s["id"] == "generate"), None)
            res.update(score_answer(q.get("expect"), data["answer"]))
    return res


def mark_failed(run_id: str, error: str) -> None:
    """Record why a run stopped, keeping the questions it finished."""
    path = run_path(run_id)
    run = read_json(path)
    if run:
        run.update(status="failed", error=error, finished_at=_now())
        write_json_atomic(path, run)


# --------------------------------------------------------------------------- the ladder

# Spec §8: each row is one change from the row before it. A row's numbers come
# from runs tagged with it; until then every cell is "-".
LADDER: list[dict[str, Any]] = [
    {"row": 1, "change": "Baseline: PyMuPDF, NFKC, fixed passages, nomic-embed-text, dense search", "milestone": "M1"},
    {"row": 2, "change": "Docling and the full cleaning order", "milestone": "M2"},
    {"row": 3, "change": "Structure-aware passages with section labels", "milestone": "M3"},
    {"row": 4, "change": "Hybrid search (dense + BM25)", "milestone": "M3"},
    {"row": 5, "change": "Qwen3-Embedding 0.6B, then 8B on Colab", "milestone": "M4"},
    {"row": 6, "change": "Reranking (Qwen3-Reranker)", "milestone": "M5"},
    {"row": 7, "change": "Contextual retrieval", "milestone": "M5"},
    {"row": 8, "change": "Query rewriting", "milestone": "M5"},
    {"row": 9, "change": "The agentic loop", "milestone": "M5"},
    {"row": 10, "change": "A new answer model", "milestone": "M6"},
    {"row": 11, "change": "Figure paths", "milestone": "M7"},
]


def config_key(cfg: dict[str, Any]) -> tuple:
    """What must match for two runs to sit in one ladder row."""
    index = cfg.get("index") or {}
    answer = cfg.get("answer") or {}
    retrieval = cfg.get("retrieval") or {}
    return (index.get("table"), (index.get("embedding") or {}).get("digest"), cfg.get("k"),
            answer.get("model"), answer.get("digest"), answer.get("prompt_sha256"),
            retrieval.get("method", "dense"), retrieval.get("section_cap"))


def tag_ladder(run_id: str, row: int, label: str | None = None) -> dict[str, Any]:
    if not any(r["row"] == row for r in LADDER):
        raise ValueError(f"The ladder has rows 1-{len(LADDER)}.")
    path = run_path(run_id)
    run = read_json(path)
    if run is None:
        raise ValueError(f"No run {run_id}.")
    if run["status"] != "done":
        raise ValueError(f"Run {run_id} is {run['status']}; only a finished run can be a ladder row.")
    run["ladder"] = {"row": row, "label": label or next(r["change"] for r in LADDER if r["row"] == row)}
    write_json_atomic(path, run)
    return run


def ladder() -> list[dict[str, Any]]:
    """Every ladder row, each with its variants: runs tagged with the row,
    grouped by setup (row 2 has "PyMuPDF + cleaning" and "Docling + cleaning").
    A variant shows the latest finished result per test set; `mixed` is true
    when its runs were not made with one configuration, so its numbers must
    not be read together. A row with no tagged run has no variants: "-"."""
    runs = [r for r in (get_run(b["id"]) for b in list_runs()) if r and r.get("ladder")
            and r["status"] == "done"]
    out = []
    for step in LADDER:
        tagged = [r for r in runs if r["ladder"]["row"] == step["row"]]   # newest first
        keys: list[tuple] = []
        for r in tagged:
            if variant_key(r["config"]) not in keys:
                keys.append(variant_key(r["config"]))
        variants = []
        for key in reversed(keys):                                         # oldest variant first
            group = [r for r in tagged if variant_key(r["config"]) == key]
            label = min((r["ladder"]["label"] for r in group), key=len)
            variants.append(_variant(label, group))
        out.append({**step, "variants": variants})
    return out


def variant_key(cfg: dict[str, Any]) -> tuple:
    """A ladder variant is one setup: index profile, embedding, k, answer
    model, prompt and search method. Rebuilding the same index (a new table) is not a new
    variant, but `mixed` reports it."""
    return (((cfg.get("index") or {}).get("profile")),) + config_key(cfg)[1:]


def _variant(label: str, tagged: list[dict[str, Any]]) -> dict[str, Any]:
    per_set: dict[str, dict[str, Any]] = {}
    used: list[dict[str, Any]] = []
    for r in tagged:
        for sid in r["spec"]["sets"]:
            if sid in per_set:
                continue
            rows = [x for x in r["results"] if x["set"] == sid]
            per_set[sid] = {
                "run_id": r["id"],
                "recall": (r["summary"]["recall"] or {}).get(sid),
                "outcomes": ({o: sum(1 for x in rows if x["outcome"] == o) for o in OUTCOMES}
                             if any(x["outcome"] for x in rows) else None),
            }
            if r not in used:
                used.append(r)
    return {"label": label, "sets": per_set,
            "runs": [{"id": r["id"], "label": r.get("label"), "started_at": r["started_at"]} for r in used],
            "config": _brief(used[0]["config"]) if used else None,
            "mixed": len({config_key(r["config"]) for r in used}) > 1}


# --------------------------------------------------------------------------- comparing two runs

def _flat(d: Any, prefix: str = "") -> dict[str, Any]:
    if isinstance(d, dict):
        out: dict[str, Any] = {}
        for k, v in d.items():
            out.update(_flat(v, f"{prefix}.{k}" if prefix else str(k)))
        return out
    return {prefix: d}


# Keys that differ between any two runs and say nothing about the setup.
_NOISE = ("index.built_at", "index.table", "app_version")


def compare(a_id: str, b_id: str) -> dict[str, Any]:
    """Two runs side by side (spec §8): what differs in their configuration,
    and every question both asked, with the rows whose outcome or retrieval
    changed marked. Run A is "before", run B is "after"."""
    a, b = get_run(a_id), get_run(b_id)
    if a is None or b is None:
        raise ValueError("No such run. It may have been deleted.")
    fa, fb = _flat(a["config"]), _flat(b["config"])
    config = [{"key": k, "a": fa.get(k), "b": fb.get(k)} for k in sorted(set(fa) | set(fb))
              if fa.get(k) != fb.get(k) and not k.startswith(_NOISE)]
    by_a = {(x["set"], x["id"]): x for x in a["results"]}
    rows = []
    for x in b["results"]:
        y = by_a.get((x["set"], x["id"]))
        if y is None:
            continue
        hit_a, hit_b = y["hit_rank"] is not None, x["hit_rank"] is not None
        rows.append({
            "set": x["set"], "id": x["id"], "question": x["question"], "category": x.get("category"),
            "a": _brief_row(y), "b": _brief_row(x),
            "outcome_changed": y["outcome"] != x["outcome"],
            "retrieval_changed": hit_a != hit_b,
            "rank_moved": (y["best_rank"] or 0) != (x["best_rank"] or 0),
        })
    sets = sorted({r["set"] for r in rows})
    return {"a": _run_head(a), "b": _run_head(b), "config": config, "rows": rows,
            "summary": {"a": summarize([r for r in a["results"] if r["set"] in sets], sets),
                        "b": summarize([r for r in b["results"] if r["set"] in sets], sets)},
            "changed": sum(1 for r in rows if r["outcome_changed"] or r["retrieval_changed"])}


def _brief_row(x: dict[str, Any]) -> dict[str, Any]:
    return {k: x.get(k) for k in ("outcome", "hit_rank", "best_rank", "answer", "found", "missing",
                                  "refused", "error")}


def _run_head(r: dict[str, Any]) -> dict[str, Any]:
    return {k: r.get(k) for k in ("id", "label", "ladder", "started_at", "status")} | {
        "config": _brief(r.get("config") or {})}


# --------------------------------------------------------------------------- history

def list_runs() -> list[dict[str, Any]]:
    """Every run, newest first, without per-question detail. A run still marked
    running while no process holds the lock was interrupted (the API restarted
    or the terminal closed), and is reported as such."""
    if not paths.RUNS.is_dir():
        return []
    active = run_is_active()
    out = []
    for p in sorted(paths.RUNS.glob("*.json"), reverse=True):
        run = read_json(p)
        if not run:
            continue
        if run["status"] == "running" and not active:
            run["status"] = "interrupted"
        out.append({k: run.get(k) for k in ("id", "status", "started_at", "finished_at", "label",
                                            "ladder", "spec", "sets", "progress", "summary",
                                            "seconds", "error")}
                   | {"config": _brief(run.get("config") or {})})
    return out


def get_run(run_id: str) -> dict[str, Any] | None:
    run = read_json(run_path(run_id))
    if run and run["status"] == "running" and not run_is_active():
        run["status"] = "interrupted"
    if run:
        for r in run.get("results", []):
            _upgrade_row(r)
    return run


def _upgrade_row(r: dict[str, Any]) -> None:
    """Rows saved before 2026-10-15 named one `doc_id` and `answer_pages`
    instead of `targets`. The same facts, in the current shape."""
    if "targets" not in r:
        r["targets"] = ([{"doc_id": r["doc_id"], "pages": r.get("answer_pages") or []}]
                        if r.get("doc_id") else [])
    if "expect_labels" not in r:
        exp = r.get("expect")
        r["expect_labels"] = [fact_label(f) for f in exp] if isinstance(exp, list) else exp
    r.setdefault("category", None)


def _brief(cfg: dict[str, Any]) -> dict[str, Any]:
    index = cfg.get("index") or {}
    answer = cfg.get("answer") or {}
    return {"mode": cfg.get("mode"), "k": cfg.get("k"), "profile": index.get("profile"),
            "index_label": index.get("label"), "table": index.get("table"),
            "parser": (index.get("config") or {}).get("parser"),
            "cleaner": (index.get("config") or {}).get("cleaner"),
            "embedding_model": (index.get("embedding") or {}).get("model"),
            "retrieval": (cfg.get("retrieval") or {}).get("method"),
            "section_cap": (cfg.get("retrieval") or {}).get("section_cap"),
            "answer_model": answer.get("model"), "prompt_version": answer.get("prompt_version")}

