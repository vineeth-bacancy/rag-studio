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
claim is supported. The faithfulness judge (spec §5.8) was cut from this
version (2026-10-22), so this scorecard is the only judge.

Retrieval is scored beside the outcome: **Hit@N** when the first passage from
an answer page is at rank N within the top k, otherwise **Miss**. The run also
records where the first answer page actually ranked (searching 50 deep), which
is how "global growth" is known to sit at rank 11.
"""
from __future__ import annotations

import fcntl
import json
import os
import re
import secrets
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from math import comb
from typing import Any, Callable

import pymupdf

from rag_studio import __version__, paths, profiles, rerank as reranking, retrieve
from rag_studio.answer import DEFAULT_K, GenerationSpec, ask, is_refusal
from rag_studio.config import role_model
from rag_studio.files import read_json, write_json_atomic
from rag_studio.llm import OllamaClient, OllamaError, OllamaUnavailable
from rag_studio.prompt import DEFAULT_VERSION, messages_sha256, prompt_sha256
from rag_studio.rerank import RerankError
from rag_studio.retrieve import Retriever, SearchSpec, retriever as active_retriever
from rag_studio.rewrite import RewriteBadOutput, RewriteUnavailable

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
        hist = q.get("history")
        if hist is not None and not (isinstance(hist, list) and hist and all(
                isinstance(t, dict) and str(t.get("question", "")).strip() and str(t.get("answer", "")).strip()
                for t in hist)):
            raise GoldenError(f"{name} {qid}: 'history' is a list of recorded turns, each with a question and an answer.")
        for key in ("standalone_expect", "standalone_forbid"):
            terms = q.get(key)
            if terms is not None and not (isinstance(terms, list) and terms and all(isinstance(x, str) and x for x in terms)):
                raise GoldenError(f"{name} {qid}: '{key}' is a list of words the rewritten question must "
                                  f"{'contain' if key == 'standalone_expect' else 'not contain'}.")
        if (q.get("standalone_expect") or q.get("standalone_forbid")) and not hist:
            raise GoldenError(f"{name} {qid}: a rewrite can only be checked for a question with a 'history'.")
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


def rewrite_ok(q: dict[str, Any], standalone: str | None) -> bool | None:
    """Is a follow-up's rewritten question right? It must contain every word the
    question's `standalone_expect` lists and none of its `standalone_forbid`
    (case-insensitive substrings). None when the question has no history to rewrite
    or was asked as typed. The rewrite is judged first, then retrieval, then the answer."""
    if not q.get("history") or standalone is None:
        return None
    low = standalone.lower()
    return (all(t.lower() in low for t in q.get("standalone_expect") or [])
            and not any(t.lower() in low for t in q.get("standalone_forbid") or []))


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


def fact_spans(fact: Any, text: str) -> list[tuple[int, int]]:
    """Where a fact matches in an answer, by the same rule as `fact_found`
    (every match, in order), so How it works can mark exactly what the
    scorecard counted."""
    if isinstance(fact, dict):
        return [m.span() for m in re.finditer(fact["pattern"], text, re.I)]
    low = text.lower()
    spans = []
    for alt in ([fact] if isinstance(fact, str) else fact):
        needle, at = alt.lower(), 0
        while needle and (i := low.find(needle, at)) >= 0:
            spans.append((i, i + len(needle)))
            at = i + len(needle)
    return sorted(spans)


# --------------------------------------------------------------------------- how sure a run is

def pass_interval(passes: int, n: int, z: float = 1.96) -> tuple[float, float] | None:
    """The Wilson 95% interval for a pass rate of `passes` of `n` questions:
    the range of true rates this one run is consistent with. None for n = 0."""
    if n <= 0:
        return None
    p = passes / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5) / (1 + z * z / n)
    return (round(max(0.0, centre - half), 4), round(min(1.0, centre + half), 4))


def sign_test(better: int, worse: int) -> float:
    """Two runs on the same questions: the exact two-sided sign test (McNemar's
    exact test) on the questions that changed. How often a split at least this
    lopsided happens by chance if neither setup is better. 1.0 when nothing changed."""
    n = better + worse
    if n == 0:
        return 1.0
    tail = sum(comb(n, i) for i in range(min(better, worse) + 1)) / 2 ** n
    return round(min(1.0, 2 * tail), 4)


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
    # Replay a reranker's remembered scores when the same question and passages
    # were scored before (rerank.CachedScorer). Off for a fresh timing measurement.
    rerank_cache: bool = True
    # Follow-up questions (Set E) are searched as typed, without the rewrite: the table that shows what memory is worth.
    as_typed: bool = False

    @property
    def search_spec(self) -> SearchSpec:
        return SearchSpec(**self.search)

    @property
    def two_pass(self) -> bool:
        """Reranking (and, later, rewriting) holds models that must not sit in
        memory beside the answer model's own, so every question is searched
        first, those models are released, and then every question is answered.
        A plain run keeps one question at a time, exactly as before."""
        return self.search_spec.extended


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
    if search.rewrite:
        from rag_studio import rewrite
        cfg["retrieval"]["rewrite"] = rewrite.describe(search.rewrite)
    if search.agent_rounds:
        from rag_studio import agent
        cfg["retrieval"]["agent"] = agent.describe(search.agent_rounds)
        del cfg["retrieval"]["agent_rounds"]
    if search.rewrite or search.agent_rounds:
        cfg["runtime"] = {**(cfg.get("runtime") or {}), **_remote_runtime()}
    if search.rerank:
        # The component's stable description (model, revision, dtype, where,
        # depth, instruction hash) replaces the bare id; measurements that
        # vary (versions, threads) go under `runtime`, never `retrieval`.
        cfg["retrieval"]["rerank"] = reranking.get(search.rerank).describe(search.rerank_depth)
        del cfg["retrieval"]["rerank_depth"]
        cfg["runtime"] = {**(cfg.get("runtime") or {}), **reranking.runtime_facts(search.rerank)}
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


def ranking_stats(results: list[dict[str, Any]]) -> dict[str, Any]:
    """MRR and recall@20 over the questions that have an answer page, from the
    deep search's `best_rank` (so runs made before M5 have them too), and the
    same split by single-document and multi-document questions. A question
    with no answer page does not count; with none counted, the value is None
    (shown as "-", never 0)."""
    def stats(rows: list[dict[str, Any]]) -> dict[str, Any]:
        rows = [x for x in rows if x["targets"]]
        if not rows:
            return {"n": 0, "mrr": None, "recall20": None}
        return {"n": len(rows),
                "mrr": round(sum(1 / x["best_rank"] for x in rows if x["best_rank"]) / len(rows), 4),
                "recall20": sum(1 for x in rows if x["best_rank"] and x["best_rank"] <= 20)}
    return {**stats(results),
            "single": stats([x for x in results if len(x["targets"]) == 1]),
            "multi": stats([x for x in results if len(x["targets"]) > 1])}


def _rewrite_counts(results: list[dict[str, Any]]) -> dict[str, int] | None:
    """How many follow-ups were rewritten right, of those that were rewritten; None when none were."""
    rows = [x for x in results if x.get("rewrite_ok") is not None]
    return {"ok": sum(1 for x in rows if x["rewrite_ok"]), "total": len(rows)} if rows else None


def followup_stats(results: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Set E's chain, judged in order: of the follow-up questions, how many were
    rewritten right, then found (a Hit), then answered right (a Pass). None
    when the run asked no follow-up. `rewritten` is 0 when they were asked as typed."""
    rows = [x for x in results if x.get("history")]
    if not rows:
        return None
    scored = [x for x in rows if x.get("outcome") is not None]
    return {"n": len(rows),
            "rewritten": sum(1 for x in rows if x.get("rewrite_ok") is not None),
            "rewrite_ok": sum(1 for x in rows if x.get("rewrite_ok")),
            "hits": sum(1 for x in rows if x.get("hit_rank") is not None),
            "scored": len(scored),
            "passes": sum(1 for x in scored if x["outcome"] == "PASS")}


def _remote_runtime() -> dict[str, Any]:
    """What a run records about the runtime that wrote its rewrites and decisions: volatile facts, never `retrieval`."""
    from rag_studio import remote
    try:
        if not remote.connected():
            return {"remote_llm": {"runtime": "not connected: rewrites come from the cache"}}
        gpu = (remote.client().gpu() or {}).get("gpus") or []
        return {"remote_llm": {"gpu": gpu[0]["name"] if gpu else None, "model": "qwen3:8b"}}
    except Exception:  # noqa: BLE001 - a record of the runtime must never fail the run
        return {"remote_llm": {"runtime": "unknown"}}


def summarize(results: list[dict[str, Any]], sets: list[str]) -> dict[str, Any]:
    recall = {}
    for sid in sets:
        rows = [x for x in results if x["set"] == sid and x["targets"]]
        recall[sid] = {"hits": sum(1 for x in rows if x["hit_rank"] is not None), "total": len(rows)}
    scored = [x for x in results if x["outcome"] is not None]
    return {
        "recall": recall,
        "ranking": ranking_stats(results),
        "outcomes": {o: sum(1 for x in scored if x["outcome"] == o) for o in OUTCOMES} if scored else None,
        "scored": len(scored),
        "questions": len(results),
        "errors": sum(1 for x in results if x.get("error")),
        "rewrite": _rewrite_counts(results),
        "followups": followup_stats(results),
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
        # A build on this computer holds Ollama and the CPU: questions would
        # wait behind its batches and every timing would be meaningless.
        from rag_studio import ingest
        building = ingest.active_build()
        if building and building.get("on", "local") == "local":
            raise RunBusy(f"An index is being built on this computer ({building.get('profile', 'unknown')}). "
                          "Wait for it to finish, then start the run.")
        client = client or OllamaClient()
        r = active_retriever(spec.profile)
        if spec.search_spec.method != "dense" and not r.has_word_index():
            raise RuntimeError(f"The {r.profile.label} index has no word index yet, so BM25 and hybrid "
                               f"search are unavailable. Add it with ./rag ingest --profile {spec.profile}.")
        questions = [(sid, q) for sid in spec.sets for q in all_sets[sid]["questions"]]
        run = {
            "id": run_id or new_run_id(), "status": "running", "started_at": _now(),
            "finished_at": None, "label": spec.label, "ladder": spec.ladder,
            "spec": asdict(spec), "config": _config(spec, r, client) | _followup_config(spec, questions),
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

        def save(res_: dict[str, Any]) -> None:
            run["summary"] = summarize(run["results"], spec.sets)
            write_json_atomic(path, run)
            if on_result:
                on_result(run, res_)

        cancelled = False
        try:
            scorer = (reranking.scorer_for(reranking.get(spec.search_spec.rerank), cache=spec.rerank_cache)
                      if spec.search_spec.rerank else None)
            if spec.two_pass:
                # Pass 1: search every question. Pass 2: answer them, with the
                # search models released first. The answer model of the run
                # before stays resident for Ollama's 5-minute keep-alive (about
                # 5 GB here) and left too little memory for EmbeddingGemma 2 in
                # back-to-back runs (2026-10-22): free it first. Only our own
                # answer model is unloaded, never another app's.
                if spec.mode == "answer" and hasattr(client, "unload"):
                    try:
                        client.unload(spec.model or role_model("answer"))
                    except (OllamaUnavailable, OllamaError):
                        pass
                pending: list[tuple[dict[str, Any], dict[str, Any], retrieve.Found]] = []
                for sid, q in questions:
                    if cancel and cancel():
                        cancelled = True
                        break
                    res, q_found = _search_one(sid, q, spec, r, scorer)
                    run["results"].append(res)
                    if not (spec.mode == "answer" and all_sets[sid]["scored"]):
                        run["progress"]["done"] += 1
                    else:
                        pending.append((res, q, q_found))
                    save(res)
                if not cancelled and pending:
                    release_search_models()
                    for res, q, q_found in pending:
                        if cancel and cancel():
                            cancelled = True
                            break
                        _answer_one(res, q, spec, r, client, q_found)
                        run["progress"]["done"] += 1
                        save(res)
            else:
                for sid, q in questions:
                    if cancel and cancel():
                        cancelled = True
                        break
                    res, q_found = _search_one(sid, q, spec, r, scorer)
                    if spec.mode == "answer" and all_sets[sid]["scored"]:
                        _answer_one(res, q, spec, r, client, q_found)
                    run["results"].append(res)
                    run["progress"]["done"] += 1
                    save(res)
        except (RewriteUnavailable, RewriteBadOutput, RerankError) as e:
            # Remote AI went away, or returned junk, mid-run: the run FAILS, with the questions it finished,
            # rather than finishing as a run full of errors that looks like a result.
            run.update(status="failed", error=str(e), finished_at=_now(),
                       seconds=round(time.perf_counter() - t_run, 1))
            run["summary"] = summarize(run["results"], spec.sets)
            write_json_atomic(path, run)
            raise
        run["status"] = "cancelled" if cancelled else "done"
        run["finished_at"] = _now()
        run["seconds"] = round(time.perf_counter() - t_run, 1)
        run["summary"] = summarize(run["results"], spec.sets)
        write_json_atomic(path, run)
        return run
    finally:
        lock.release()


def _followup_config(spec: RunSpec, questions: list[tuple[str, dict[str, Any]]]) -> dict[str, Any]:
    """How follow-up questions (Set E) were handled, recorded only when the run has any. Not part
    of `retrieval`, so a run with Set E still sits in the same ladder variant as one without."""
    if not any(q.get("history") for _, q in questions):
        return {}
    if spec.as_typed:
        return {"followup": {"as_typed": True}}
    from rag_studio import rewrite
    return {"followup": rewrite.describe("standalone")}


def release_search_models() -> None:
    """Give back the memory the search models hold (EmbeddingGemma 2 and the
    reranker), so the answer model has room."""
    from rag_studio.embed import STModel
    reranking.LocalReranker.release()
    STModel.release()


def _search_one(sid: str, q: dict[str, Any], spec: RunSpec, r: Retriever,
                scorer: Any = None) -> tuple[dict[str, Any], retrieve.Found]:
    """Search one question once, 50 deep, and score where the answer pages
    ranked. The answer (if any) is written from this same retrieval."""
    found = None
    for kind, payload in retrieve.run(q["question"], retriever=r, k=spec.k, spec=spec.search_spec, deep=DEEP_K,
                                      scorer=scorer, history=q.get("history"), as_typed=spec.as_typed):
        if kind == "found":
            found = payload
    assert found is not None
    t = found.timings
    hits = found.hits
    top = hits[:spec.k]
    where = targets(q)
    res: dict[str, Any] = {
        "set": sid, "id": q["id"], "category": q.get("category"), "question": q["question"],
        "targets": where, "expect": q.get("expect"),
        "expect_labels": ([fact_label(f) for f in q["expect"]] if isinstance(q.get("expect"), list)
                          else q.get("expect")),
        "hits": [{k: h.get(k) for k in ("rank", "chunk_id", "doc_id", "page", "score", "ranks", "capped",
                                        "rerank_score")}
                 for h in top],
        "hit_rank": first_answer_rank(top, where),
        "best_rank": first_answer_rank(hits, where),
        "answer": None, "outcome": None, "found": [], "missing": [], "refused": None,
        "citations": None, "usage": None, "error": None,
        "timings": {"embed_ms": round(t["embed_ms"], 1) if "embed_ms" in t else None,
                    "search_ms": round(t.get("search_ms", 0) + t.get("bm25_ms", 0) + t.get("fuse_ms", 0), 1),
                    "generate_ms": None},
    }
    if found.before_rerank is not None:
        res["timings"]["rerank_ms"] = round(t["rerank_ms"], 1)
        step = next(s for s in found.steps if s.id == "rerank")
        res["rerank"] = {"before_rank": first_answer_rank(found.before_rerank, where),
                         "after_rank": res["best_rank"], "cache": step.detail.get("cache"),
                         "compute_ms": step.detail.get("compute_ms")}
    for key in ("rewrites", "rounds", "agent", "network", "standalone_question", "cache"):
        if getattr(found, key) is not None:
            res[key] = getattr(found, key)
    if q.get("history"):
        res["history"] = q["history"]
        res["rewrite_ok"] = rewrite_ok(q, found.standalone_question)
    return res, found


def _answer_one(res: dict[str, Any], q: dict[str, Any], spec: RunSpec, r: Retriever, client: OllamaClient,
                found: retrieve.Found) -> None:
    done = None
    for event, data in ask(q["question"], k=spec.k, model=spec.model, profile_id=spec.profile,
                           prompt_version=spec.prompt_version, search=spec.search_spec,
                           retriever=r, client=client, found=found, history=q.get("history"),
                           as_typed=spec.as_typed):
        if event in ("done", "error"):
            done = (event, data)
    assert done is not None
    event, data = done
    if event == "error":
        res["error"] = data["error"]
        return
    res["answer"] = data["answer"]
    res["citations"] = data["citations"]
    res["usage"] = data["usage"]
    # The exact messages, as a hash and a size: a replay (How it works 9) proves it rebuilt them byte for byte.
    res["prompt"] = {"chars": data["prompt"]["chars"], "messages_sha256": messages_sha256(data["prompt"]["messages"])}
    res["timings"]["generate_ms"] = next((s["ms"] for s in data["trace"] if s["id"] == "generate"), None)
    res.update(score_answer(q.get("expect"), data["answer"]))


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
    {"row": 5, "change": "A newer embedding model: Qwen3-Embedding 0.6B, EmbeddingGemma 2, Qwen3-Embedding 8B (Colab)", "milestone": "M4"},
    {"row": 6, "change": "Reranking (Qwen3-Reranker)", "milestone": "M5"},
    {"row": 7, "change": "Contextual retrieval", "milestone": "M5"},
    {"row": 8, "change": "Query rewriting", "milestone": "M5"},
    {"row": 9, "change": "The agentic loop", "milestone": "M5"},
    {"row": 10, "change": "A new answer model", "milestone": "M6"},
    {"row": 11, "change": "Figure paths", "milestone": "M7"},
]


def retrieval_key(cfg: dict[str, Any]) -> str:
    """The retrieval setup as canonical JSON, minus the word index's own facts
    (they change with every rebuild of the same setup). Every switch a run
    records under `retrieval` is part of it, so a new switch separates runs
    without anyone editing this function."""
    retrieval = {k: v for k, v in (cfg.get("retrieval") or {}).items() if k != "word_index"}
    return json.dumps(retrieval or {"method": "dense"}, sort_keys=True, separators=(",", ":"))


def _legacy_key(cfg: dict[str, Any]) -> tuple:
    """The key until M4: the method and the section limit were the only
    retrieval switches. Kept so a test can prove the new key groups every
    earlier run exactly as this one did."""
    index = cfg.get("index") or {}
    answer = cfg.get("answer") or {}
    retrieval = cfg.get("retrieval") or {}
    return (index.get("table"), (index.get("embedding") or {}).get("digest"), cfg.get("k"),
            answer.get("model"), answer.get("digest"), answer.get("prompt_sha256"),
            retrieval.get("method", "dense"), retrieval.get("section_cap"))


def config_key(cfg: dict[str, Any]) -> tuple:
    """What must match for two runs to sit in one ladder row."""
    return _legacy_key(cfg)[:-2] + (retrieval_key(cfg),)


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


def _variant_ranking(tagged: list[dict[str, Any]], per_set: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """MRR and recall@20 over the runs each set's numbers came from."""
    rows = [x for r in tagged for x in r["results"] if per_set.get(x["set"], {}).get("run_id") == r["id"]]
    return ranking_stats(rows)


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
    return {"label": label, "sets": per_set, "ranking": _variant_ranking(tagged, per_set),
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
    return {"a": run_head(a), "b": run_head(b), "config": config, "rows": rows,
            "summary": {"a": summarize([r for r in a["results"] if r["set"] in sets], sets),
                        "b": summarize([r for r in b["results"] if r["set"] in sets], sets)},
            "changed": sum(1 for r in rows if r["outcome_changed"] or r["retrieval_changed"])}


def _brief_row(x: dict[str, Any]) -> dict[str, Any]:
    return {k: x.get(k) for k in ("outcome", "hit_rank", "best_rank", "answer", "found", "missing",
                                  "refused", "error")}


def run_head(r: dict[str, Any]) -> dict[str, Any]:
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
        for r in run.get("results", []):
            _upgrade_row(r)
        _upgrade_summary(run)
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
        _upgrade_summary(run)
    return run


def _upgrade_summary(run: dict[str, Any]) -> None:
    """Runs saved before M5 have no MRR or recall@20; they follow from the
    per-question `best_rank` every run has always recorded."""
    summary = run.get("summary")
    if summary is not None and "ranking" not in summary:
        for r in run.get("results", []):
            r.setdefault("targets", [])
        summary["ranking"] = ranking_stats(run.get("results", []))
    if summary is not None and "followups" not in summary:
        summary["followups"] = followup_stats(run.get("results", []))


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
    for key in ("rewrites", "rounds", "agent", "network", "rerank", "standalone_question", "cache"):
        r.setdefault(key, None)


def _rerank_label(retrieval: dict[str, Any]) -> str | None:
    rr = retrieval.get("rerank")
    if not rr:
        return None
    return f"{rr['id']} (top {rr['depth']})" if isinstance(rr, dict) else str(rr)


def _technique(retrieval: dict[str, Any]) -> str | None:
    """decompose, multi or hyde: a run records the full description, a search spec the bare name."""
    rw = retrieval.get("rewrite")
    return rw.get("technique") if isinstance(rw, dict) else rw


def _agent_rounds(retrieval: dict[str, Any]) -> int | None:
    """The multi-step search's round limit: a run records it under `agent`, a search spec as `agent_rounds`."""
    agent = retrieval.get("agent")
    return agent.get("rounds") if isinstance(agent, dict) else retrieval.get("agent_rounds")


def _rerank_name(retrieval: dict[str, Any]) -> str | None:
    """The reranker as the pickers name it ("Remote AI · 4B"), or its id if the app no longer knows it."""
    rr = retrieval.get("rerank")
    if not rr:
        return None
    rid = rr.get("id") if isinstance(rr, dict) else str(rr)
    spec = reranking.RERANKERS.get(rid)
    return spec.label if spec else rid


def _brief(cfg: dict[str, Any]) -> dict[str, Any]:
    """The setup in a few display fields (run history, the ladder, a comparison). Never used to group runs."""
    index = cfg.get("index") or {}
    answer = cfg.get("answer") or {}
    retrieval = cfg.get("retrieval") or {}
    rr = retrieval.get("rerank")
    followup = cfg.get("followup")
    return {"mode": cfg.get("mode"), "k": cfg.get("k"), "profile": index.get("profile"),
            "index_label": index.get("label"), "table": index.get("table"),
            "parser": (index.get("config") or {}).get("parser"),
            "cleaner": (index.get("config") or {}).get("cleaner"),
            "embedding_model": (index.get("embedding") or {}).get("model"),
            "retrieval": (cfg.get("retrieval") or {}).get("method"),
            "section_cap": (cfg.get("retrieval") or {}).get("section_cap"),
            "rerank": _rerank_label(retrieval),
            "rerank_name": _rerank_name(retrieval),
            "rerank_depth": (rr.get("depth") if isinstance(rr, dict) else retrieval.get("rerank_depth")) if rr else None,
            "rewrite": _technique(retrieval),
            "agent_rounds": _agent_rounds(retrieval),
            "followup": None if not followup else "as typed" if followup.get("as_typed") else "rewritten",
            "contextual": bool((index.get("config") or {}).get("contextual")),
            "answer_model": answer.get("model"), "prompt_version": answer.get("prompt_version")}

