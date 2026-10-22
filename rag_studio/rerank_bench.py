"""`./rag rerank bench`: which dtype and batch size should the local reranker use?

Measured on real (question, passage) pairs from the default index, with
EmbeddingGemma 2 already loaded in the same process (as in the app, where both
live in one process; research §6, spec §17 M5). Each dtype runs in its own
subprocess so peak memory is its own, and the parent compares its scores with
float32's: the top-5 overlap and the Spearman rank correlation per question.

Run it on an idle computer: the timings are the point.
"""
from __future__ import annotations

import json
import resource
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

BENCH_QUESTIONS = ("set-b:B4", "set-a:A2", "set-a:A5", "set-a:A1", "set-b:B1", "set-d:D3")
DTYPES = ("float32", "bfloat16", "int8")
CONFIGS = ((20, 4), (20, 8), (10, 4), (10, 8))      # (depth, batch size)


def _candidates(profile_id: str) -> list[dict[str, Any]]:
    """The dense top 20 for each bench question, on the default index."""
    from rag_studio import eval as ev
    from rag_studio.retrieve import retriever
    r = retriever(profile_id)
    sets = ev.load_sets()
    out = []
    for key in BENCH_QUESTIONS:
        sid, qid = key.split(":")
        q = next(x for x in sets[sid]["questions"] if x["id"] == qid)
        hits, _ = r.dense(q["question"], 20)
        out.append({"key": key, "question": q["question"],
                    "documents": [h["search_text"] for h in hits], "chunk_ids": [h["chunk_id"] for h in hits]})
    return out


def _rss_mb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024


def _load(dtype: str):
    """The reranker in the requested dtype, with EmbeddingGemma 2 loaded first."""
    import torch
    from rag_studio import embed, profiles, rerank
    from rag_studio.embed import STModel
    emb = profiles.get(profiles.DEFAULT).embedding
    STModel.get(emb).embed(["warm up"])
    rss_embedder = _rss_mb()
    spec = rerank.RERANKERS["qwen3-0.6b"]
    folder = rerank.snapshot(spec)
    from sentence_transformers import CrossEncoder
    t = time.perf_counter()
    ce = CrossEncoder(str(folder), device="cpu", local_files_only=True, max_length=spec.max_length,
                      prompts={"rag": rerank.INSTRUCTIONS[rerank.INSTRUCTION_VERSION]}, default_prompt_name="rag",
                      model_kwargs={"dtype": torch.float32 if dtype == "int8" else getattr(torch, dtype)})
    ce.eval()
    if dtype == "int8":
        # Dynamic int8 on the Linear layers (weights int8, activations quantised on the fly).
        ce[0].auto_model = torch.ao.quantization.quantize_dynamic(ce[0].auto_model, {torch.nn.Linear}, dtype=torch.qint8)
    return ce, {"load_s": round(time.perf_counter() - t, 1), "rss_after_embedder_mb": round(rss_embedder),
                "embedder": embed.model_name(emb)}


def run_one(dtype: str, cand_path: str, out_path: str) -> None:
    """The subprocess: time every config on every question and save the scores."""
    import torch
    cands = json.loads(Path(cand_path).read_text())
    ce, info = _load(dtype)
    result: dict[str, Any] = {"dtype": dtype, "threads": torch.get_num_threads(), **info, "configs": {}, "scores": {}}
    for depth, batch in CONFIGS:
        times = []
        for c in cands:
            docs = c["documents"][:depth]
            t = time.perf_counter()
            with torch.inference_mode():
                s = ce.predict([(c["question"], d) for d in docs], batch_size=batch, show_progress_bar=False)
            times.append(time.perf_counter() - t)
            if (depth, batch) == (20, 4):
                result["scores"][c["key"]] = [float(x) for x in s]
        result["configs"][f"depth{depth}-batch{batch}"] = {
            "median_s": round(statistics.median(times), 1), "max_s": round(max(times), 1),
            "per_question_s": [round(x, 1) for x in times]}
        result["peak_rss_mb"] = round(_rss_mb())
    Path(out_path).write_text(json.dumps(result))


def _spearman(a: list[float], b: list[float]) -> float:
    def ranks(v: list[float]) -> list[float]:
        order = sorted(range(len(v)), key=lambda i: -v[i])
        r = [0.0] * len(v)
        for pos, i in enumerate(order):
            r[i] = pos
        return r
    ra, rb = ranks(a), ranks(b)
    n = len(a)
    return 1 - 6 * sum((x - y) ** 2 for x, y in zip(ra, rb)) / (n * (n * n - 1))


def bench(profile_id: str | None = None, say=print) -> dict[str, Any]:
    from rag_studio import profiles
    profile_id = profile_id or profiles.DEFAULT
    tmp = Path(tempfile.mkdtemp(prefix="rerank-bench-"))
    cand_path = tmp / "candidates.json"
    # Candidates come from their own process, so the parent never holds a model.
    code = ("import json,sys; sys.path.insert(0,'.'); from rag_studio import rerank_bench as b; "
            f"open({str(cand_path)!r},'w').write(json.dumps(b._candidates({profile_id!r})))")
    subprocess.run([sys.executable, "-I", "-c", code], check=True)
    cands = json.loads(cand_path.read_text())
    say(f"{len(cands)} questions x 20 candidates from {profile_id}")
    runs: dict[str, Any] = {}
    for dtype in DTYPES:
        out = tmp / f"{dtype}.json"
        say(f"running {dtype} ...")
        p = subprocess.run([sys.executable, "-I", "-m", "rag_studio.rerank_bench", "run", dtype, str(cand_path), str(out)],
                           capture_output=True, text=True)
        if p.returncode != 0 or not out.exists():
            runs[dtype] = {"error": (p.stderr or "failed").strip().splitlines()[-1][:300]}
            say(f"  {dtype} failed: {runs[dtype]['error']}")
            continue
        runs[dtype] = json.loads(out.read_text())
        say(f"  {dtype}: depth20/batch4 median {runs[dtype]['configs']['depth20-batch4']['median_s']} s, "
            f"peak {runs[dtype]['peak_rss_mb']} MB")
    ref = runs.get("float32", {}).get("scores")
    for dtype, r in runs.items():
        if "scores" not in r or not ref:
            continue
        overlaps, rhos = [], []
        for key, s in r["scores"].items():
            top = lambda v: set(sorted(range(len(v)), key=lambda i: -v[i])[:5])   # noqa: E731
            overlaps.append(len(top(s) & top(ref[key])) / 5)
            rhos.append(_spearman(s, ref[key]))
        r["agreement_with_float32"] = {"top5_overlap": round(statistics.mean(overlaps), 3),
                                       "spearman": round(statistics.mean(rhos), 3)}
    result = {"profile": profile_id, "questions": [c["key"] for c in cands], "runs": runs}
    (tmp / "result.json").write_text(json.dumps(result, indent=1))
    say(f"saved {tmp / 'result.json'}")
    return result


if __name__ == "__main__":
    if sys.argv[1] == "run":
        run_one(sys.argv[2], sys.argv[3], sys.argv[4])
