"""Command-line views of the engine, behind `./rag`.

More technical than the browser on purpose: endpoints, tags and paths are what
a terminal user needs. Lines follow docs/product-language.md (CLI section).
"""
from __future__ import annotations

import argparse
import json
import sys

from rag_studio import health, profiles

MARK = {"healthy": "ok", "running": "ok", "info": "--", "warning": "!!",
        "needs_attention": "XX", "unavailable": "XX"}


def cmd_health(args: argparse.Namespace) -> int:
    report = health.report()
    if args.json:
        print(json.dumps(report, indent=2))
        return 0
    for section in report["sections"]:
        print(f"\n{section['name']}")
        for c in section["checks"]:
            print(f"  [{MARK[c['status']]}] {c['label']}: {c['message']}")
            if c["fix"]:
                print(f"       Run: {c['fix']}")
    snap = report["ollama"]
    if snap["up"]:
        print(f"\nOllama {snap['version']} at {snap['host']}")
        for m in snap["installed"]:
            caps = ",".join(m["capabilities"])
            print(f"  {m['name']:<28} {m['size'] / 1e9:5.1f} GB  {caps}")
    print(f"\nOverall: {report['overall'].replace('_', ' ')}")
    return 0 if report["overall"] in ("healthy", "warning") else 1


def cmd_ingest(args: argparse.Namespace) -> int:
    from rag_studio import ingest

    profile = profiles.get(args.profile)
    last = {"pct": -1}

    def progress(stage: str, info: dict) -> None:
        if stage == "clean":
            print(f"Cleaning {info['doc_id']} ({info['parser']})", flush=True)
        elif stage == "docling_start":
            print(f"  Docling is converting {info['doc_id']}: about 3 s per page on this CPU", flush=True)
        elif stage == "docling_done":
            print(f"  Docling converted {info['pages']} pages ({info['seconds']:.0f} s)", flush=True)
        elif stage == "parse":
            print(f"Parsed {info['documents']} documents, {info['pages']} pages ({info['seconds']:.1f} s)")
        elif stage == "sentences":
            print(f"Semantic chunking: {info['sentences']} sentences to compare"
                  f" ({info['cached']} already embedded)", flush=True)
        elif stage == "chunk":
            print(f"Cut {info['passages']} passages ({info['seconds']:.1f} s)")
        elif stage == "embed_start":
            print(f"Embedding {info['to_embed']} passages with {info['model']}"
                  f" ({info['reused']} unchanged, reused)")
        elif stage == "embed":
            pct = 100 * info["done"] // max(info["total"], 1)
            if pct // 10 != last["pct"] // 10 or info["done"] == info["total"]:
                print(f"  {info['done']}/{info['total']}", flush=True)
                last["pct"] = pct
        elif stage == "word_index":
            print(f"Word index (BM25) on {info['column']} ({info['seconds']:.1f} s)")
        elif stage == "up_to_date":
            print(f"The index is up to date ({info['table']}). Nothing to do.")
        elif stage == "done":
            print(f"Built {info['table']}: {info['rows']} passages ({info['seconds']:.0f} s)")
            if info["dropped"]:
                print(f"Removed old builds: {', '.join(info['dropped'])}")

    print(f"Index profile: {profile.id} ({profile.label})")
    try:
        result = ingest.build(profile, force=args.force, progress=progress)
    except Exception as e:  # noqa: BLE001 - a terminal user needs the reason, not a traceback
        print(f"Error: {e}", file=sys.stderr)
        return 1
    m = result.manifest
    for d in m["documents"]:
        print(f"  {d['doc_id']:<28} {d['pages']:>4} pages  {d['passages']:>5} passages")
    return 0


def _hit(res: dict) -> str:
    if not res["targets"]:
        return "-"
    if res["hit_rank"]:
        return f"Hit@{res['hit_rank']}"
    return f"Miss ({'rank ' + str(res['best_rank']) if res['best_rank'] else 'not in top 50'})"


def cmd_eval(args: argparse.Namespace) -> int:
    from rag_studio import eval as ev

    if args.tag:
        if not args.ladder:
            print("Error: --tag needs --ladder ROW.", file=sys.stderr)
            return 1
        try:
            run = ev.tag_ladder(args.tag, args.ladder, args.label)
        except ValueError as e:
            print(f"Error: {e}", file=sys.stderr)
            return 1
        print(f"Run {run['id']} is now ladder row {args.ladder}: {run['ladder']['label']}")
        return 0

    sets = ev.load_sets()
    chosen = [f"set-{s}" if not s.startswith("set-") else s for s in args.set] or list(sets)
    ladder = {"row": args.ladder, "label": args.label or f"Row {args.ladder}"} if args.ladder else None
    spec = ev.RunSpec(sets=chosen, mode=args.mode, k=args.k, model=args.model,
                      profile=args.profile, prompt_version=args.prompt, label=args.label, ladder=ladder,
                      search={"method": args.search, "section_cap": args.cap})

    def on_result(run: dict, res: dict) -> None:
        done, total = run["progress"]["done"], run["progress"]["total"]
        secs = (res["timings"]["generate_ms"] or 0) / 1000
        outcome = res["error"] and "ERROR" or res["outcome"] or "-"
        facts = f"{res['found']} / {res['expect_labels']}" if isinstance(res["expect"], list) else ""
        print(f"[{done:>2}/{total}] {res['id']:<3} {outcome:<7} {_hit(res):<18} "
              f"{secs:5.0f}s  {res['question'][:60]}", flush=True)
        if args.mode == "answer" and res["answer"] is not None:
            print(f"        {facts}  {res['answer'].strip()[:300]!r}")
        if res["error"]:
            print(f"        error: {res['error']}")

    print(f"Run: {', '.join(chosen)} · {args.profile} · {spec.search_spec.label} · {args.mode} · top {args.k}"
          f" · prompt {args.prompt}" + (f" · {args.model}" if args.model else ""))
    try:
        run = ev.execute(spec, on_result=on_result)
    except Exception as e:  # noqa: BLE001
        print(f"Error: {e}", file=sys.stderr)
        return 1
    s = run["summary"]
    print()
    for sid, r in s["recall"].items():
        if r["total"]:
            print(f"{sid}: recall@{args.k} {r['hits']}/{r['total']}")
    if s["outcomes"]:
        o = s["outcomes"]
        model = run["config"]["answer"]["model"]
        print(f"PASS {o['PASS']} | REFUSED {o['REFUSED']} | WRONG {o['WRONG']}  ({model})")
    print(f"Saved run {run['id']} ({run['seconds']:.0f} s)")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="./rag")
    sub = parser.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("health", help="check Ollama, models, documents, indexes and this computer")
    p.add_argument("--json", action="store_true", help="print the full report as JSON")
    p.set_defaults(func=cmd_health)

    p = sub.add_parser("ingest", help="parse, chunk and embed the PDFs into an index")
    p.add_argument("--profile", default=profiles.DEFAULT, choices=sorted(profiles.PROFILES),
                   help="which index profile to build (default: %(default)s)")
    p.add_argument("--force", action="store_true", help="rebuild even if nothing changed")
    p.set_defaults(func=cmd_ingest)

    p = sub.add_parser("eval", help="ask the test questions and score the answers")
    p.add_argument("--set", nargs="*", default=[], metavar="ID",
                   help="test-question sets, e.g. a b (default: all)")
    p.add_argument("--mode", choices=("answer", "retrieval"), default="answer",
                   help="answer: the full scorecard; retrieval: search only, no model (default: %(default)s)")
    p.add_argument("--model", help="answer model (default: the Answer role's model)")
    p.add_argument("--k", type=int, default=5, help="passages per answer (default: %(default)s)")
    p.add_argument("--prompt", default="v2", help="system prompt version (default: %(default)s)")
    p.add_argument("--profile", default=profiles.DEFAULT, choices=sorted(profiles.PROFILES),
                   help="which index to search (default: %(default)s)")
    p.add_argument("--search", choices=("dense", "bm25", "hybrid"), default="dense",
                   help="dense (by meaning), bm25 (by words) or hybrid (both, fused with RRF) (default: %(default)s)")
    p.add_argument("--cap", type=int, metavar="N",
                   help="at most N passages per section ahead of the rest; contents pages and "
                        "reference lists at most 1 (default: no cap)")
    p.add_argument("--label", help="a name for this run (with --ladder: the row's variant name)")
    p.add_argument("--ladder", type=int, metavar="ROW", help="record this run as a ladder row")
    p.add_argument("--tag", metavar="RUN_ID", help="with --ladder: record a finished run as a ladder row")
    p.set_defaults(func=cmd_eval)
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
