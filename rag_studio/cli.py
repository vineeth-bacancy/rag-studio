"""Command-line views of the engine, behind `./rag`.

More technical than the browser on purpose: endpoints, tags and paths are what
a terminal user needs. Lines follow docs/product-language.md (CLI section).
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys

from rag_studio import health, profiles
from rag_studio import rerank as reranking
from rag_studio.prompt import DEFAULT_VERSION
from rag_studio.retrieve import SearchSpec

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
    on = getattr(args, "on", "local")

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
            where = " on the Colab runtime" if info.get("on") == "colab" else ""
            print(f"Embedding {info['to_embed']} passages with {info['model']}{where}"
                  f" ({info['reused']} unchanged since the last build; {info.get('cached', 0)} already"
                  " embedded earlier, by this or an interrupted build)")
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
        result = ingest.build(profile, force=args.force, progress=progress, on=on)
    except KeyboardInterrupt:
        print("\nStopped. The vectors made so far are kept: building again resumes from them.",
              file=sys.stderr)
        return 130
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
    # No --set means every set that runs on this computer: a remote-only set (E: follow-ups) needs
    # Remote AI to rewrite its questions, so it is named on purpose.
    chosen = [f"set-{s}" if not s.startswith("set-") else s for s in args.set] or [
        sid for sid, s in sets.items() if not s.get("remote_only")]
    ladder = {"row": args.ladder, "label": args.label or f"Row {args.ladder}"} if args.ladder else None
    try:
        search = SearchSpec(args.search, args.cap, rerank=args.rerank, rerank_depth=args.rerank_depth,
                            rewrite=args.rewrite, agent_rounds=args.agent)
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    spec = ev.RunSpec(sets=chosen, mode=args.mode, k=args.k, model=args.model,
                      profile=args.profile, prompt_version=args.prompt, label=args.label, ladder=ladder,
                      search=search.as_args(), rerank_cache=not args.fresh_rerank, as_typed=args.as_typed)

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
    if s.get("rewrite"):
        print(f"Follow-ups rewritten right: {s['rewrite']['ok']}/{s['rewrite']['total']}")
    rk = s["ranking"]
    if rk["n"]:
        print(f"MRR {rk['mrr']:.3f} | recall@20 {rk['recall20']}/{rk['n']}")
    if s["outcomes"]:
        o = s["outcomes"]
        model = run["config"]["answer"]["model"]
        print(f"PASS {o['PASS']} | REFUSED {o['REFUSED']} | WRONG {o['WRONG']}  ({model})")
    print(f"Saved run {run['id']} ({run['seconds']:.0f} s)")
    return 0


# --------------------------------------------------------------------------- ./rag llm

def _clipboard_read() -> str:
    for cmd in (["xclip", "-o", "-selection", "clipboard"], ["xsel", "--clipboard", "--output"],
                ["wl-paste", "--no-newline"]):
        try:
            out = subprocess.run(cmd, capture_output=True, text=True, timeout=5)
        except (FileNotFoundError, subprocess.TimeoutExpired):
            continue
        if out.returncode == 0:
            return out.stdout.strip()
    return ""


def _clipboard_write(text: str) -> bool:
    for cmd in (["xclip", "-selection", "clipboard"], ["xsel", "--clipboard", "--input"], ["wl-copy"]):
        try:
            out = subprocess.run(cmd, input=text, text=True, timeout=5)
        except (FileNotFoundError, subprocess.TimeoutExpired):
            continue
        if out.returncode == 0:
            return True
    return False


def _print_remote(info: dict | None) -> None:
    if not info:
        return
    if not info.get("reachable"):
        print(f"  Not reachable: {info.get('error')}")
        return
    if info.get("ollama_version"):
        print(f"  Ollama {info['ollama_version']} · {info['latency_ms']} ms")
    else:
        print("  Ollama not running here (a rerank-only runtime)")
    gpus = (info.get("gpu") or {}).get("gpus") or []
    for g in gpus:
        print(f"  GPU    {g['name']} · {g['memory_used_mb']:,} / {g['memory_total_mb']:,} MB in use")
    if not gpus:
        print(f"  GPU    none ({(info.get('gpu') or {}).get('reason', 'not reported')})")
    for m in info.get("models", []):
        loaded = next((x for x in info.get("loaded", []) if x["name"] == m["name"]), None)
        where = ("on the GPU" if loaded["on_gpu"] else "on the CPU") if loaded else "not loaded"
        print(f"  Model  {m['name']:<26} {(m['digest'] or '-')[:12]}  {m.get('quantization') or '-':<7} {where}")
    w = info.get("worker")
    if w:
        peak = (w.get("vram") or {}).get("peak_mb")
        print(f"  Worker serves {', '.join(w.get('served') or []) or 'nothing'} · loaded {w.get('loaded') or '-'}"
              + (f" · peak {peak:,} MB" if peak else ""))
    if info.get("missing"):
        print(f"  Missing for the Colab indexes: {', '.join(info['missing'])}")


def cmd_llm(args: argparse.Namespace) -> int:
    from rag_studio import colab, remote
    from rag_studio.llm import OllamaClient, OllamaUnavailable

    if args.action == "colab":
        token = None
        if not args.no_token:
            token = None if args.new_token else remote.remembered_token()
            if token is None:
                token = remote.mint_token()
                remote.remember_token(token)
        try:
            version = OllamaClient().version()
        except OllamaUnavailable:
            version = None
        try:
            # --worker alone is a rerank-only runtime: no Ollama models, so no Ollama.
            models = args.models if args.models is not None else (() if args.worker else colab.DEFAULT_MODELS)
            cell = colab.build_cell(models=models, token=token, ollama_version=version,
                                    worker=args.worker or ())
        except (ValueError, colab.MissingPayload) as e:
            print(f"Error: {e}", file=sys.stderr)
            return 1
        stats = colab.cell_stats(cell)
        if args.copy:
            if not _clipboard_write(cell):
                print("Error: no clipboard tool found (xclip, xsel or wl-copy). Run without --copy.",
                      file=sys.stderr)
                return 1
            print(f"Copied the setup cell ({stats['kilobytes']} KB, {stats['lines']} lines) to the clipboard.")
        else:
            print(cell)
        note = ("This cell carries a token this computer remembers: after Colab reclaims the runtime, "
                "re-run the cell and connect with the new URL alone." if token else
                "This cell carries no token: the runtime makes its own and shows it inside a connect code.")
        print(f"\nIn Colab: Runtime > Change runtime type > T4 GPU, paste into one cell, run it.\n{note}\n"
              f"Then: ./rag llm connect   (reads the clipboard)", file=sys.stderr)
        return 0

    if args.action == "connect":
        target = (args.target or "").strip()
        source = "argument"
        if not target and not args.no_paste:
            target, source = _clipboard_read(), "clipboard"
        if not target:
            print("Error: nothing to connect to. Paste the connect code (or the URL) the Colab cell "
                  "showed: ./rag llm connect <code>", file=sys.stderr)
            return 1
        is_code = remote.looks_like_code(target)
        print(f"Connecting with the {'connect code' if is_code else 'URL'} from the {source}...")
        try:
            result = remote.connect(code=target if is_code else "", url="" if is_code else target,
                                    token=args.token or "")
        except ValueError as e:
            print(f"Error: {e}", file=sys.stderr)
            return 1
        conn = remote.connection()
        print(f"Saved: {conn['host']} (token kept in {conn['state_path']}, mode 600)")
        if result["ok"]:
            for r in result["embedded"]:
                print(f"  {r['model']:<26} embedded a test sentence: {r['dimensions']} dimensions, {r['seconds']} s")
            for r in result.get("chatted") or []:
                print(f"  {r['model']:<26} answered a one-token chat, {r['seconds']} s")
            if result.get("worker"):
                print(f"  worker                     serves {', '.join(result['worker']['served'] or [])}")
            _print_remote(result.get("remote"))
            if result["missing"]:
                print(f"Warning: the runtime does not serve {', '.join(result['missing'])}, which the Colab "
                      "indexes need. Run the setup cell with those models.")
            print("Remote AI is connected.")
            return 0
        print(f"The connection is saved but the test failed at the {result['stage']} step: {result['error']}",
              file=sys.stderr)
        print("Fix the runtime (or re-run the cell), then: ./rag llm test", file=sys.stderr)
        return 1

    if args.action == "keepalive":
        try:
            cell = colab.build_keepalive_cell(args.interval)
        except ValueError as e:
            print(f"Error: {e}", file=sys.stderr)
            return 1
        if args.copy:
            if not _clipboard_write(cell):
                print("Error: no clipboard tool found (xclip, xsel or wl-copy). Run without --copy.", file=sys.stderr)
                return 1
            print("Copied the keep-alive cell to the clipboard.")
        else:
            print(cell)
        print("In Colab: paste it into a NEW cell of the same notebook and run it. It keeps a cell running and "
              f"touches the GPU every {args.interval} s. Stop it with the cell's stop button; the runtime keeps running.",
              file=sys.stderr)
        return 0

    if args.action == "status":
        conn = remote.connection()
        if not conn["connected"]:
            print("Remote AI: not connected. Everything runs on this computer.")
            print(f"  Remembered token for the next cell: {'yes' if conn['token_remembered'] else 'no'}")
            print("  Connect: ./rag llm colab --copy, run it in Colab, then ./rag llm connect")
            return 0
        print(f"Remote AI: connected to {conn['host']} since {conn['connected_at']}")
        last = (f"failed ({conn['last_error_kind']}): {conn['last_error']} at {conn['last_error_at']}"
                if conn.get("last_error") else f"passed at {conn['last_ok_at']}" if conn.get("last_ok_at")
                else "never tested")
        print(f"  Last test: {last}")
        _print_remote(remote.probe())
        return 0

    if args.action == "test":
        result = remote.smoke_test()
        if result["ok"]:
            for r in result["embedded"]:
                print(f"  {r['model']:<26} {r['dimensions']} dimensions, {r['seconds']} s")
            for r in result.get("chatted") or []:
                print(f"  {r['model']:<26} one-token chat, {r['seconds']} s")
            if result.get("worker"):
                print(f"  worker                     serves {', '.join(result['worker']['served'] or [])}")
            print(f"Remote AI works ({result['seconds']} s).")
            return 0
        print(f"Error: the test failed at the {result['stage']} step: {result['error']}", file=sys.stderr)
        return 1

    if args.action == "disconnect":
        was = remote.connected()
        remote.disconnect(forget=args.forget_token)
        print("Disconnected. Everything runs on this computer." if was else "Remote AI was not connected.")
        if args.forget_token:
            print("The remembered token is gone: the next setup cell will carry a new one.")
        return 0
    return 2


# --------------------------------------------------------------------------- ./rag context

def cmd_context(args: argparse.Namespace) -> int:
    from rag_studio import context, remote
    profile = profiles.get(args.profile_id)
    if not profile.contextual:
        print(f"Error: {profile.label} has no context lines. Pick a -ctx profile.", file=sys.stderr)
        return 1
    try:
        if args.action == "status":
            s = context.progress_for(profile)
            print(f"{profile.label}: {s['written']:,} of {s['passages']:,} passages have a context line "
                  f"({s['blocks']} windows); model digest {s['digest'] or '-'}")
            job = context.last_job(profile.id)
            if job:
                print(f"  last job: {job['status']} at {job.get('finished_at') or job['started_at']}")
            return 0
        if args.action == "export":
            r = context.export(profile)
            print(f"Wrote {r['lines']:,} lines to {r['path']}")
            return 0
        if args.action == "import":
            print(f"Loaded {context.import_export(profile.id):,} lines into the cache.")
            return 0
        client = remote.client()
        info = remote.worker_info(client)
        if not info or not info.get("context"):
            print("Error: the connected runtime is not running the context job. Run the setup cell with "
                  "`./rag llm colab --worker context --copy`, then connect.", file=sys.stderr)
            return 1
        if args.action == "check":
            r = context.determinism_check(profile, client=client, n=args.n)
            print(f"{r['identical']} of {r['checked']} regenerated lines were word for word identical "
                  f"({r['exact_match_rate']:.0%}), in a shuffled order and without the runtime's memory.")
            for d in r["examples_of_difference"]:
                print(f"  {d['chunk_id']}\n    before: {d['before']}\n    after:  {d['after']}")
            return 0
        if args.action == "write":
            def say(stage: str, i: dict) -> None:
                if stage == "context":
                    print(f"\r  {i['done']:,} / {i['total']:,} lines", end="", flush=True)
            r = context.write_lines(profile, client=client, say=say, limit=args.limit)
            print()
            print(f"Wrote {r['lines']:,} lines in {r['requests']} requests, {r['seconds']} s "
                  f"({r['lines_per_s']} lines/s); {r['retried']} batches needed a second request.")
            if r["first_in_block_ms"] and r["later_in_block_ms"]:
                print(f"  first passage of a window {r['first_in_block_ms']:.0f} ms, the rest "
                      f"{r['later_in_block_ms']:.0f} ms (the window's start is reused)")
            net = r.get("network") or {}
            if net.get("calls"):
                print(f"  network: sent {net['bytes_out_raw']/1e6:.2f} MB as {net['bytes_out_wire']/1e6:.2f} MB, "
                      f"{net['network_ms']/1000:.0f} s on the tunnel, {net['server_ms']/1000:.0f} s on the runtime")
            if args.limit is None:
                e = context.export(profile)
                print(f"Exported {e['lines']:,} lines to {e['path']}")
            return 0
    except (context.ContextMissing, context.ContextBusy, context.ContextCancelled, remote.RemoteNotConnected) as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    return 2


# --------------------------------------------------------------------------- ./rag rerank

def cmd_rerank(args: argparse.Namespace) -> int:
    from rag_studio import rerank
    spec = rerank.RERANKERS["qwen3-0.6b"]
    if args.action == "download":
        if rerank.snapshot(spec):
            print(f"{spec.model} is already downloaded.")
            return 0
        print(f"Downloading {spec.model} (about 1.2 GB) ...")
        print(rerank.download(spec))
        return 0
    if args.action == "bench":
        from rag_studio import ingest, rerank_bench
        free = ingest.available_ram_gb()
        if free is not None and free < 5.5:
            print(f"Error: the benchmark holds both models (about 5 GB) and only {free:.1f} GB is free. "
                  "Close some apps and run it again.", file=sys.stderr)
            return 1
        rerank_bench.bench(args.profile)
        return 0
    return 2


# --------------------------------------------------------------------------- ./rag index

def cmd_index(args: argparse.Namespace) -> int:
    from rag_studio import embed, ingest, store

    if args.action == "list":
        for p in profiles.PROFILES.values():
            m = store.active_manifest(p.id)
            spec = p.embedding
            where = {"local": "this computer", "colab": "Colab", "either": "either"}[spec.build_on]
            if m:
                e = m["embedding"]
                state = (f"{m['rows']:>5} passages, built {m['built_at'][:16].replace('T', ' ')} "
                         f"on {'Colab' if e.get('built_on') == 'colab' else 'this computer'}")
            else:
                state = "not built"
            check = embed.last_match(p.id)
            mark = "" if not check or check.get("passed") is None else (
                "  match ok" if check["passed"] else "  MATCH FAILED")
            print(f"{p.id:<36} {embed.model_name(spec):<22} builds on {where:<13} {state}{mark}")
        return 0

    if args.action == "build":
        args.profile = args.profile_id
        return cmd_ingest(args)

    if args.action == "check":
        profile = profiles.get(args.profile_id)
        print(f"Vector match check: {profile.label}")
        try:
            r = embed.match_check(profile.id)
        except (LookupError, embed.ProfileMismatch) as e:
            print(f"Error: {e}", file=sys.stderr)
            return 1
        for side, info in r["sides"].items():
            name = "this computer" if side == "local" else "Colab runtime"
            if info["available"]:
                extra = " · ".join(str(info[k]) for k in ("ollama_version", "gpu") if info.get(k))
                print(f"  {name:<14} embedded {r['passages']} passages and {r['questions']} questions in "
                      f"{info['seconds']} s{' (' + extra + ')' if extra else ''}")
            else:
                print(f"  {name:<14} skipped: {info['reason']}")
        names = {"local": "this computer", "colab": "Colab", "stored": "the index"}
        for pr in r["pairs"]:
            what = {"noise": f"{names[pr['a']]} vs itself, another order",
                    "index": f"{names[pr['a']]} vs the index (built in batches)",
                    "cross": f"this computer vs Colab, {pr['what']}"}[pr["kind"]]
            mark = "" if pr["kind"] != "cross" else ("  [ok]" if pr["min"] >= r["threshold"] else "  [BELOW]")
            print(f"  {what:<46} n={pr['n']:<3} lowest {pr['min']:.6f}  median {pr['median']:.6f}{mark}")
        if not r["pairs"]:
            print("Nothing to compare: no machine could embed for this index right now.")
            return 1
        if r["passed"] is None:
            print(f"Result: one machine only, nothing to compare across (its own noise floor: lowest "
                  f"{r['noise_floor']:.6f}).")
            return 0
        print(f"Result: {'passed' if r['passed'] else 'FAILED'}: every cosine across machines must be at least "
              f"{r['threshold']} (noise floor on one machine: lowest {r['noise_floor']:.6f}).")
        return 0 if r["passed"] else 1

    if args.action == "download":
        spec = next((p.embedding for p in profiles.PROFILES.values()
                     if p.embedding.model == args.model and p.embedding.backend == embed.ST), None)
        if spec is None:
            names = sorted({p.embedding.model for p in profiles.PROFILES.values() if p.embedding.backend == embed.ST})
            print(f"Error: {args.model} is not a model this app downloads. Known: {', '.join(names)}. "
                  "Ollama models are installed with: ollama pull <tag>", file=sys.stderr)
            return 1
        if embed.st_snapshot(spec):
            print(f"{embed.model_name(spec)} is already downloaded (revision {spec.revision[:12]}).")
            return 0
        print(f"Downloading {embed.model_name(spec)} (revision {spec.revision[:12]}, about 1.5 GB)...")
        path = embed.st_download(spec)
        print(f"Done: {path}")
        return 0
    return 2


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="./rag")
    sub = parser.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("health", help="check Ollama, models, documents, indexes and this computer")
    p.add_argument("--json", action="store_true", help="print the full report as JSON")
    p.set_defaults(func=cmd_health)

    p = sub.add_parser("ingest", help="parse, chunk and embed the PDFs into an index")
    p.add_argument("--profile", default=profiles.BASELINE.id, choices=sorted(profiles.PROFILES),
                   help="which index profile to build (default: %(default)s, the quick one; the app's default "
                        f"index is {profiles.DEFAULT}: build it with ./rag index build {profiles.DEFAULT})")
    p.add_argument("--force", action="store_true", help="rebuild even if nothing changed")
    p.add_argument("--on", choices=("local", "colab"), default="local",
                   help="embed on this computer or on the connected Colab runtime (default: %(default)s)")
    p.set_defaults(func=cmd_ingest)

    p = sub.add_parser("llm", help="connect a Colab GPU runtime (Remote AI)")
    llm = p.add_subparsers(dest="action", required=True)
    q = llm.add_parser("colab", help="print the setup cell to paste into Colab")
    q.add_argument("--copy", action="store_true", help="copy it to the clipboard instead of printing it")
    q.add_argument("--no-token", action="store_true",
                   help="carry no token: the runtime makes its own and hands it back in a connect code")
    q.add_argument("--new-token", action="store_true", help="make a new remembered token for this cell")
    q.add_argument("--models", nargs="+", metavar="TAG",
                   help=f"models the runtime serves (default: {' '.join(('qwen3-embedding:8b', 'qwen3-embedding:0.6b'))}; "
                        "none when only --worker is given)")
    q.add_argument("--worker", nargs="+", metavar="ID",
                   help="rerankers the GPU worker downloads and serves, e.g. qwen3-0.6b-gpu qwen3-4b "
                        "(the first is loaded; the others swap in when asked)")
    q = llm.add_parser("connect", help="connect to a runtime with its connect code (or URL)")
    q.add_argument("target", nargs="?", help="the connect code or URL (default: read the clipboard)")
    q.add_argument("--token", help="the token, with a bare URL whose token this computer does not remember")
    q.add_argument("--no-paste", action="store_true", help="never read the clipboard")
    q = llm.add_parser("keepalive", help="a cell that keeps a running Colab runtime from being reclaimed for idling")
    q.add_argument("--copy", action="store_true", help="copy it to the clipboard instead of printing it")
    q.add_argument("--interval", type=int, default=240, help="seconds between GPU touches (default: %(default)s)")
    llm.add_parser("status", help="what is connected, its GPU and models; no inference")
    llm.add_parser("test", help="a real request: embed one sentence with each model")
    q = llm.add_parser("disconnect", help="back to this computer only")
    q.add_argument("--forget-token", action="store_true", help="also forget the token the last cell carried")
    p.set_defaults(func=cmd_llm)

    p = sub.add_parser("index", help="list, build and check index profiles")
    idx = p.add_subparsers(dest="action", required=True)
    idx.add_parser("list", help="every index profile: model, where it builds, state")
    q = idx.add_parser("build", help="build one profile's index")
    q.add_argument("profile_id", choices=sorted(profiles.PROFILES), metavar="PROFILE")
    q.add_argument("--on", choices=("local", "colab"), default="local",
                   help="embed on this computer or on the connected Colab runtime (default: %(default)s)")
    q.add_argument("--force", action="store_true", help="rebuild even if nothing changed")
    q = idx.add_parser("check", help="the vector match check: re-embed 50 passages on each machine")
    q.add_argument("profile_id", choices=sorted(profiles.PROFILES), metavar="PROFILE")
    q = idx.add_parser("download", help="download a model that runs in this app (EmbeddingGemma 2)")
    q.add_argument("model", help="the Hugging Face name, e.g. google/embeddinggemma-2")
    p.set_defaults(func=cmd_index)

    p = sub.add_parser("context", help="context lines: write them on Remote AI, then build the -ctx index")
    cx = p.add_subparsers(dest="action", required=True)
    for name, text in (("status", "how many passages have a line"), ("write", "write the missing lines on Remote AI"),
                       ("check", "regenerate some lines and report how many match word for word"),
                       ("export", "write the lines to data/context/<profile>.jsonl"),
                       ("import", "load an exported file back into the cache")):
        q = cx.add_parser(name, help=text)
        q.add_argument("profile_id", choices=sorted(i for i, pr in profiles.PROFILES.items() if pr.contextual),
                       metavar="PROFILE")
        if name == "check":
            q.add_argument("-n", type=int, default=20, help="lines to regenerate (default: %(default)s)")
        if name == "write":
            q.add_argument("--limit", type=int, help="write at most this many lines (a trial run)")
    p.set_defaults(func=cmd_context)

    p = sub.add_parser("rerank", help="the local reranker: download it, or measure dtype and batch size")
    rr = p.add_subparsers(dest="action", required=True)
    rr.add_parser("download", help="download Qwen3-Reranker-0.6B at its pinned revision (about 1.2 GB)")
    q = rr.add_parser("bench", help="time fp32, bf16 and int8 on real passages, beside EmbeddingGemma 2")
    q.add_argument("--profile", default=profiles.DEFAULT, choices=sorted(profiles.PROFILES))
    p.set_defaults(func=cmd_rerank)

    p = sub.add_parser("eval", help="ask the test questions and score the answers")
    p.add_argument("--set", nargs="*", default=[], metavar="ID",
                   help="test-question sets, e.g. a b (default: all)")
    p.add_argument("--mode", choices=("answer", "retrieval"), default="answer",
                   help="answer: the full scorecard; retrieval: search only, no model (default: %(default)s)")
    p.add_argument("--model", help="answer model (default: the Answer role's model)")
    p.add_argument("--k", type=int, default=5, help="passages per answer (default: %(default)s)")
    p.add_argument("--prompt", default=DEFAULT_VERSION, help="system prompt version (default: %(default)s)")
    p.add_argument("--profile", default=profiles.DEFAULT, choices=sorted(profiles.PROFILES),
                   help="which index to search (default: %(default)s)")
    p.add_argument("--search", choices=("dense", "bm25", "hybrid"), default="dense",
                   help="dense (by meaning), bm25 (by words) or hybrid (both, fused with RRF) (default: %(default)s)")
    p.add_argument("--cap", type=int, metavar="N",
                   help="at most N passages per section ahead of the rest; contents pages and "
                        "reference lists at most 1 (default: no cap)")
    p.add_argument("--rerank", choices=sorted(reranking.RERANKERS), metavar="ID",
                   help="re-order the top passages with a reranker: " + ", ".join(sorted(reranking.RERANKERS))
                        + " (default: off)")
    p.add_argument("--rewrite", choices=("decompose", "multi", "hyde"),
                   help="rewrite each question on Remote AI first: split into parts (decompose), several phrasings "
                        "(multi) or a drafted answer to search with (hyde)")
    p.add_argument("--agent", type=int, metavar="ROUNDS", help="multi-step search with at most this many rounds (2-6)")
    p.add_argument("--as-typed", action="store_true",
                   help="follow-up questions (Set E) are searched as typed, without the Remote AI rewrite")
    p.add_argument("--fresh-rerank", action="store_true",
                   help="score again instead of replaying remembered reranker scores (for timing)")
    p.add_argument("--rerank-depth", type=int, default=reranking.DEFAULT_DEPTH, metavar="N",
                   help="how many passages the reranker reads (default: %(default)s)")
    p.add_argument("--label", help="a name for this run (with --ladder: the row's variant name)")
    p.add_argument("--ladder", type=int, metavar="ROW", help="record this run as a ladder row")
    p.add_argument("--tag", metavar="RUN_ID", help="with --ladder: record a finished run as a ladder row")
    p.set_defaults(func=cmd_eval)
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
