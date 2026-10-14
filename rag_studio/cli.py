"""Command-line views of the engine, behind `./rag`.

More technical than the browser on purpose: endpoints, tags and paths are what
a terminal user needs. Lines follow docs/product-language.md (CLI section).
"""
from __future__ import annotations

import argparse
import json
import sys

from rag_studio import health

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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="./rag")
    sub = parser.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("health", help="check Ollama, models, documents, indexes and this computer")
    p.add_argument("--json", action="store_true", help="print the full report as JSON")
    p.set_defaults(func=cmd_health)
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
