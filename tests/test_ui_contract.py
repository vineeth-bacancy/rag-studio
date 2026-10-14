"""Static checks on the UI that a type checker cannot make.

1. Every launch path binds 127.0.0.1. Next defaults to 0.0.0.0, and a UI on a
   routable interface would proxy the whole loopback-only API to the network.
2. Every page name comes from one list (src/lib/routes.ts), and every listed
   route has a page. A second list is how one page ends up with three names.
3. Copy rules from docs/product-language.md that are cheap to enforce.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
UI = ROOT / "web" / "ui"
SRC = UI / "src"


def test_package_scripts_pin_loopback():
    scripts = json.loads((UI / "package.json").read_text())["scripts"]
    for name in ("dev", "start"):
        assert "--hostname 127.0.0.1" in scripts[name], name


def test_launcher_pins_loopback():
    sh = (ROOT / "scripts" / "run_web.sh").read_text()
    assert "--host 127.0.0.1" in sh
    for line in sh.splitlines():
        if re.search(r"next (dev|start)", line) and not line.strip().startswith("#"):
            assert "--hostname 127.0.0.1" in line, line


def _routes() -> list[str]:
    text = (SRC / "lib" / "routes.ts").read_text()
    return re.findall(r'^\s*"(/[^"]*)":\s*\{', text, re.MULTILINE)


def test_every_route_has_a_page():
    for route in _routes():
        page = SRC / "app" / route.strip("/") / "page.tsx"
        assert page.exists(), f"{route} is named in routes.ts but has no page"


def test_every_page_is_named():
    named = set(_routes())
    for page in (SRC / "app").rglob("page.tsx"):
        rel = page.parent.relative_to(SRC / "app").as_posix()
        route = "/" if rel == "." else f"/{rel.split('/')[0]}"
        assert route in named, f"{page} has no entry in routes.ts"


def test_nav_labels_are_not_hard_coded():
    # Sidebar, palette and top bar must read names from routes.ts.
    for f in ("nav.ts", "app-sidebar.tsx", "command-palette.tsx", "top-bar.tsx"):
        text = (SRC / "components" / "shell" / f).read_text()
        for title in ("System Health", "How it works", "Evaluation", "Documents"):
            assert f'"{title}"' not in text, f"{f} spells the page name {title!r} itself"


def test_no_banned_implementation_words_in_primary_copy():
    # docs/product-language.md: "ollama offline" -> "AI unavailable", etc.
    banned = [r"ollama offline", r"Python API", r"No usable model"]
    for path in SRC.rglob("*.tsx"):
        text = path.read_text()
        for word in banned:
            assert not re.search(word, text, re.IGNORECASE), f"{path.name}: {word}"
