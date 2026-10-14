"""Sum wheel download sizes for a uv-compiled requirements file (metadata only, nothing installed)."""

import json
import re
import sys
import urllib.request


def pick(files):
    best = None
    for f in files:
        fn = f["filename"]
        if not fn.endswith(".whl"):
            continue
        ok = ("cp312" in fn or "abi3" in fn or "py3-none" in fn or "py2.py3" in fn) and \
             ("manylinux" in fn and "x86_64" in fn or fn.endswith("-any.whl"))
        if ok and "musllinux" not in fn:
            if best is None or ("cp312" in fn and "cp312" not in best["filename"]):
                best = f
    return best


rows = []
for line in open(sys.argv[1]):
    m = re.match(r"^([A-Za-z0-9_.\-]+)==([^\s;]+)", line)
    if not m:
        continue
    name, ver = m.groups()
    if "+cpu" in ver:
        rows.append((name, ver, None))
        continue
    try:
        d = json.load(urllib.request.urlopen(f"https://pypi.org/pypi/{name}/{ver}/json", timeout=30))
        w = pick(d["urls"])
        rows.append((name, ver, w["size"] if w else None))
    except Exception as exc:  # noqa: BLE001
        rows.append((name, ver, None))
        print("fail", name, exc, file=sys.stderr)

known = sum(r[2] for r in rows if r[2])
print(f"{sys.argv[1]}: {len(rows)} pkgs, known wheel bytes {known / 1e6:.0f} MB; unknown: {[r[0] for r in rows if not r[2]]}")
for name, ver, size in sorted(rows, key=lambda r: -(r[2] or 0))[:12]:
    print(f"  {name}=={ver}: {(size or 0) / 1e6:.0f} MB")
