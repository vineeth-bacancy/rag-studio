import json, math, sys, time, urllib.request, urllib.error, base64

URL = "http://127.0.0.1:11434"


def post(path, body, timeout=600):
    req = urllib.request.Request(URL + path, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    t = time.time()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, json.loads(r.read()), time.time() - t
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode()[:400], time.time() - t


def get(path):
    with urllib.request.urlopen(URL + path) as r:
        return json.loads(r.read())


def norm(v):
    return math.sqrt(sum(x * x for x in v))


def cos(a, b):
    return sum(x * y for x, y in zip(a, b)) / (norm(a) * norm(b))


M = "nomic-embed-text"
text = "search_document: Apple iMac 24-inch product environmental report, carbon footprint by life cycle stage."

st, full, dt = post("/api/embed", {"model": M, "input": text, "keep_alive": "2m"})
v = full["embeddings"][0]
print("default: status", st, "dims", len(v), "L2norm %.6f" % norm(v), "prompt_eval_count", full.get("prompt_eval_count"), "load_s %.2f" % (full["load_duration"] / 1e9), "wall_s %.2f" % dt)
print("response keys:", sorted(full.keys()))

ps = get("/api/ps")
print("/api/ps while loaded:", json.dumps(ps, indent=1))

st, d256, dt = post("/api/embed", {"model": M, "input": text, "dimensions": 256, "keep_alive": "2m"})
if st == 200:
    w = d256["embeddings"][0]
    pref = v[:256]
    print("dimensions=256: dims", len(w), "L2norm %.6f" % norm(w), "| full[:256] L2norm %.6f" % norm(pref))
    print("  max|w - full[:256]| = %.3e" % max(abs(a - b) for a, b in zip(w, pref)))
    pn = [x / norm(pref) for x in pref]
    print("  max|w - full[:256]/||full[:256]|| | = %.3e" % max(abs(a - b) for a, b in zip(w, pn)))
    print("  cos(w, full[:256]) = %.6f" % cos(w, pref))
else:
    print("dimensions=256 error", st, d256)

# dimensions larger than native
st, r, dt = post("/api/embed", {"model": M, "input": text, "dimensions": 1024, "keep_alive": "2m"})
print("dimensions=1024 (>768):", st, (len(r["embeddings"][0]) if st == 200 else r))

# over-long input
long_text = " ".join(["The quarterly revenue of the company grew steadily across all regions and segments."] * 900)
st, r, dt = post("/api/embed", {"model": M, "input": long_text, "truncate": False, "keep_alive": "2m"})
print("truncate=false, long input:", st, (r if st != 200 else {"dims": len(r["embeddings"][0]), "prompt_eval_count": r.get("prompt_eval_count")}), "wall_s %.2f" % dt)
st, r, dt = post("/api/embed", {"model": M, "input": long_text, "keep_alive": "2m"})
print("truncate default(true), long input:", st, (r if st != 200 else {"dims": len(r["embeddings"][0]), "prompt_eval_count": r.get("prompt_eval_count")}), "wall_s %.2f" % dt)
st, r, dt = post("/api/embed", {"model": M, "input": long_text, "truncate": False, "options": {"num_ctx": 8192}, "keep_alive": "2m"})
print("truncate=false + options.num_ctx=8192, long input:", st, (r if st != 200 else {"dims": len(r["embeddings"][0]), "prompt_eval_count": r.get("prompt_eval_count")}), "wall_s %.2f" % dt)

# list input
st, r, dt = post("/api/embed", {"model": M, "input": ["search_query: a", "search_query: b", "search_query: c"], "keep_alive": "2m"})
print("list of 3:", st, len(r["embeddings"]) if st == 200 else r, "prompt_eval_count", r.get("prompt_eval_count") if st == 200 else None)

# multimodal object input on 0.32.6 (tiny 1x1 png)
png = base64.b64encode(bytes.fromhex("89504e470d0a1a0a0000000d4948445200000001000000010806000000" "1f15c4890000000d49444154789c6360000002000154a24f5d0000000049454e44ae426082")).decode()
st, r, dt = post("/api/embed", {"model": M, "input": {"text": "a cat", "image": png}, "keep_alive": "2m"})
print("object input {text,image}:", st, r if st != 200 else "OK dims %d" % len(r["embeddings"][0]))

# all-minilm with dimensions
st, r, dt = post("/api/embed", {"model": "all-minilm", "input": "hello world", "dimensions": 128, "keep_alive": 0})
print("all-minilm dimensions=128:", st, ("dims %d norm %.4f" % (len(r["embeddings"][0]), norm(r["embeddings"][0]))) if st == 200 else r)

# unload
st, r, dt = post("/api/embed", {"model": M, "input": "x", "keep_alive": 0})
print("unload via keep_alive 0:", st)
time.sleep(1)
print("/api/ps after:", get("/api/ps"))
