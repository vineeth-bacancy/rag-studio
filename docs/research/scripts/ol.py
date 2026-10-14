"""Small Ollama helper for live tests. Usage: python3 -I ol.py <test> [args]"""
import json, subprocess, sys, time, urllib.request, urllib.error, base64, pathlib

URL = "http://127.0.0.1:11434"
WORK = pathlib.Path(__file__).parent


def post(path, body, timeout=1800):
    req = urllib.request.Request(URL + path, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    t = time.time()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, json.loads(r.read()), time.time() - t
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode()[:600], time.time() - t


def get(path):
    with urllib.request.urlopen(URL + path) as r:
        return json.loads(r.read())


def guard():
    ps = get("/api/ps")["models"]
    tr = subprocess.run(["systemctl", "--user", "is-active", "trading-daily.service"], capture_output=True, text=True).stdout.strip()
    print("GUARD: trading-daily =", tr, "| loaded =", [(m["name"], round(m["size"] / 1e9, 2)) for m in ps])
    if tr == "active":
        sys.exit("trading active, skipping")
    big = [m for m in ps if m["size"] > 2.5e9 and m["name"] != "qwen3.5:4b"]
    if big:
        sys.exit("big model loaded, skipping: %s" % big[0]["name"])


def metrics(r, wall):
    s = lambda k: r.get(k, 0) / 1e9
    pe, ev = r.get("prompt_eval_count", 0), r.get("eval_count", 0)
    ped, evd = s("prompt_eval_duration"), s("eval_duration")
    return ("load_s=%.2f prompt_tokens=%d prompt_s=%.2f prompt_tok/s=%.1f gen_tokens=%d gen_s=%.2f gen_tok/s=%.2f total_s=%.2f wall_s=%.2f done_reason=%s"
            % (s("load_duration"), pe, ped, pe / ped if ped else 0, ev, evd, ev / evd if evd else 0, s("total_duration"), wall, r.get("done_reason")))


def chat(model, messages, think=None, options=None, keep_alive="5m", label=""):
    body = {"model": model, "messages": messages, "stream": False, "keep_alive": keep_alive, "options": options or {}}
    if think is not None:
        body["think"] = think
    st, r, wall = post("/api/chat", body)
    print("\n=== %s (think=%r, options=%s)" % (label, think, options))
    if st != 200:
        print("HTTP", st, r)
        return None
    m = r["message"]
    print(metrics(r, wall))
    th = m.get("thinking")
    print("thinking field present:", th is not None, "| thinking chars:", len(th or ""))
    if th:
        print("thinking (first 400 chars):", th[:400].replace("\n", " "))
    print("content:", repr(m.get("content", "")[:1200]))
    return r


def unload(model):
    st, r, _ = post("/api/chat", {"model": model, "messages": [], "keep_alive": 0})
    time.sleep(1.5)
    print("\nunload", model, st, "| /api/ps now:", [m["name"] for m in get("/api/ps")["models"]])


FILLER = (WORK / "filler.txt").read_text()
SYSTEM = (WORK / "system.txt").read_text()
PASSAGES = (WORK / "passages.txt").read_text()


def t_qwen3_small():
    M = "qwen3:1.7b"
    guard()
    chat(M, [{"role": "user", "content": "In two sentences, what is retrieval-augmented generation?"}], think=False,
         options={"temperature": 0, "num_ctx": 4096}, label="qwen3:1.7b think:false")
    print("/api/ps:", json.dumps(get("/api/ps")["models"], indent=0))
    # overflow test: marker at start, question at end, num_ctx 512
    prompt = ("The secret code word is PELICAN-42. Remember it.\n\n" + FILLER +
              "\n\nQuestion: What is the secret code word stated in the very first sentence of this message? "
              "If you cannot see it, answer exactly NOT FOUND.")
    chat(M, [{"role": "user", "content": prompt}], think=False, options={"temperature": 0, "num_ctx": 512},
         label="overflow single msg num_ctx=512")
    chat(M, [{"role": "user", "content": prompt}], think=False, options={"temperature": 0, "num_ctx": 4096},
         label="control same prompt num_ctx=4096")
    # multi-message: marker in the system prompt, filler in an earlier turn
    msgs = [{"role": "system", "content": "You are a helpful assistant. The secret code word is PELICAN-42."},
            {"role": "user", "content": "Read this background text:\n" + FILLER},
            {"role": "assistant", "content": "I have read it."},
            {"role": "user", "content": "What is the secret code word given in your system prompt? If you cannot see it, answer exactly NOT FOUND."}]
    chat(M, msgs, think=False, options={"temperature": 0, "num_ctx": 512}, label="overflow multi-turn num_ctx=512 (marker in system)")
    unload(M)


def rag_messages(question):
    return [{"role": "system", "content": SYSTEM},
            {"role": "user", "content": "Context passages:\n\n" + PASSAGES + "\n\nQuestion: " + question}]


def t_qwen35():
    M = "qwen3.5:4b"
    guard()
    opts = {"temperature": 0, "num_ctx": 8192}
    q1 = "By how much has Apple reduced iMac emissions against its business-as-usual scenario, and how much of that came from recycled content?"
    chat(M, rag_messages(q1), think=False, options=opts, label="qwen3.5:4b RAG think:false (cold load)")
    print("/api/ps:", json.dumps(get("/api/ps")["models"], indent=0))
    q2 = "What is the battery capacity of the Magic Keyboard?"
    chat(M, rag_messages(q2), think=False, options=opts, label="qwen3.5:4b RAG think:false unanswerable (warm)")
    chat(M, rag_messages(q1), think=True, options=dict(opts, num_predict=700), label="qwen3.5:4b RAG think:true (num_predict 700)")
    chat(M, [{"role": "user", "content": "Is 391 a prime number? Answer yes or no."}], think="low",
         options=dict(opts, num_predict=400), label="qwen3.5:4b think:'low' string")
    chat(M, [{"role": "user", "content": "Is 391 a prime number? Answer yes or no."}], think=None,
         options=dict(opts, num_predict=400), label="qwen3.5:4b think omitted (model default)")


def t_qwen35_image():
    M = "qwen3.5:4b"
    guard()
    img = base64.b64encode((WORK / "page2.png").read_bytes()).decode()
    chat(M, [{"role": "user", "content": "What does the chart on this page show? Answer in at most 3 sentences and include the main numbers.", "images": [img]}],
         think=False, options={"temperature": 0, "num_ctx": 8192}, label="qwen3.5:4b image page2 think:false")
    unload(M)


def t_qwen3_control():
    M = "qwen3:1.7b"
    guard()
    q = "\n\nWhat is the secret code word? Reply with the code word only, or NOT FOUND."
    short = "The secret code word is PELICAN-42.\n\n" + FILLER[:900] + q
    chat(M, [{"role": "user", "content": short}], think=False, options={"temperature": 0, "num_ctx": 512}, label="control: fits in 512")
    full = "The secret code word is PELICAN-42.\n\n" + FILLER + q
    chat(M, [{"role": "user", "content": full}], think=False, options={"temperature": 0, "num_ctx": 512}, label="overflow: 1.2k words in 512")
    msgs = [{"role": "system", "content": "The secret code word is PELICAN-42."},
            {"role": "user", "content": "Background text:\n" + FILLER},
            {"role": "assistant", "content": "OK."},
            {"role": "user", "content": "What is the secret code word? Reply with the code word only, or NOT FOUND."}]
    chat(M, msgs, think=False, options={"temperature": 0, "num_ctx": 512}, label="multi-turn overflow, marker in system")
    unload(M)


if __name__ == "__main__":
    globals()["t_" + sys.argv[1]]()
