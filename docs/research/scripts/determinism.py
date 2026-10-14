import hashlib
import json
import urllib.request

PROMPT = ("Context: The Eiffel Tower was completed in 1889 and is 330 metres tall.\n"
          "Claims: 1) It was completed in 1889. 2) It was painted red. 3) It was built in Lyon.\n"
          "For each claim give verdict 1 if supported by the context, else 0, with a one-sentence reason.")
SCHEMA = {"type": "object", "properties": {"verdicts": {"type": "array", "items": {"type": "object",
          "properties": {"claim": {"type": "integer"}, "verdict": {"type": "integer"}, "reason": {"type": "string"}},
          "required": ["claim", "verdict", "reason"]}}}, "required": ["verdicts"]}


def call(seed):
    body = {"model": "qwen3:1.7b", "stream": False, "think": False, "format": SCHEMA,
            "messages": [{"role": "user", "content": PROMPT}],
            "options": {"temperature": 0, "seed": seed, "num_ctx": 4096}}
    req = urllib.request.Request("http://127.0.0.1:11434/api/chat", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req, timeout=600))["message"]["content"]


outs = [call(42) for _ in range(3)] + [call(7)]
for i, o in enumerate(outs):
    print(i, hashlib.sha256(o.encode()).hexdigest()[:12], [v["verdict"] for v in json.loads(o)["verdicts"]])
print("identical across 3 runs seed=42:", len(set(outs[:3])) == 1, "| seed=7 same as seed=42:", outs[3] == outs[0])
