"""Run an Ollama-served model over a split, in evaluate.py's format.

Deliberately sends ONLY the user turn: the system prompt comes from the model's
own Modelfile. That makes this a test of the deployed artifact rather than of a
prompt handed to it here -- if the Modelfile's SYSTEM block were wrong or stale,
this run shows it as a score drop instead of hiding it behind a correct prompt
supplied at call time. That failure has happened once already in this project.

Same greedy decoding and same JSON extraction as predict_mlx.py, so any gap
against the MLX numbers is attributable to GGUF conversion and quantisation,
which is the thing being measured.

    python pipeline/eval/predict_ollama.py --split test_real --model lfx-3b \\
        --out results/preds_ollama_3b_real.jsonl
"""

import argparse
import json
import urllib.request

from lfx.jsonio import extract_json

ap = argparse.ArgumentParser()
ap.add_argument("--split", required=True)
ap.add_argument("--model", required=True, help="ollama model name, e.g. lfx-3b")
ap.add_argument("--out", required=True)
ap.add_argument("--host", default="http://127.0.0.1:11434")
args = ap.parse_args()

rows = [json.loads(l) for l in open(f"data/splits/{args.split}.jsonl")]


def ask(text):
    body = json.dumps({
        "model": args.model,
        "messages": [{"role": "user", "content": text}],
        "stream": False,
        # mirror the Modelfile, in case the served copy was created differently
        "options": {"temperature": 0, "top_p": 1, "num_predict": 768},
    }).encode()
    req = urllib.request.Request(f"{args.host}/api/chat", data=body,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r:
        return json.loads(r.read())["message"]["content"]


with open(args.out, "w") as fh:
    for i, r in enumerate(rows, 1):
        raw = ask(r["text"])
        fh.write(json.dumps({"id": r["id"], "raw": raw, "parsed": extract_json(raw)},
                            ensure_ascii=False) + "\n")
        fh.flush()
        if i % 10 == 0:
            print(f"  {i}/{len(rows)}", flush=True)
print(f"wrote {args.out}")
