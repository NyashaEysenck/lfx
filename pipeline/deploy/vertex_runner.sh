#!/bin/bash
# Runs inside a Vertex AI custom job. Pulls the published model, scores a split,
# writes predictions back to GCS. No GPU: this project has zero GPU quota on every
# accelerator type, and requesting more is a console process we are not doing.
#
# CPU inference on a quantised 3B is slower than a laptop GPU in wall-clock terms
# and that is the entire point -- it is not the laptop.
set -euo pipefail

GOLD_URI="$1"     # gs://.../bench.jsonl
OUT_URI="$2"      # gs://.../preds.jsonl
MODEL="${3:-nyasha_stino/lfx:3b-q8}"   # :3b-q4 was never published -- that push was killed mid-upload
SHARD="${4:-0}"   # this shard index
SHARDS="${5:-1}"  # total shards

echo "==> installing ollama"
# The installer extracts a zstd archive and the deeplearning base-cpu image does
# not ship zstd, so it fails with a message about package managers rather than
# anything to do with this job.
apt-get update -qq && apt-get install -y -qq zstd >/dev/null
curl -fsSL https://ollama.com/install.sh | sh
nohup ollama serve > /tmp/ollama.log 2>&1 &
for i in $(seq 1 60); do
  curl -sf http://127.0.0.1:11434/api/tags >/dev/null && break || sleep 2
done

echo "==> pulling $MODEL"
ollama pull "$MODEL"

echo "==> fetching input"
python3 -m pip install --quiet google-cloud-storage
gsutil cp "$GOLD_URI" /tmp/gold.jsonl
wc -l /tmp/gold.jsonl

cat > /tmp/run.py <<'PY'
import json, os, sys, urllib.request
gold = [json.loads(l) for l in open("/tmp/gold.jsonl")]
shard, shards = int(sys.argv[1]), int(sys.argv[2])
model = sys.argv[3]
mine = [r for i, r in enumerate(gold) if i % shards == shard]
print(f"shard {shard}/{shards}: {len(mine)} of {len(gold)} records", flush=True)

def ask(text):
    body = json.dumps({"model": model, "stream": False,
                       "messages": [{"role": "user", "content": text}],
                       "options": {"temperature": 0, "top_p": 1,
                                   "num_predict": 768}}).encode()
    req = urllib.request.Request("http://127.0.0.1:11434/api/chat", data=body,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=900) as r:
        return json.loads(r.read())["message"]["content"]

def extract(raw):
    s = raw.find("{")
    if s < 0: return None
    depth = 0
    for i, c in enumerate(raw[s:], s):
        depth += (c == "{") - (c == "}")
        if depth == 0:
            try: return json.loads(raw[s:i+1])
            except Exception: return None
    return None

with open("/tmp/preds.jsonl", "w") as fh:
    for n, r in enumerate(mine, 1):
        raw = ask(r["text"])
        fh.write(json.dumps({"id": r["id"], "raw": raw, "parsed": extract(raw)},
                            ensure_ascii=False) + "\n")
        fh.flush()
        if n % 25 == 0:
            print(f"  {n}/{len(mine)}", flush=True)
print("done", flush=True)
PY

echo "==> predicting"
python3 /tmp/run.py "$SHARD" "$SHARDS" "$MODEL"
gsutil cp /tmp/preds.jsonl "$OUT_URI"
echo "==> wrote $OUT_URI"
