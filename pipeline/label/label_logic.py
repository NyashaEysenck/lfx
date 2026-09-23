"""Fill in the extraction fields for the LOGIC import, keeping the human `form`.

Gemini labels each passage BLIND — it is never told the human label. We then:
  - keep Gemini's premises / conclusion / argument_type / suppressed_premise
  - overwrite `form` with the human label from LOGIC
  - record what Gemini said as `gemini_form`

That last field turns this run into a measurement. Agreement between `gemini_form`
and the human label, on 820 passages whose classes Gemini is suspected of getting
wrong, quantifies the bias that produced the fallacy defect — against real ground
truth rather than the four examples that exposed it by hand.

    python label_logic.py --limit 60      # measure the bias cheaply first
    python label_logic.py                 # full run, resumable
"""

import argparse
import json
import os
import sys
import threading
from concurrent.futures import ThreadPoolExecutor

from google import genai

from lfx.vertex import build_examples, client, label_one

ap = argparse.ArgumentParser()
ap.add_argument("--infile", default="data/interim/logic_imported.jsonl")
ap.add_argument("--out", default="data/interim/logic_labeled.jsonl")
ap.add_argument("--fewshot", default="data/raw/fewshot.jsonl")
ap.add_argument("--limit", type=int)
ap.add_argument("--workers", type=int, default=4)
args = ap.parse_args()

rows = [json.loads(l) for l in open(args.infile)]
done = set()
if os.path.exists(args.out):
    done = {json.loads(l)["id"] for l in open(args.out)}
todo = [r for r in rows if r["id"] not in done]
if args.limit:
    todo = todo[: args.limit]
print(f"{len(rows)} imported, {len(done)} labeled, {len(todo)} to go")
if not todo:
    sys.exit()

cl = client()
examples = build_examples(args.fewshot)
out = open(args.out, "a")
lock = threading.Lock()
n = [0]


def one(r):
    try:
        lab = label_one(cl, examples, r["text"])
        gemini_form = lab["form"]
        lab["form"] = [r["gold_form"]] if isinstance(r["gold_form"], str) else list(r["gold_form"])
        category = r.get("category", f"logic_{r.get('gold_form', 'unknown').replace(' ', '_')}")
        source = r.get("source", f"LOGIC (Jin et al. 2022), human label: {r.get('gold_form', '')}")
        rec = {
            "id": r["id"],
            "category": category,
            "source": source,
            "text": r["text"],
            "label": lab,
            "gemini_form": gemini_form,
            "form_source": "human (LOGIC)",
            "reviewed": False,
            "review_notes": "",
        }
        with lock:
            out.write(json.dumps(rec, ensure_ascii=False) + "\n")
            out.flush()
            n[0] += 1
            print(f"  [{n[0]}/{len(todo)}] {r['id']} -> {lab['form']} (gemini: {gemini_form})", flush=True)
    except Exception as exc:  # noqa: BLE001
        print(f"  ! {r['id']}: {type(exc).__name__}: {exc}"[:120], file=sys.stderr)
        return


with ThreadPoolExecutor(max_workers=args.workers) as pool:
    list(pool.map(one, todo))
out.close()
print(f"wrote {n[0]} -> {args.out}")
