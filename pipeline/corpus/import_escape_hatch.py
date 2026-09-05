"""Collect real arguments whose form is genuinely outside the 22, for `other`.

Dropping the 24 unstable generated records left `other` with 0 training examples
against 16 in the held-out splits -- a label the schema offers and no model could
produce, which is exactly what `sign` was. The escape hatch needs real examples or
an appeal to ignorance gets forced into the nearest wrong class at inference.

Drawn from the LOGIC classes deliberately NOT imported in Phase 2, because they
had no counterpart in our enum. That is precisely the property wanted here.

  appeal to emotion     151   "Dad, why do I have to spend my summer at Jesus camp?"
  fallacy of relevance  156   red herring: "to look at global warming we must first
                              consider how the homeless suffer when it is cold"
  intentional           130   mixed; holds appeal to ignorance, "aliens must exist
                              because there is no evidence they don't"

`false causality` (203) is DELIBERATELY EXCLUDED even though it is the largest.
Our `causal` is the VALID inductive form, so post hoc examples filed under `other`
would teach the model that causal-looking text is unnameable -- and `causal` was
already one of the least stable classes in the Phase 7 control (5/9). That is the
shape of the `faulty generalization` error: a class whose NAME suggests a mapping
its EXTENSION does not support.

Sampled ACROSS the three rather than taken from one, so the class means "none of
the 22 apply" rather than "emotional appeal". A single-source `other` would just
rebuild the double-meaning problem that made it unlearnable.

    python pipeline/corpus/import_escape_hatch.py --apply
"""

import argparse
import collections
import csv
import glob
import json
import random

from import_logic import is_argument

SOURCES = ["appeal to emotion", "fallacy of relevance", "intentional"]
EXCLUDED = {"false causality": "collides with our `causal`, the valid inductive form"}

ap = argparse.ArgumentParser()
ap.add_argument("--per-class", type=int, default=16)
ap.add_argument("--seed", type=int, default=11)
ap.add_argument("--out", default="data/interim/escape_hatch.jsonl")
ap.add_argument("--apply", action="store_true")
args = ap.parse_args()

pool = collections.defaultdict(list)
for path in sorted(glob.glob("data/logic/*.csv")):
    with open(path, newline="", encoding="utf-8", errors="replace") as fh:
        for row in csv.DictReader(fh):
            lab = (row.get("updated_label") or row.get("label") or "").strip()
            txt = (row.get("source_article") or row.get("text") or "").strip()
            # Short fragments carry no argument to extract; the LOGIC classes here
            # are noisier than the imported ones, so the floor is higher.
            if lab in SOURCES and txt and is_argument(txt) and len(txt) > 60:
                pool[lab].append(txt)

rng = random.Random(args.seed)
picked, seen = [], set()
for cls in SOURCES:
    for txt in rng.sample(pool[cls], min(args.per_class, len(pool[cls]))):
        if txt in seen:
            continue
        seen.add(txt)
        picked.append({"id": f"esc_{len(picked):04d}", "text": txt,
                       "origin": "logic-human", "logic_class": cls})

print(f"excluded: {EXCLUDED}")
for cls in SOURCES:
    print(f"  {cls:24} pool {len(pool[cls]):>4}  ->  taking {args.per_class}")
print(f"\n{len(picked)} candidates selected\n")
for r in picked[:6]:
    print(f"  [{r['logic_class']:20}] {r['text'][:96]}")

if args.apply:
    with open(args.out, "w") as fh:
        for r in picked:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"\nwrote {args.out} — NOT yet labelled; blind-label next and keep only "
          f"the records the labeler does not confidently place in an existing class")
else:
    print("\nDRY RUN — re-run with --apply to write")
