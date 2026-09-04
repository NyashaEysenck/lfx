"""Combine the three data sources into one corpus, capping oversized classes.

Sources, and why each exists:
  generated_labeled.jsonl  400  Gemini prose + Gemini labels. Broad coverage of all
                                17 forms; weak precisely on the fallacies.
  logic_labeled.jsonl      413  Real passages, HUMAN form labels (LOGIC, Jin et al.
                                2022). Covers ad hominem and false dilemma only.
  formal_generated.jsonl   256  Template-built. Labels correct by construction.
                                Covers the two formal fallacies and their valid twins.
  disjunctive_generated     64  Template-built from EXHAUSTIVE disjunctions. Added
                                because the LOGIC import pushed false dilemma to 84
                                against 15 disjunctive syllogisms — the two classes
                                the model already confuses. Without this the bias
                                would simply reverse.

CAPPING. LOGIC contributes 285 ad hominem against ~20 per class elsewhere. Trained
as-is the model would learn that ad hominem is fifteen times more likely than modus
tollens and predict accordingly. Imported classes are capped; the surplus is written
to extra_test_human.jsonl rather than discarded — human-labeled records are more
valuable as evaluation data than as the 200th example of a class.

    python merge_corpus.py --cap 60
"""

import argparse
import collections
import json
import random

ap = argparse.ArgumentParser()
ap.add_argument("--cap", type=int, default=60,
                help="max records per form from the imported (LOGIC) source")
ap.add_argument("--out", default="data/interim/corpus_all.jsonl")
ap.add_argument("--surplus", default="data/splits/extra_test_human.jsonl")
ap.add_argument("--seed", type=int, default=20260902)
args = ap.parse_args()

rng = random.Random(args.seed)

def load(path, origin):
    rows = []
    for line in open(path):
        r = json.loads(line)
        r["origin"] = origin
        rows.append(r)
    return rows

gen = load("data/interim/generated_labeled.jsonl", "gemini")
logic = load("data/interim/logic_labeled.jsonl", "logic-human")
formal = load("data/interim/formal_generated.jsonl", "template")
formal += load("data/interim/disjunctive_generated.jsonl", "template")

# Cap only the imported source — the other two are already balanced by design.
kept, surplus = [], []
by_form = collections.defaultdict(list)
for r in logic:
    by_form[r["label"]["form"]].append(r)
for form, rows in by_form.items():
    rng.shuffle(rows)
    kept += rows[: args.cap]
    surplus += rows[args.cap:]

merged = gen + kept + formal
rng.shuffle(merged)

with open(args.out, "w") as fh:
    for r in merged:
        fh.write(json.dumps(r, ensure_ascii=False) + "\n")
with open(args.surplus, "w") as fh:
    for r in surplus:
        fh.write(json.dumps(r, ensure_ascii=False) + "\n")

counts = collections.Counter(r["label"]["form"] for r in merged)
origin = collections.Counter(r["origin"] for r in merged)
print(f"{len(merged)} records -> {args.out}")
print(f"  by origin: {dict(origin)}")
print(f"  surplus (human-labeled, held out) -> {args.surplus}: {len(surplus)}\n")

w = max(len(f) for f in counts) + 2
print(f"{'form':<{w}}{'total':>7}{'gemini':>8}{'human':>7}{'tmpl':>6}")
for form, n in sorted(counts.items(), key=lambda kv: -kv[1]):
    c = lambda o: sum(1 for r in merged if r["label"]["form"] == form and r["origin"] == o)
    print(f"{form:<{w}}{n:>7}{c('gemini'):>8}{c('logic-human'):>7}{c('template'):>6}")
