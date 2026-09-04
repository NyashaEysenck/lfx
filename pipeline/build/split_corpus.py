"""Build train / val / test splits, stratified by the labeled `form`.

Two test sets, deliberately:
  test_gen.jsonl  — held out from the generated pool. In-distribution: same
                    generator, same prose style as training.
  test_real.jsonl — the 34 human-and-model-reviewed passages condensed from real
                    public-domain sources. Never trained on, never generated.

The gap between scores on these two is the number that matters. test_gen says
whether the model learned the task; test_real says whether it learned the task or
just learned Gemini's prose habits. Only test_real resembles what the finished
tool gets pointed at.

Held-out splits are filled from AGREEING records first — those where the blind
labeler independently assigned the form the passage was generated to instantiate.
Agreement is only a consistency signal (generator and labeler are the same model),
but a contested label is a known-bad thing to measure against, so the disagreements
are pushed into train, where label noise is survivable.
"""

import argparse
import collections
import json
import random

ap = argparse.ArgumentParser()
ap.add_argument("--generated", default="data/interim/corpus_all.jsonl")
ap.add_argument("--real", default="data/interim/labeled_reviewed.jsonl")
ap.add_argument("--seed", type=int, default=20260902)
ap.add_argument("--val-frac", type=float, default=0.15)
ap.add_argument("--test-frac", type=float, default=0.15)
args = ap.parse_args()

rng = random.Random(args.seed)
gen = [json.loads(l) for l in open(args.generated)]

by_form = collections.defaultdict(list)
for r in gen:
    by_form[r["label"]["form"]].append(r)

train, val, test = [], [], []
for form, rows in sorted(by_form.items()):
    for r in rows:
        # Trust depends on where the label came from:
        #   template     — correct by construction
        #   logic-human  — human annotation
        #   gemini       — trusted only where the blind label matched what the
        #                  passage was generated to instantiate
        origin = r.get("origin", "gemini")
        r["intent_agreement"] = (
            True if origin in ("template", "logic-human")
            else r.get("intended_form") == r["label"]["form"])
    agree = [r for r in rows if r["intent_agreement"]]
    disagree = [r for r in rows if not r["intent_agreement"]]
    rng.shuffle(agree)
    rng.shuffle(disagree)

    n = len(rows)
    n_val = max(1, round(n * args.val_frac)) if n >= 3 else 0
    n_test = max(1, round(n * args.test_frac)) if n >= 3 else 0
    if n_val + n_test >= n:                      # never starve train
        n_val = n_test = max(0, (n - 1) // 2)

    # agreeing records fill the held-out splits; whatever is left, plus every
    # contested record, becomes training data
    pool = agree + disagree
    test += pool[:n_test]
    val += pool[n_test:n_test + n_val]
    train += pool[n_test + n_val:]

for name, rows in (("train", train), ("val", val), ("test_gen", test)):
    rng.shuffle(rows)
    with open(f"data/splits/{name}.jsonl", "w") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")

real = [json.loads(l) for l in open(args.real)]
with open("data/splits/test_real.jsonl", "w") as fh:
    for r in real:
        fh.write(json.dumps(r, ensure_ascii=False) + "\n")

print(f"train {len(train)}  ·  val {len(val)}  ·  test_gen {len(test)}  "
      f"·  test_real {len(real)} (held out, real prose)\n")

forms = sorted(by_form)
w = max(len(f) for f in forms) + 2
print(f"{'form':<{w}} {'train':>6}{'val':>5}{'test':>5}{'real':>6}")
rc = collections.Counter(r["label"]["form"] for r in real)
for f in forms:
    c = lambda rows: sum(1 for r in rows if r["label"]["form"] == f)
    print(f"{f:<{w}} {c(train):>6}{c(val):>5}{c(test):>5}{rc[f]:>6}")

missing = [f for f in forms if not any(r["label"]["form"] == f for r in test)]
if missing:
    print("\nWARNING — forms absent from test_gen:", ", ".join(missing))

for name, rows in (("train", train), ("val", val), ("test_gen", test)):
    n_bad = sum(1 for r in rows if not r["intent_agreement"])
    print(f"{name:<9} {len(rows):>4} records, {n_bad:>3} with a contested label "
          f"({n_bad / max(len(rows), 1):.0%})")
