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

from lfx.schema import forms_of

ap = argparse.ArgumentParser()
ap.add_argument("--generated", default="data/interim/corpus_v12.jsonl")
ap.add_argument("--real", default="data/interim/labeled_reviewed.jsonl")
ap.add_argument("--seed", type=int, default=20260902)
ap.add_argument("--val-frac", type=float, default=0.15)
ap.add_argument("--test-frac", type=float, default=0.15)
args = ap.parse_args()

rng = random.Random(args.seed)
gen = [json.loads(l) for l in open(args.generated)]

# LOGIC carries the same passage under different ids, so a text-level dedup has to
# happen BEFORE the split or the duplicates land on opposite sides of it: the first
# run of this put two passages in both train and val. Ids are also made unique
# against the held-out human set, which keeps its own ids from an earlier merge --
# evaluate.py joins on id, so two records sharing one is a silent mis-score.
def _key(t):
    return " ".join(t.lower().split())[:150]


held_ids, held_texts = set(), set()
for path in ("data/splits/extra_test_human.jsonl", "data/splits/test_real.jsonl"):
    try:
        for line in open(path):
            r = json.loads(line)
            held_ids.add(r["id"])
            held_texts.add(_key(r["text"]))
    except FileNotFoundError:
        pass

seen, deduped, dropped = set(), [], 0
for r in gen:
    k = _key(r["text"])
    if k in seen or k in held_texts:
        dropped += 1
        continue
    seen.add(k)
    if r["id"] in held_ids:
        r["id"] = f"{r['id']}_s"
    deduped.append(r)
if dropped:
    print(f"dropped {dropped} passages already present elsewhere "
          f"(duplicate text within the corpus, or in a held-out split)")
gen = deduped

# Keyed by the form SET as a stable string: v2.0 labels are lists, and a chain
# should stratify as its own stratum rather than be filed under one of its moves.
by_form = collections.defaultdict(list)
for r in gen:
    by_form[" + ".join(forms_of(r["label"]))].append(r)

train, val, test = [], [], []
for form, rows in sorted(by_form.items()):
    for r in rows:
        # Trust depends on where the label came from:
        #   template     — correct by construction
        #   logic-human  — human annotation
        #   gemini       — trusted only where the blind label matched what the
        #                  passage was generated to instantiate
        origin = r.get("origin", "gemini")
        if origin == "template":
            r["intent_agreement"] = True          # correct by construction
        elif origin == "logic-human":
            # A human label is not automatically trustworthy: LOGIC's own tail is
            # noisy (a bare claim labelled equivocation, a non sequitur labelled
            # equivocation). Agreement between the human label and Gemini's blind
            # label is the available check, and per-class rates vary 64-89%.
            r["intent_agreement"] = (forms_of({"form": r.get("gemini_form")})
                                     == forms_of(r["label"]))
        else:
            r["intent_agreement"] = (forms_of({"form": r.get("intended_form")})
                                     == forms_of(r["label"]))
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
rc = collections.Counter(" + ".join(forms_of(r["label"])) for r in real)
for f in forms:
    c = lambda rows: sum(1 for r in rows if " + ".join(forms_of(r["label"])) == f)
    print(f"{f:<{w}} {c(train):>6}{c(val):>5}{c(test):>5}{rc[f]:>6}")

missing = [f for f in forms if not any(" + ".join(forms_of(r["label"])) == f for r in test)]
if missing:
    print("\nWARNING — forms absent from test_gen:", ", ".join(missing))

for name, rows in (("train", train), ("val", val), ("test_gen", test)):
    n_bad = sum(1 for r in rows if not r["intent_agreement"])
    print(f"{name:<9} {len(rows):>4} records, {n_bad:>3} with a contested label "
          f"({n_bad / max(len(rows), 1):.0%})")
