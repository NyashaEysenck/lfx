"""Remove non-argumentative records from the splits, and retire the `sign` label.

Two defects, both properties of the TEXT or the SCHEMA rather than of any model's
output, so removing them cannot flatter a model:

  - quiz prompts and dictionary definitions that LOGIC files as instances of the
    fallacy they describe ("Doing something because everyone else is doing it",
    "...What fallacy has Louise committed? http://www.funtrivia.com"). No model
    can extract an argument from a question about arguments, so these are
    unwinnable records inflating every error rate. 5.3% of the held-out human set.
  - two records in test_real still labelled `sign`, a form schema v1.2 deleted and
    merged into `inference to the best explanation`. No model trained after v1.2
    has ever seen that label, so those records were permanently wrong -- 6% of a
    34-record split, a real part of the floor it has been stuck at.

The detection rule lives in import_logic.is_argument so the next rebuild applies
it at import and these records never enter the corpus again. This script exists to
fix the splits we have already measured against.

Predictions join on id, so shrinking a gold set shrinks it identically for every
model and the comparison between them stays fair.

    python pipeline/build/clean_splits.py --apply
"""

import argparse
import json
import pathlib
import sys

sys.path.insert(0, "pipeline/corpus")
from import_logic import is_argument  # noqa: E402

from lfx.schema import INDUCTIVE_FORMS, forms_of, inconsistency, schema_ok  # noqa: E402

# v1.2 merged `sign` into IBE: every record it held was observed-indicator ->
# underlying-condition, which is IBE in Duke's framing.
RETIRED_FORMS = {"sign": "inference to the best explanation"}

ap = argparse.ArgumentParser()
ap.add_argument("--splits", default="data/splits")
ap.add_argument("--apply", action="store_true", help="write; otherwise dry-run")
args = ap.parse_args()

total_dropped = total_retitled = total_repaired = 0
for path in sorted(pathlib.Path(args.splits).glob("*.jsonl")):
    if path.name.endswith(".chat.jsonl"):
        continue  # regenerated from the plain split by format_for_training
    rows = [json.loads(l) for l in open(path)]
    kept, dropped, retitled, repaired = [], [], [], []

    for r in rows:
        if not is_argument(r["text"], r.get("origin")):
            dropped.append(r)
            continue
        forms = forms_of(r["label"])
        if any(f in RETIRED_FORMS for f in forms):
            want = [RETIRED_FORMS.get(f, f) for f in forms]
            r = dict(r, label=dict(r["label"], form=want, argument_type="inductive"))
            retitled.append((r["id"], " + ".join(forms), " + ".join(want)))
        kept.append(r)

    # Five corpus records carry an inductive form with argument_type "deductive",
    # four of them in train.jsonl -- so every model from v2 through the 3B trained
    # on labels that contradict themselves. corpus_v13 fixes them at the source;
    # the already-built splits need the same repair. Form implies argument_type,
    # so the type is the field that moves.
    for r in kept:
        if inconsistency(r["label"]):
            want = "inductive" if forms_of(r["label"])[0] in INDUCTIVE_FORMS else "deductive"
            r["label"] = dict(r["label"], argument_type=want)
            repaired.append((r["id"], " + ".join(forms_of(r["label"])), want))

    bad = [r["id"] for r in kept
           if not schema_ok(r["label"]) or inconsistency(r["label"])]
    if bad:
        raise SystemExit(f"{path.name}: would write invalid records {bad}")

    total_dropped += len(dropped)
    total_retitled += len(retitled)
    total_repaired += len(repaired)
    verb = "writing" if args.apply else "would drop"
    print(f"{path.name:26} {len(rows):>4} -> {len(kept):>4}   "
          f"{verb} {len(dropped)} non-arguments"
          + (f", {len(retitled)} retired-form relabels" if retitled else "")
          + (f", {len(repaired)} argument_type repairs" if repaired else ""))
    for rid, was, now in retitled:
        print(f"      {rid}: {was} -> {now}")

    if args.apply and (dropped or retitled or repaired):
        with open(path, "w") as fh:
            for r in kept:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")

print(f"\n{'applied' if args.apply else 'DRY RUN'}: "
      f"{total_dropped} non-arguments, {total_retitled} retired-form relabels, "
      f"{total_repaired} argument_type repairs")
if not args.apply:
    print("re-run with --apply to write")
