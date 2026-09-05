"""Drop the generated `other` records: the labeler cannot label them stably.

Measured, not assumed. The same 26 `other` passages were labelled twice at
temperature 0, changing nothing but adding two chain examples to the few-shot:
**65% of the labels changed**, and only 5 of the 17 changes were the intended
single->chain effect. The rest were single form -> a DIFFERENT single form
(`inference to the best explanation` -> `modus ponens`, `categorical syllogism` ->
`authority`, `authority` -> `other`), which the change had nothing to do with.

The control says this is specific to `other`, not a property of the labeler: the
same test over 60 records from six well-defined classes flipped 13%, and six of
those eight were the chain examples working as intended, leaving ~3% genuine
instability. ad hominem, categorical syllogism and analogy were 100% stable.

So for these passages the answer is driven by prompt context rather than by the
text. Re-labelling them would inject noise; keeping them teaches the model to
answer `other` for arguments that are not actually unnameable. They go.

The two logic-human records STAY. Those were adjudicated by a person during the
Phase 6 audit and are what `other` is now for -- an appeal to ignorance and a
continuum fallacy, real arguments this vocabulary cannot name.

    python pipeline/label/drop_unstable_other.py --apply
"""

import argparse
import collections
import json

from lfx.schema import forms_of

ap = argparse.ArgumentParser()
ap.add_argument("--corpus", default="data/interim/corpus_v13.jsonl")
ap.add_argument("--out", default="data/interim/corpus_v20.jsonl")
ap.add_argument("--splits", nargs="*",
                default=["data/splits/train.jsonl", "data/splits/val.jsonl"])
ap.add_argument("--apply", action="store_true")
args = ap.parse_args()


def unstable(r):
    """A generated `other` record: no stable label, and not a real escape hatch."""
    return forms_of(r["label"]) == ["other"] and r.get("origin") == "gemini"


rows = [json.loads(l) for l in open(args.corpus)]
kept = [r for r in rows if not unstable(r)]
gone = [r for r in rows if unstable(r)]
left = collections.Counter(r.get("origin") for r in kept if forms_of(r["label"]) == ["other"])
print(f"{args.corpus}: {len(rows)} -> {len(kept)}   dropped {len(gone)} generated `other`")
print(f"  `other` remaining: {dict(left)}  (human-adjudicated escape hatches)")

# The already-built training splits carry the same records; test splits are left
# alone, since removing held-out records on the basis of a labelling experiment
# would be adjusting the ruler.
for path in args.splits:
    srows = [json.loads(l) for l in open(path)]
    skept = [r for r in srows if not unstable(r)]
    print(f"{path}: {len(srows)} -> {len(skept)}")
    if args.apply and len(skept) != len(srows):
        with open(path, "w") as fh:
            for r in skept:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")

if args.apply:
    with open(args.out, "w") as fh:
        for r in kept:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"\nwrote {args.out}")
else:
    print("\nDRY RUN — re-run with --apply to write")
