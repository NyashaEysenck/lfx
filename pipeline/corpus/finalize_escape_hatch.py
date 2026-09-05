"""Turn the vetted escape-hatch candidates into corpus records for `other`.

Two filters, in this order:

  1. the blind labeler must also have said `other`. 67 of 134 did. Where it placed
     a candidate in an existing class instead, that record is dropped -- either it
     is not really outside the enum, or it is close enough to a real class that
     training on it as `other` would blur that class. The 67 came out evenly split
     across the three sources (22 / 23 / 22), which is what the escape hatch needs:
     `other` has to mean "none of the 22 apply", not "emotional appeal".
  2. a human read all 67. The exclusions below are that pass.

LOGIC's glossary entries in these classes are written as bare noun phrases -- "an
attempt to persuade the reader/audience based on feelings or emotions" -- so they
carry none of the phrases `is_argument` looks for and have to come out by hand.

The labeler's own premises and conclusion are kept: it read the passage correctly
in these cases, and only the FORM was ever in question.
"""

import argparse
import collections
import json

from lfx.schema import inconsistency, schema_ok

EXCLUDE = {
    # definitions of a fallacy, not instances of one
    "esc_0002": "definition: 'an attempt to persuade ... based on feelings'",
    "esc_0091": "definition of argument from incredulity",
    "esc_0118": "definition of shifting the burden of proof",
    # descriptions of an advert rather than an argument
    "esc_0011": "describes a real-estate ad; no argument",
    "esc_0024": "describes a commercial; no argument",
    "esc_0033": "describes a commercial; no argument",
    # not arguments at all
    "esc_0021": "loaded description, no conclusion drawn",
    "esc_0034": "first-person narrative, no argument",
    "esc_0064": "commentary ABOUT a red herring, not one",
    "esc_0014": "quiz prompt: 'Which type of appeal is used ...'",
    "esc_0094": "advertising claim, not a fallacy",
    # near-duplicates of another kept record: same argument, trivially reworded
    "esc_0023": "duplicate of the cat-sweater record",
    "esc_0127": "duplicate of the moon short-ribs record",
    "esc_0132": "duplicate of the invisible-unicorns record",
    "esc_0086": "duplicate of the phone-bill record",
    "esc_0081": "duplicate of the phone-bill record",
}

ap = argparse.ArgumentParser()
ap.add_argument("--labeled", default="data/interim/escape_hatch_labeled.jsonl")
ap.add_argument("--out", default="data/interim/escape_hatch_final.jsonl")
ap.add_argument("--apply", action="store_true")
args = ap.parse_args()

rows = [json.loads(l) for l in open(args.labeled)]
agreed = [r for r in rows if r["gemini_label"]["form"] == ["other"]]
kept = []
for r in agreed:
    if r["id"] in EXCLUDE:
        continue
    label = dict(r["gemini_label"], form=["other"])
    if not schema_ok(label) or inconsistency(label):
        print(f"  skipping {r['id']}: invalid label")
        continue
    kept.append({"id": r["id"], "text": r["text"], "origin": "logic-human",
                 "logic_class": r["logic_class"], "label": label})

mix = collections.Counter(r["logic_class"] for r in kept)
print(f"{len(rows)} labelled -> {len(agreed)} agreed `other` -> "
      f"{len(kept)} after the human pass ({len(EXCLUDE)} excluded)")
print(f"source mix: {dict(mix)}")
print("\nexcluded:")
for rid, why in EXCLUDE.items():
    print(f"  {rid}  {why}")

if args.apply:
    with open(args.out, "w") as fh:
        for r in kept:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"\nwrote {args.out}")
else:
    print("\nDRY RUN — re-run with --apply to write")
