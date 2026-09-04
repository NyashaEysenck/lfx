"""Import human-labeled fallacies from LOGIC into our schema.

Division of labour, and the whole point of this script:
  - `form`  comes from LOGIC's HUMAN annotation. Gemini is demonstrably bad here —
            it maps false dilemma onto disjunctive syllogism, and the fine-tuned
            model inherited that bias.
  - everything else (premises, conclusion, argument_type, suppressed_premise) is
            left blank for `label_corpus.py` to fill. Gemini IS good at extraction
            (premise F1 0.83) and LOGIC provides none of those fields.

Only classes with an unambiguous counterpart in our enum are imported. The rest are
dropped rather than forced into `other`: importing 232 ad populum examples as
"other" would teach the model that `other` means "a fallacy I don't recognise",
which is not what `other` means in our schema (a multi-step chained argument).

    python import_logic.py --out logic_imported.jsonl
"""

import argparse
import csv
import json
import re

# LOGIC class -> our `form`. Only 1:1 correspondences.
#
# `faulty generalization` was in this map and has been REMOVED. Inspecting the
# passages (not the class name) showed LOGIC uses it for roughly "any bad inductive
# leap": slippery slopes, false cause, composition, and unsupported leaps with no
# sample at all. Our `hasty generalization` is the narrow thing — generalising from
# a sample too small to bear it. Gemini disagreed with the human label on 21 of 33
# sampled, and on inspection Gemini was right about nearly all of them. Importing
# those 407 rows would have poisoned the class it was meant to strengthen.
KEEP = {
    "ad hominem": "ad hominem",              # 87% Gemini/human agreement on the full set
    "false dilemma": "false dilemma",        # 79%
    # Added for schema v1.2 (Duke Think Again IV topics). Extensions were checked by
    # reading samples, not by trusting the class name — see the faulty
    # generalization error above. Sample quality varies:
    #   equivocation        6/6 clean, the best-curated class in LOGIC
    #   fallacy of extension ~5/6 straw man (one was ad hominem, one a quiz question)
    #   circular reasoning  ~4/6 (one was tu quoque)
    #   ad populum          ~3/6 — half the class is DEFINITIONS, not arguments
    "equivocation": "equivocation",
    "fallacy of extension": "straw man",
    "circular reasoning": "begging the question",
    "ad populum": "ad populum",
}

# LOGIC mixes glossary entries, quiz questions and sentence fragments in with real
# arguments. These are not label noise — they are not arguments at all, and training
# on them would teach the model to extract structure from text that has none.
NOT_AN_ARGUMENT = [
    # quiz scaffolding
    "which logical fallacy", "which fallacy", "is an example of which",
    "what fallacy", "identify the fallacy", "the statement above",
    # glossary phrasing: describes a fallacy rather than committing one
    "the claim, as evidence", "this type of", "refers to the", "is a fallacy",
    "is the fallacy", "fallacy of ", "this fallacy", "occurs when",
    "encourages the audience", "mentality",
]


def is_argument(text):
    """Reject glossary entries, quiz questions and fragments."""
    low = text.lower()
    return not any(m in low for m in NOT_AN_ARGUMENT)

# Deliberately not imported, with the reason — so this decision is reviewable.
DROP = {
    "ad populum": "no counterpart in our enum",
    "false causality": "our `causal` is the VALID inductive form; importing the fallacy under it would poison the class",
    "circular reasoning": "no counterpart",
    "appeal to emotion": "no counterpart",
    "fallacy of relevance": "too coarse to map",
    "fallacy of logic": "too coarse — this is where affirming-the-consequent hides, but unlabelled",
    "intentional": "not a fallacy category in our sense",
    "fallacy of extension": "no counterpart (straw man)",
    "fallacy of credibility": "close to `authority` but that is our VALID inductive form",
    "equivocation": "no counterpart",
    "faulty generalization": "LOGIC's class is far broader than our `hasty generalization` — see KEEP",
}


def clean(t):
    t = re.sub(r"\s+", " ", (t or "")).strip()
    return t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/interim/logic_imported.jsonl")
    ap.add_argument("--min-words", type=int, default=8,
                    help="drop fragments too short to hold a premise and a conclusion")
    args = ap.parse_args()

    seen, rows, dropped, short = set(), [], {}, 0
    for split in ("train", "dev", "test"):
        for r in csv.DictReader(open(f"data/logic/edu_{split}.csv")):
            lab = r.get("updated_label", "").strip()
            text = clean(r.get("source_article"))
            if lab not in KEEP:
                dropped[lab] = dropped.get(lab, 0) + 1
                continue
            if len(text.split()) < args.min_words or not is_argument(text):
                short += 1
                continue
            key = text.lower()
            if key in seen:
                continue
            seen.add(key)
            rows.append({
                "id": f"logic_{len(rows) + 1:04d}",
                "category": f"logic_{KEEP[lab].replace(' ', '_')}",
                "source": f"LOGIC (Jin et al. 2022), human label: {lab}",
                "text": text,
                "gold_form": KEEP[lab],      # human label — do NOT let Gemini overwrite
                "logic_split": split,
            })

    with open(args.out, "w") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"imported {len(rows)} -> {args.out}")
    for form in sorted(set(r["gold_form"] for r in rows)):
        print(f"  {sum(1 for r in rows if r['gold_form'] == form):>4}  {form}")
    print(f"\ndropped {sum(dropped.values())} rows in unmapped classes:")
    for lab, n in sorted(dropped.items(), key=lambda kv: -kv[1]):
        print(f"  {n:>4}  {lab:<24} {DROP.get(lab, '?')}")
    print(f"\nalso dropped: {short} that were too short or not arguments "
          f"(glossary entries, quiz questions, fragments), "
          f"{len(seen) - len(rows)} duplicates")


if __name__ == "__main__":
    main()
