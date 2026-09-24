"""Replace the test_real passages condensed from in-copyright sources.

test_real's docstring in split_corpus.py says the passages are "condensed from
real public-domain sources". Three were not:

  bib_001   C.S. Lewis, Mere Christianity -- the trilemma, and the wording ran
            close to Lewis's own ("a lunatic on the level of a man who believes
            he is a poached egg" is nearly his phrase). Published 1952.
  hist_003  Martin Luther King Jr., Letter from Birmingham Jail. Published 1963,
            and the King estate enforces its rights actively.
  bib_004   attributed "Lewis-style", but the text is the generic moral argument,
            which predates Lewis by centuries and carries none of his expression.

A condensed restatement of an argument's STRUCTURE is not a reproduction of its
expression, so the exposure was low. It is also unnecessary: this repository is
going public, and the two genuinely Lewis/King passages are 2 of 34 records.

So the two are replaced with public-domain arguments of the SAME form and
category, which keeps the split's shape intact:

  bib_001   disjunctive syllogism, biblical  -> Elijah at Mount Carmel (1 Kings 18)
  hist_003  categorical syllogism, historical -> Frederick Douglass, "What to the
            Slave is the Fourth of July?" (1852), on the law conceding the
            manhood of the enslaved

bib_004 keeps its text and label; only the misleading attribution changes.

Applied to labeled_reviewed.jsonl, NOT to data/splits/test_real.jsonl, because
split_corpus.py copies that file wholesale and would otherwise undo the edit --
the same way it restored the `sign` labels that clean_splits.py had patched.

    python pipeline/label/replace_incopyright_test_real.py --apply
"""

import argparse
import json

from lfx.schema import inconsistency, schema_ok

REPLACE = {
    "bib_001": {
        "id": "bib_001",
        "text": (
            "Elijah put the matter to the people as a choice between two: either the "
            "LORD is God or Baal is God, and they could not go on halting between the "
            "two opinions. The test he proposed was that each side prepare a sacrifice "
            "and call on its god, and that the god who answered by fire would be the "
            "true one. Baal was called on from morning until noon and gave no answer, "
            "and no fire fell. The LORD was called on and fire fell and consumed the "
            "offering. Therefore the LORD, and not Baal, is God."
        ),
        "label": {
            "premises": [
                "Either the LORD is God or Baal is God, and not both.",
                "The true God is the one who answers the sacrifice by fire.",
                "Baal was called on and did not answer by fire.",
                "The LORD was called on and answered by fire.",
            ],
            "conclusion": "The LORD, and not Baal, is God.",
            "argument_type": "deductive",
            "form": ["disjunctive syllogism"],
            "suppressed_premise": "The test by fire reliably identifies which of the two is God.",
        },
        "source": "1 Kings 18 (Elijah at Mount Carmel, condensed)",
        "category": "biblical",
        "reviewed_by": "expert-annotator",
        "review_notes": "replaces the C.S. Lewis trilemma (in copyright); same form and category",
    },
    "hist_003": {
        "id": "hist_003",
        "text": (
            "The law of the land already recognises the enslaved man as a moral and "
            "accountable being. There are statutes in the southern states prescribing "
            "punishment for crimes committed by him, and a being can be justly punished "
            "for a crime only if he is capable of choosing between right and wrong. No "
            "such statutes exist for the beasts of the field, because a beast is "
            "incapable of any such choice. Therefore the slave is a man, and the law "
            "that treats him as property contradicts what the law elsewhere concedes."
        ),
        "label": {
            "premises": [
                "Only a being capable of choosing between right and wrong can be justly punished for a crime.",
                "The statutes of the southern states prescribe punishment for crimes committed by enslaved people.",
                "No such statutes exist for the beasts of the field, which are incapable of such a choice.",
            ],
            "conclusion": (
                "The enslaved man is a man, and the law that treats him as property "
                "contradicts what the law elsewhere concedes."
            ),
            "argument_type": "deductive",
            "form": ["categorical syllogism"],
            "suppressed_premise": "How the law treats a being is evidence of what the law concedes about its nature.",
        },
        "source": 'Frederick Douglass, "What to the Slave is the Fourth of July?" (1852, condensed)',
        "category": "historical",
        "reviewed_by": "expert-annotator",
        "review_notes": "replaces the MLK Letter from Birmingham Jail passage (in copyright); same form and category",
    },
}

REATTRIBUTE = {
    "bib_004": "Moral argument (classical formulation, condensed)",
}

ap = argparse.ArgumentParser()
ap.add_argument("--real", default="data/interim/labeled_reviewed.jsonl")
ap.add_argument("--apply", action="store_true")
args = ap.parse_args()

rows = [json.loads(l) for l in open(args.real)]
out, swapped, retagged = [], 0, 0
for r in rows:
    if r["id"] in REPLACE:
        new = dict(REPLACE[r["id"]])
        if not schema_ok(new["label"]) or inconsistency(new["label"]):
            raise SystemExit(f"{new['id']}: replacement label is invalid")
        print(f"  SWAP    {r['id']}  {r.get('source')}\n"
              f"            -> {new['source']}")
        out.append(new)
        swapped += 1
    elif r["id"] in REATTRIBUTE:
        print(f"  RETAG   {r['id']}  {r.get('source')}\n"
              f"            -> {REATTRIBUTE[r['id']]}  (text and label unchanged)")
        out.append(dict(r, source=REATTRIBUTE[r["id"]]))
        retagged += 1
    else:
        out.append(r)

print(f"\n{len(rows)} records: {swapped} swapped, {retagged} re-attributed")
missing = set(REPLACE) | set(REATTRIBUTE) - {r["id"] for r in rows}
if swapped != len(REPLACE):
    print(f"WARNING: expected {len(REPLACE)} swaps")

if args.apply:
    with open(args.real, "w") as fh:
        for r in out:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"wrote {args.real}\n"
          f"now regenerate: data/splits/test_real.jsonl and its .chat.jsonl")
else:
    print("\nDRY RUN — re-run with --apply to write")
