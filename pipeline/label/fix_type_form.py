"""Repair records whose argument_type and form contradict each other.

`form` and `argument_type` are not independent: naming an inductive form commits
you to argument_type "inductive", and vice versa. Four records in corpus_all.jsonl
violate that — Gemini labelling slips that taught the model an impossible pairing
is available. The model reproduced it (a passage came back "deductive · authority").

Each fix below names which field the passage supports, and why.
"""
import json

FIXES = {
    # Charcoal density as a proxy for past wildfires: reasoning from a reliable
    # indicator to an underlying condition. `sign` is right; the type was wrong.
    "gen_sign_0204": ("argument_type", "inductive"),

    # "Whenever a nation industrialises, its population urbanises. Britain
    # industrialised. Therefore..." — that is If P then Q; P; so Q. The blind
    # labeller called it causal because the subject matter is causal, but the
    # INFERENCE is modus ponens, and the type was already deductive.
    "gen_modus_ponens_0008": ("form", "modus ponens"),

    # Infers that consciousness "in its entirety" is non-physical from a single
    # aspect of it — generalising from too little. Hasty generalisation is
    # inductive; the type was wrong.
    "gen_hasty_generalization_0287": ("argument_type", "inductive"),

    # "People with the flu OFTEN experience fever and aches; I have both; so I have
    # the flu." Affirming the consequent needs a strict conditional, and "often"
    # is not one. This is sign reasoning from symptoms to condition — which the
    # existing inductive type already matches.
    "gen_affirming_the_consequent_0251": ("form", "sign"),
}

rows = [json.loads(l) for l in open("data/interim/corpus_all.jsonl")]
n = 0
for r in rows:
    if r["id"] in FIXES:
        field, value = FIXES[r["id"]]
        before = r["label"][field]
        r["label"][field] = value
        r["review_notes"] = (r.get("review_notes", "") +
                             f" [type/form repair: {field} {before} -> {value}]").strip()
        print(f"  {r['id']:<34} {field}: {before} -> {value}")
        n += 1

with open("data/interim/corpus_all.jsonl", "w") as fh:
    for r in rows:
        fh.write(json.dumps(r, ensure_ascii=False) + "\n")
print(f"\nrepaired {n} records")
