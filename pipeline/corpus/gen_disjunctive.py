"""Build disjunctive syllogisms from genuinely EXHAUSTIVE disjunctions.

`false dilemma` reached 84 records after the LOGIC import while `disjunctive
syllogism` sat at 15 — and those are precisely the two classes the model confuses.
Training on that ratio would just reverse the bias.

The two forms share a structure and differ on one fact about the world:

    Either P or Q.  Not P.  => Q
      exhaustive disjunction  -> disjunctive syllogism  (valid)
      false choice            -> false dilemma          (fallacy)

So the clause bank must supply disjunctions that really are exhaustive — a switch
is on or off, a defendant is liable or not. Gemini is asked only for that property;
the form is fixed by construction.

    python gen_disjunctive.py --pairs 64
"""

import argparse
import json
import random

from google import genai
from google.genai import types

from lfx.vertex import MODEL, client

SCHEMA = {
    "type": "OBJECT",
    "properties": {"pairs": {"type": "ARRAY", "items": {
        "type": "OBJECT",
        "properties": {"a": {"type": "STRING"}, "b": {"type": "STRING"},
                       "not_a": {"type": "STRING"}, "subject": {"type": "STRING"}},
        "required": ["a", "b", "not_a", "subject"]}}},
    "required": ["pairs"],
}

SYSTEM = """\
You write EXHAUSTIVE binary alternatives — pairs where "either A or B" is genuinely
true because no third option exists. This exhaustiveness is the whole point.

Good (really exhaustive):
  a="the contract was signed before the deadline"  b="it was signed after the deadline"
  a="the sample tested positive"                   b="the sample tested negative"

Bad (a third option obviously exists — this would be a FALSE DILEMMA, not what we want):
  a="we cut the education budget"   b="the city goes bankrupt"
  a="you support the proposal"      b="you oppose progress"

For each pair give:
  a       one alternative
  b       the other, genuinely exhausting the possibilities with a
  not_a   the natural denial of a (usually close to b, phrased as a denial)
  subject a few words naming the topic

Plain declarative fragments, lower case, no trailing punctuation, no dangling
pronouns. Vary the subject matter.
"""

DOMAINS = ["law and evidence", "medicine and testing", "engineering states",
           "logistics and scheduling", "finance and accounting", "science and measurement",
           "everyday decisions", "public administration"]

DISJ = ["Either {a} or {b}.", "{a}, or else {b}.", "There are two possibilities: {a}, or {b}.",
        "Either {a}, or {b} — there is no third case."]
DENY = ["{x}.", "We have established that {x}.", "It is not the case that {a_pos}.",
        "In this instance, {x}."]
CONCL = ["Therefore, {x}.", "So {x}.", "It follows that {x}.", "Hence {x}.",
         "We can conclude that {x}."]


def cap(s):
    s = s.strip().rstrip(".")
    return s[0].upper() + s[1:] if s else s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", type=int, default=64)
    ap.add_argument("--out", default="data/interim/disjunctive_generated.jsonl")
    ap.add_argument("--seed", type=int, default=20260903)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    cl = client()

    pairs, per = [], 8
    for i in range(0, args.pairs, per):
        dom = DOMAINS[(i // per) % len(DOMAINS)]
        n = min(per, args.pairs - i)
        resp = cl.models.generate_content(
            model=MODEL, contents=f"Write {n} exhaustive binary alternatives from: {dom}.",
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM, response_mime_type="application/json",
                response_schema=SCHEMA, temperature=1.0))
        pairs += json.loads(resp.text)["pairs"]
        print(f"  {dom:<24} ({len(pairs)}/{args.pairs})")

    rows = []
    for pi, p in enumerate(pairs):
        disj = cap(rng.choice(DISJ).format(a=p["a"], b=p["b"])) + "."
        tmpl = rng.choice(DENY)
        deny = cap(tmpl.format(x=p["not_a"], a_pos=p["a"])) + "."
        concl = cap(rng.choice(CONCL).format(x=p["b"])) + "."
        rows.append({
            "id": f"disj_{len(rows) + 1:04d}",
            "category": "formal_disjunctive_syllogism",
            "source": f"template-constructed, exhaustive disjunction ({p.get('subject', '')})",
            "text": f"{disj} {deny} {concl}",
            "label": {
                "premises": [disj, deny],
                "conclusion": cap(p["b"]) + ".",
                "argument_type": "deductive",
                "form": "disjunctive syllogism",
                "suppressed_premise": None,
            },
            "form_source": "construction",
            "reviewed": False, "review_notes": "",
        })

    with open(args.out, "w") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"\nwrote {len(rows)} -> {args.out}")
    print("sample:\n  " + rows[0]["text"])


if __name__ == "__main__":
    main()
