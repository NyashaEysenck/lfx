"""Generate the four conditional forms from shared clause pairs.

Affirming the consequent and denying the antecedent are the two forms the model
fails hardest on, and LOGIC has no usable examples of either. But these forms are
DEFINED by their structure, so they can be built rather than labelled:

    If P then Q.  P.      => Q       modus ponens              (valid)
    If P then Q.  Q.      => P       affirming the consequent  (INVALID)
    If P then Q.  not Q.  => not P   modus tollens             (valid)
    If P then Q.  not P.  => not Q   denying the antecedent    (INVALID)

Gemini supplies only the clause pairs — plausible P/Q with natural negations. It
never picks a form, so it cannot mislabel one. Every label here is correct by
construction, which is what makes this immune to the bias that produced the
fallacy defect.

Each clause pair emits all four forms: identical vocabulary and subject, differing
only in the logical move. Holding the topic constant strips it of predictive value
and forces the model to attend to the structure.

    python gen_formal.py --pairs 60 --out formal_generated.jsonl
"""

import argparse
import json
import random
import re

from google import genai
from google.genai import types

from lfx.vertex import MODEL, client

DOMAINS = ["medicine and diagnosis", "law and contracts", "engineering and safety",
           "everyday practical life", "business and economics", "science and research",
           "education and schooling", "history and politics"]

SCHEMA = {
    "type": "OBJECT",
    "properties": {"pairs": {"type": "ARRAY", "items": {
        "type": "OBJECT",
        "properties": {
            "p_pos": {"type": "STRING"}, "p_neg": {"type": "STRING"},
            "q_pos": {"type": "STRING"}, "q_neg": {"type": "STRING"},
            "subject": {"type": "STRING"},
        },
        "required": ["p_pos", "p_neg", "q_pos", "q_neg", "subject"]}}},
    "required": ["pairs"],
}

SYSTEM = """\
You write clause pairs for building conditional arguments. You do NOT write arguments.

Each pair is an antecedent P and a consequent Q where "if P then Q" is a plausible
claim a knowledgeable person might assert — a real-world regularity, rule, or
policy. Q should follow from P believably, WITHOUT being merely a restatement of it.

For each you give four clauses, all as plain declarative sentence fragments that
read naturally after "If ..." or "Then ...":
  p_pos  the antecedent               e.g. "the bridge is structurally unsound"
  p_neg  its natural negation         e.g. "the bridge is structurally sound"
  q_pos  the consequent               e.g. "it will be closed to traffic"
  q_neg  its natural negation         e.g. "it will stay open to traffic"

Rules:
- Negations must be IDIOMATIC, not "it is not the case that ...". Prefer an
  antonym or a natural negative phrasing.
- No pronouns whose referent sits outside the clause; each clause must stand alone.
- Lower case, no trailing punctuation.
- Crucially: P must NOT be the only possible cause of Q. A bridge can close for
  many reasons. This is what makes affirming the consequent genuinely fallacious
  rather than merely awkward.
- Vary subject matter within the batch.
"""

# Surface variation, so the model learns the form and not the phrasing.
# "Should {p}, {q}" was dropped: it needs subjunctive inversion ("Should the bridge
# BE unsound"), which the clause bank supplies in the indicative.
COND = ["If {p}, {q}.", "If {p}, then {q}.", "Whenever {p}, {q}.",
        "{q} whenever {p}.", "When {p}, {q}.", "Given that {p}, {q}."]
ASSERT = ["{x}.", "We know that {x}.", "In this case, {x}.", "It turns out that {x}.",
          "As it happens, {x}."]
CONCL = ["Therefore, {x}.", "So {x}.", "It follows that {x}.", "Which means {x}.",
         "Hence {x}.", "We can conclude that {x}."]

FORMS = [  # (form, argument_type, which clause is asserted, which is concluded)
    ("modus ponens", "deductive", "p_pos", "q_pos"),
    ("affirming the consequent", "deductive", "q_pos", "p_pos"),
    ("modus tollens", "deductive", "q_neg", "p_neg"),
    ("denying the antecedent", "deductive", "p_neg", "q_neg"),
]


def cap(s):
    s = s.strip().rstrip(".")
    return s[0].upper() + s[1:] if s else s


def build(pair, form, atype, assert_key, concl_key, rng):
    cond = rng.choice(COND).format(p=pair["p_pos"], q=pair["q_pos"])
    cond = cap(cond.rstrip(".")) + "."
    second = cap(rng.choice(ASSERT).format(x=pair[assert_key]).rstrip(".")) + "."
    concl_txt = rng.choice(CONCL).format(x=pair[concl_key])
    concl_txt = cap(concl_txt.rstrip(".")) + "."

    sentences = [cond, second, concl_txt]
    if rng.random() < 0.25:                      # sometimes lead with the conclusion
        lead = cap(re.sub(r"^(Therefore|So|It follows that|Which means|Hence|We can conclude that)[, ]*",
                          "", concl_txt).rstrip(".")) + "."
        sentences = [lead, cond, second]
        conclusion = lead
    else:
        conclusion = concl_txt

    return {
        "text": " ".join(sentences),
        "label": {
            "premises": [cond, second],
            "conclusion": cap(pair[concl_key]) + ".",
            "argument_type": atype,
            "form": form,
            "suppressed_premise": None,
        },
        "_conclusion_sentence": conclusion,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", type=int, default=60, help="clause pairs; each yields 4 passages")
    ap.add_argument("--out", default="data/interim/formal_generated.jsonl")
    ap.add_argument("--seed", type=int, default=20260902)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    cl = client()

    pairs, per_call = [], 8
    for i in range(0, args.pairs, per_call):
        dom = DOMAINS[(i // per_call) % len(DOMAINS)]
        n = min(per_call, args.pairs - i)
        resp = cl.models.generate_content(
            model=MODEL,
            contents=f"Write {n} clause pairs from: {dom}.",
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM, response_mime_type="application/json",
                response_schema=SCHEMA, temperature=1.0),
        )
        got = json.loads(resp.text)["pairs"]
        pairs += got
        print(f"  {dom:<26} +{len(got)}  ({len(pairs)}/{args.pairs})")

    rows = []
    for pi, pair in enumerate(pairs):
        for form, atype, ak, ck in FORMS:
            b = build(pair, form, atype, ak, ck, rng)
            rows.append({
                "id": f"formal_{len(rows) + 1:04d}",
                "category": f"formal_{form.replace(' ', '_')}",
                "source": f"template-constructed (clause pair {pi + 1}: {pair.get('subject', '')})",
                "text": b["text"],
                "label": b["label"],
                "form_source": "construction",
                "reviewed": False,
                "review_notes": "",
            })

    with open(args.out, "w") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"\nwrote {len(rows)} passages from {len(pairs)} clause pairs -> {args.out}")
    for form, *_ in FORMS:
        print(f"  {sum(1 for r in rows if r['label']['form'] == form):>4}  {form}")


if __name__ == "__main__":
    main()
