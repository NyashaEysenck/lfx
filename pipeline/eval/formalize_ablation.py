"""Does formalising first actually help, or does it move the error one step back?

Phase C proposes that the model should emit a logical skeleton and let lfx.formal
derive the form, rather than naming the form directly. The case for it is that
`modus ponens` and `affirming the consequent` differ by one symbol, and code
cannot misread a symbol. The case against is that a bad formalisation is just a
wrong answer arriving by a longer route.

That is an empirical question and it is cheap to settle, so it gets settled before
any training compute is spent on it.

  DIRECT       one call: here is an argument, name its form from the enum
  FORMALIZE    one call: here is an argument, give me its skeleton
               then lfx.formal derives the form by rule

Same model, same items, same temperature. The only difference is what the model is
asked to produce -- BOTH arms are run fresh by `--model`, so the comparison is
within one model rather than against a stored run by a different one.

That matters because the first version of this reused the stored gemini-3.1-pro
formalisations as the FORMALIZE arm and only ran DIRECT. It measured 0.988 vs
1.000 and looked like a null result. It was: a frontier model has no headroom on
constructed formal items, so the comparison cannot move. The proposal is about
what a WEAK model should be asked to produce -- the 3B scores 0.708 here -- so the
ablation has to be run on models with room to fail.

SELECTION CAVEAT, stated because it cuts in favour of the hypothesis being tested:
Tier 1 items survive construction only if a conclusion-supplied formalisation
derives the target form, so items whose prose resists formalisation were dropped.
The rejection rate was low (2-6%) and the surviving set is therefore mildly biased
toward formalisability. A fair reading treats the FORMALIZE arm as an upper bound.

    python pipeline/eval/formalize_ablation.py --split tier1_formal --limit 200
"""

import argparse
import collections
import concurrent.futures
import json
import os
import random
import sys
import time

from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()

from lfx.formal import classify
from lfx.schema import FORMS

LOCATION = os.environ.get("GOOGLE_CLOUD_LOCATION", "global")

DIRECT_PROMPT = """Identify the logical form of this argument.

Choose exactly one from: {forms}

Report the structure the author actually used, including any error in it. Do not \
repair a flawed argument into a valid one.

ARGUMENT
{text}"""

DIRECT_SCHEMA = {
    "type": "OBJECT",
    "properties": {"form": {"type": "STRING", "enum": FORMS}},
    "required": ["form"],
}

FORMALIZE_PROMPT = """Formalise the logical skeleton of this argument.

Use canonical labels (P, Q, R for propositions; S, M, P for categorical terms) and \
put the prose in the mapping, not in the formulas.

WORKED EXAMPLE
  argument: "If the vessel sustained hull damage, the bilge alarm would have
             triggered. The bilge alarm did not trigger. So there was no hull
             damage."
  output:   kind        = "propositional"
            mapping     = {{"P": "the vessel sustained hull damage",
                           "Q": "the bilge alarm triggered"}}
            premises    = ["P -> Q", "~Q"]
            conclusion  = "~P"

The formulas contain ONLY labels and the symbols ~ | & -> and parentheses. Never \
put English words in `premises` or `conclusion`; the English belongs in `mapping`.

Propositional connectives: ~ (not), -> (if...then), | (or), & (and).
Categorical propositions: "All S are P", "No S are P", "Some S are P", \
"Some S are not P", "S is a M".

Choose kind="propositional" when the argument turns on conditionals, negations or \
disjunctions; kind="categorical" when it turns on quantified class membership; \
kind="none" when its force does not come from its logical shape at all.

Report the structure the author actually used, including any error in it. Do not \
repair a flawed argument into a valid one.

ARGUMENT
{text}"""

FORMALIZE_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "kind": {"type": "STRING", "enum": ["propositional", "categorical", "none"]},
        "mapping": {"type": "OBJECT", "properties": {}},
        "premises": {"type": "ARRAY", "items": {"type": "STRING"}},
        "conclusion": {"type": "STRING"},
    },
    "required": ["kind", "premises", "conclusion"],
}


def call(cl, model, prompt, schema, retries=8):
    delay = 4.0
    for attempt in range(retries):
        try:
            resp = cl.models.generate_content(
                model=model, contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=0.0, response_mime_type="application/json",
                    response_schema=schema))
            body = (resp.text or "").strip()
            if not body:
                raise ValueError("empty body")
            return json.loads(body)
        except Exception as e:
            if attempt == retries - 1 or not any(
                    t in str(e) for t in ("429", "RESOURCE_EXHAUSTED", "503",
                                          "UNAVAILABLE", "empty body")):
                raise
            time.sleep(delay + random.random() * 2)
            delay = min(delay * 2, 60)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="tier1_formal")
    ap.add_argument("--model", default="gemini-3.1-pro-preview")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--out", default="results/ablation_direct.jsonl")
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(f"data/splits/{args.split}.jsonl")]
    if args.limit:
        # stratified, so a truncated run still covers every class
        by = collections.defaultdict(list)
        for r in rows:
            by[r["target_form"]].append(r)
        per = max(1, args.limit // len(by))
        rows = [r for v in by.values() for r in v[:per]]

    cl = genai.Client(vertexai=True, project=os.environ["GOOGLE_CLOUD_PROJECT"],
                      location=LOCATION)
    results, formal, raw, errors = {}, {}, {}, []

    def work(r):
        d = call(cl, args.model,
                 DIRECT_PROMPT.format(forms=", ".join(FORMS), text=r["text"]),
                 DIRECT_SCHEMA)
        f = call(cl, args.model, FORMALIZE_PROMPT.format(text=r["text"]),
                 FORMALIZE_SCHEMA)
        return r["id"], d["form"], classify(f), f

    done = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(work, r): r for r in rows}
        for fut in concurrent.futures.as_completed(futures):
            done += 1
            try:
                rid, form, derived, fz = fut.result()
                results[rid] = form
                formal[rid] = derived
                raw[rid] = fz
            except Exception as e:
                errors.append(f"{type(e).__name__}: {str(e)[:80]}")
            if done % 50 == 0:
                print(f"  {done}/{len(rows)}", flush=True)

    with open(args.out, "w") as fh:
        for rid, form in results.items():
            fh.write(json.dumps({"id": rid, "direct_form": form,
                                 "derived_form": formal.get(rid),
                                 "formalization": raw.get(rid)},
                                ensure_ascii=False) + "\n")

    scored = [r for r in rows if r["id"] in results]
    direct_ok = sum(1 for r in scored if results[r["id"]] == r["target_form"])
    formal_ok = sum(1 for r in scored if formal.get(r["id"]) == r["target_form"])
    unformalizable = sum(1 for r in scored if formal.get(r["id"]) is None)
    n = len(scored)
    print(f"\n{n} items scored, {len(errors)} errors, model {args.model}\n")
    print(f"  {'':28} {'DIRECT':>8} {'FORMALIZE':>10}")
    print(f"  {'overall':28} {direct_ok / n:8.3f} {formal_ok / n:10.3f}")
    print()
    for form in sorted({r["target_form"] for r in scored}):
        sub = [r for r in scored if r["target_form"] == form]
        d = sum(1 for r in sub if results[r["id"]] == form)
        f = sum(1 for r in sub if formal.get(r["id"]) == r["target_form"])
        print(f"  {form:28} {d:3}/{len(sub):<4} {f:6}/{len(sub):<4}")

    # McNemar: only the disagreements carry information
    b = sum(1 for r in scored if results[r["id"]] == r["target_form"]
            and formal.get(r["id"]) != r["target_form"])
    c = sum(1 for r in scored if results[r["id"]] != r["target_form"]
            and formal.get(r["id"]) == r["target_form"])
    tot = b + c
    z = (abs(b - c) - 1) / tot ** 0.5 if tot else 0.0
    print(f"\n  direct-only correct {b}, formalize-only correct {c}, "
          f"disagreements {tot}, McNemar z={z:.2f}")
    print(f"  formalisations lfx.formal could not read: {unformalizable}/{n} "
          f"= {unformalizable / n:.1%}  (these count as FORMALIZE errors)")

    print("\n  where DIRECT goes wrong:")
    conf = collections.Counter((r["target_form"], results[r["id"]]) for r in scored
                               if results[r["id"]] != r["target_form"])
    for (g, p), k in conf.most_common(8):
        print(f"    {k:3}  {g}  ->  {p}")


if __name__ == "__main__":
    sys.exit(main())
