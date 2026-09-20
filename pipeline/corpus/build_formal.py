"""Tier 1: build formal-class items whose label is true by construction.

The gap this fills: 17 of the 22 classes have at most 3 real-prose test records
and three have none at all, so the project's headline number is measured on five
informal fallacies and says nothing about whether the model can find a modus
tollens. Labelling found prose would close that gap only as well as the labeller
reasons, and the premise of this phase is that language models reason badly.

So the label is not judged, it is SPECIFIED. We fix the skeleton first --

    premises   P -> Q,  ~Q
    conclusion ~P

-- and only then ask a model to invent concrete propositions and write them as
prose. Nobody decides afterwards what form the passage has; it has the form it was
built from. That removes the labeller from the ground-truth path entirely for the
nine structurally-determined classes.

The remaining risk is not the label but the PROSE: a renderer can be handed a
modus tollens and write something that does not express it. So every item is
round-tripped. A DIFFERENT model, told nothing about the target, reads the prose
and formalises it; lfx.formal derives a form from that formalisation; the item is
kept only if the derived form matches the one it was built from. Two models from
different families have to agree, through a symbolic bottleneck, on structure
neither of them was told.

That check is deliberately lopsided. A good item rejected because the verifier
formalised it badly costs yield, which is cheap. A bad item accepted would need
the verifier to produce a formalisation that derives the target form from prose
that does not express it -- far less likely, and the thing worth protecting.

VERIFICATION RUNS TWICE, and the first version of this got it wrong. A single
blind pass conflates two questions: did the renderer write prose that expresses
the skeleton, and can a strong model READ that prose correctly? Filtering on the
second discards the hardest items, which is the opposite of what a benchmark
wants. The measurement that caught it: `affirming the consequent` was rejected at
30% against 0-10% elsewhere, and the verifier had derived `modus ponens` -- its
valid twin -- in every one of those cases.

The cause is POSITIONAL. The verifier takes the last claim to be the conclusion,
so when an author states it first the argument is read backwards. Pooled over two
runs of 30 per form:

    conclusion stated later    1/82 misread   1.2%
    conclusion stated first   13/37 misread    35%      Fisher exact p = 5e-07

Two things the first run got wrong, kept here because they are the kind of error
this file exists to prevent. The rate looked like 56%; a second run of the same
size gave 16%, and 35% is the pooled figure -- a single run of 18 items could not
resolve it. And it is NOT a bias toward validity: run 1 showed only invalid ->
valid, but run 2 also produced `modus ponens` read as `affirming the consequent`
and `modus tollens` as `denying the antecedent`. Swapping premise for conclusion
moves in whichever direction the swap happens to land. The model is not protecting
validity, it is using position as a cue for which claim is the conclusion.

Those items are precisely the ones worth keeping.

So: stage B, with the conclusion supplied, is the QUALITY FILTER -- it asks only
whether the prose expresses the skeleton. Stage A, blind, is recorded as a
DIFFICULTY ANNOTATION. Every item ships knowing whether a frontier model read it
correctly unaided.

`begging the question` is deliberately absent from the skeletons below. The checker
can see it -- a conclusion restating a premise is circular whatever else its shape
suggests -- but only when the restatement is SYNTACTIC. In prose the repetition is
paraphrased, and noticing that two differently worded sentences express the same
proposition is a semantic judgement, which is what Tier 1 exists to avoid. So Tier 1
covers 8 classes, not 9; begging the question stays a Tier 2/3 class.

What Tier 1 certifies is the FORM and the ARGUMENT TYPE. The premise and
conclusion spans come from the renderer and carry no such guarantee; they are
ordinary model output and should not be treated as gold for premise F1.

    python pipeline/corpus/build_formal.py --per-form 2 --dry-run
    python pipeline/corpus/build_formal.py --per-form 25 --apply
"""

import argparse
import collections
import json
import os
import random
import sys
import time

os.environ.setdefault("GOOGLE_CLOUD_PROJECT", "agentic-school-506719")

from google import genai
from google.genai import types

from lfx.formal import classify

# The Gemini 3.x models are served from `global`, not a regional endpoint: asking
# us-central1 for them returns 404 NOT_FOUND, which reads like a permissions
# problem and is not one.
LOCATION = os.environ.get("GOOGLE_CLOUD_LOCATION", "global")

RENDER_MODEL = "gemini-3.8-flash"
# A different family for the round trip. Same-model verification would mostly
# confirm that the renderer is self-consistent, which is not the question.
VERIFY_MODEL = "gemini-3.1-pro-preview"

# Skeletons, in the canonical-label language lfx.formal reads. Both valid forms
# and formal fallacies: the task is naming the shape, not endorsing it.
SKELETONS = {
    "modus ponens":             ("propositional", ["P -> Q", "P"], "Q"),
    "modus tollens":            ("propositional", ["P -> Q", "~Q"], "~P"),
    "affirming the consequent": ("propositional", ["P -> Q", "Q"], "P"),
    "denying the antecedent":   ("propositional", ["P -> Q", "~P"], "~Q"),
    "hypothetical syllogism":   ("propositional", ["P -> Q", "Q -> R"], "P -> R"),
    "disjunctive syllogism":    ("propositional", ["P | Q", "~P"], "Q"),
    "reductio ad absurdum":     ("propositional", ["P -> (Q & ~Q)"], "~P"),
    "categorical syllogism":    ("categorical", ["All M are P", "All S are M"], "All S are P"),
}

# Variation knobs. Constructed data's failure mode is uniformity -- a model can
# score well by learning the generator's habits rather than the logic, which is
# exactly what test_gen (0.812) flatters relative to real prose (0.765). Domain
# and register are varied so surface cues cannot carry the signal.
DOMAINS = ["medicine", "climate science", "criminal law", "software engineering",
           "monetary policy", "sports journalism", "archaeology", "nutrition",
           "urban planning", "astronomy", "employment disputes", "agriculture",
           "maritime safety", "pharmacology", "civil engineering", "epidemiology",
           "contract disputes", "wildlife conservation", "education policy",
           "aviation incidents"]
REGISTERS = ["a newspaper editorial", "a technical report", "conversational speech",
             "an academic paper", "a legal submission", "a blog post",
             "a committee minute", "a letter to a colleague"]
ORDERS = ["premises first, conclusion last",
          "conclusion stated first, then the reasons",
          "one premise, then the conclusion, then the remaining premise"]

RENDER_PROMPT = """You are writing a short argument that instantiates an exact logical skeleton.

SKELETON ({kind})
  premises:   {premises}
  conclusion: {conclusion}

Invent concrete, plausible {kind_word} for each label, set in {domain}, and write \
the argument as {register}, ordered {order}.

Hard requirements:
- The prose must express EXACTLY this skeleton: every premise, and that conclusion.
- Do not add extra inferential steps, hedges that weaken the conclusion, or \
qualifiers like "probably" or "suggests" unless the skeleton contains them.
- Never name the form, and do not use textbook giveaways ("it follows validly", \
"this commits the fallacy of", "modus", "syllogism", "affirming", "denying").
- Do not use the labels themselves ({labels}) in the prose.
- 2 to 5 sentences. Natural writing, not a logic exercise.

Return JSON with: mapping (label -> the proposition or term it stands for), \
text (the argument prose), premises (each premise as it appears in the text), \
conclusion (the conclusion as it appears in the text)."""

VERIFY_PROMPT = """Formalise the logical skeleton of this argument.

Use canonical labels (P, Q, R for propositions; S, M, P for categorical terms) and \
put the prose in the mapping, not in the formulas.

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

RENDER_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "mapping": {"type": "OBJECT", "properties": {}},
        "text": {"type": "STRING"},
        "premises": {"type": "ARRAY", "items": {"type": "STRING"}},
        "conclusion": {"type": "STRING"},
    },
    "required": ["text", "premises", "conclusion"],
}

VERIFY_GIVEN_PROMPT = """Formalise the logical skeleton of this argument.

The author's conclusion is: {conclusion}

Everything else asserted is a premise. Use canonical labels (P, Q, R for \
propositions; S, M, P for categorical terms) and put the prose in the mapping, not \
in the formulas.

Propositional connectives: ~ (not), -> (if...then), | (or), & (and).
Categorical propositions: "All S are P", "No S are P", "Some S are P", \
"Some S are not P", "S is a M".

Report the structure the author actually used, including any error in it. Do not \
repair a flawed argument into a valid one, and do not swap the premises and the \
conclusion to make the inference come out valid.

ARGUMENT
{text}"""

VERIFY_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "kind": {"type": "STRING", "enum": ["propositional", "categorical", "none"]},
        "mapping": {"type": "OBJECT", "properties": {}},
        "premises": {"type": "ARRAY", "items": {"type": "STRING"}},
        "conclusion": {"type": "STRING"},
    },
    "required": ["kind", "premises", "conclusion"],
}


def call(cl, model, prompt, schema, temperature, retries=5):
    """One structured call, with backoff.

    429 RESOURCE_EXHAUSTED is routine on the newer models rather than exceptional,
    and an exhausted call returns an empty body that json.loads reports as a parse
    error -- which looks like a schema bug and is not one. Both are retried.
    """
    delay = 4.0
    for attempt in range(retries):
        try:
            resp = cl.models.generate_content(
                model=model, contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=temperature, response_mime_type="application/json",
                    response_schema=schema))
            body = (resp.text or "").strip()
            if not body:
                raise ValueError(f"empty body (finish_reason="
                                 f"{resp.candidates[0].finish_reason if resp.candidates else '?'})")
            return json.loads(body)
        except Exception as e:
            transient = any(t in str(e) for t in ("429", "RESOURCE_EXHAUSTED", "503",
                                                  "UNAVAILABLE", "empty body"))
            if attempt == retries - 1 or not transient:
                raise
            time.sleep(delay + random.random())
            delay *= 2
    raise RuntimeError("unreachable")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-form", type=int, default=25)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", default="data/interim/formal_tier1.jsonl")
    ap.add_argument("--rejects", default="data/interim/formal_tier1_rejects.jsonl")
    ap.add_argument("--forms", nargs="*", help="subset of forms; default all")
    ap.add_argument("--render-model", default=RENDER_MODEL)
    ap.add_argument("--verify-model", default=VERIFY_MODEL)
    ap.add_argument("--sleep", type=float, default=0.0,
                    help="pause between items, to stay under a rate limit")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--apply", action="store_true")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    cl = genai.Client(vertexai=True, project=os.environ["GOOGLE_CLOUD_PROJECT"],
                      location=LOCATION)
    targets = args.forms or list(SKELETONS)
    kept, rejected = [], []
    stats = collections.Counter()
    # Written as they are produced, not buffered to the end: a long run that dies
    # on the last call should not cost every item before it.
    out_fh = open(args.out, "w") if args.apply else None
    rej_fh = open(args.rejects, "w") if args.apply else None

    for form in targets:
        kind, prems, concl = SKELETONS[form]
        labels = sorted({t for p in prems + [concl] for t in p.replace("(", " ")
                         .replace(")", " ").replace("~", " ").replace("->", " ")
                         .replace("|", " ").replace("&", " ").split()
                         if t.isalnum() and t[0].isupper()})
        for i in range(args.per_form):
            spec = dict(kind=kind, premises=", ".join(prems), conclusion=concl,
                        kind_word="propositions" if kind == "propositional" else "terms",
                        domain=rng.choice(DOMAINS), register=rng.choice(REGISTERS),
                        order=rng.choice(ORDERS), labels=", ".join(labels))
            try:
                # temperature high on render for variety, 0 on verify for a
                # reproducible check
                r = call(cl, args.render_model, RENDER_PROMPT.format(**spec),
                         RENDER_SCHEMA, 1.0)
                # stage A: blind. Records whether a frontier model reads it
                # correctly with no help -- difficulty, not quality.
                blind = call(cl, args.verify_model, VERIFY_PROMPT.format(text=r["text"]),
                             VERIFY_SCHEMA, 0.0)
                # stage B: conclusion supplied. This is the quality filter: does
                # the PROSE express the skeleton?
                v = call(cl, args.verify_model,
                         VERIFY_GIVEN_PROMPT.format(text=r["text"],
                                                    conclusion=r["conclusion"]),
                         VERIFY_SCHEMA, 0.0)
                if args.sleep:
                    time.sleep(args.sleep)
            except Exception as e:
                stats["api error"] += 1
                print(f"  [{form}] api error: {type(e).__name__}: {str(e)[:120]}")
                continue

            derived = classify(v)
            blind_derived = classify(blind)
            rec = {"id": f"formal_{form.replace(' ', '_')}_{i:03d}",
                   "text": r["text"], "origin": "constructed", "tier": 1,
                   "target_form": form,
                   "skeleton": {"kind": kind, "premises": prems, "conclusion": concl},
                   "render_mapping": r.get("mapping"),
                   "verifier": {"model": args.verify_model, "formalization": v,
                                "derived_form": derived},
                   "unaided": {"formalization": blind, "derived_form": blind_derived,
                               "correct": blind_derived == form},
                   "label": {"premises": r["premises"], "conclusion": r["conclusion"],
                             "argument_type": "deductive", "form": [form],
                             "suppressed_premise": None},
                   "certain": ["form", "argument_type"]}
            if derived == form:
                kept.append(rec)
                stats[f"{form}: kept"] += 1
                fh = out_fh
            else:
                rec["reject_reason"] = f"round trip derived {derived!r}, built as {form!r}"
                rejected.append(rec)
                stats[f"{form}: rejected"] += 1
                fh = rej_fh
            if fh:
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
                fh.flush()
        print(f"  {form:28} {stats[f'{form}: kept']:3} kept / "
              f"{stats[f'{form}: kept'] + stats[f'{form}: rejected']:3}", flush=True)

    print(f"\n{len(kept)} kept, {len(rejected)} rejected "
          f"({len(kept) / max(len(kept) + len(rejected), 1):.0%} yield)")
    print(f"{'form':28} {'kept':>5} {'rej':>4}   unaided-correct (difficulty)")
    for form in targets:
        k, r = stats[f"{form}: kept"], stats[f"{form}: rejected"]
        sub = [x for x in kept if x["target_form"] == form]
        u = sum(1 for x in sub if x["unaided"]["correct"])
        pct = f"{u}/{len(sub)} = {u / len(sub):.0%}" if sub else "-"
        print(f"  {form:28} {k:4} {r:4}   {pct}")
    u = sum(1 for x in kept if x["unaided"]["correct"])
    print(f"\n  frontier model reading unaided: {u}/{len(kept)} = "
          f"{u / max(len(kept), 1):.0%} correct on the kept set")
    if stats["api error"]:
        print(f"  api errors: {stats['api error']}")

    if kept:
        print("\nsample:")
        s = kept[0]
        print(f"  [{s['target_form']}] {s['text'][:220]}")
    if rejected:
        print("\nrejected sample (what the verifier saw instead):")
        s = rejected[0]
        print(f"  built as {s['target_form']}, derived {s['verifier']['derived_form']!r}")
        print(f"  {s['text'][:220]}")
        print(f"  verifier: {s['verifier']['formalization'].get('premises')} "
              f"|- {s['verifier']['formalization'].get('conclusion')}")

    if args.apply:
        out_fh.close()
        rej_fh.close()
        print(f"\nwrote {args.out} ({len(kept)}) and {args.rejects} ({len(rejected)})")
    else:
        print("\nDRY RUN — re-run with --apply to write")


if __name__ == "__main__":
    sys.exit(main())
