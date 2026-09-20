"""Tier 1b: classes whose label can be SPECIFIED but not symbolically verified.

Six valid inductive forms, which no public dataset covers, plus two fallacies that
are structural enough to construct honestly.

Every corpus in this space annotates FALLACIES. LOGIC, MAFALDA, Argotario, the
climate sets -- all of them label what went wrong. Nothing labels an argument that
reasons well from a sample, an analogy, or an authority. So `generalization`,
`application of generalization`, `inference to the best explanation`, `analogy`,
`causal` and `authority` have between 0 and 3 real-prose test records each, and no
source to draw more from.

Tier 1 cannot reach them either. Its guarantee comes from lfx.formal deriving the
form symbolically, and these forms are not deductively valid -- there is no
skeleton whose shape makes an argument an induction rather than a bad deduction.
The whole point of an inductive argument is that the conclusion outruns the
premises.

So the guarantee here is weaker, and named differently rather than blurred into
Tier 1:

  Tier 1   label specified by construction, VERIFIED SYMBOLICALLY by lfx.formal
  Tier 1b  label specified by construction, VERIFIED BY CONSENSUS of two models
           from different families, each labelling blind from the full 22-form
           enum with no knowledge of what was specified

EQUIVOCATION NEEDS A DIFFERENT CHECK, and finding that out is itself a result.
Asked to classify a constructed equivocation from the 22-form enum, both verifiers
failed on every item -- calling it `application of generalization` or `modus
ponens`, i.e. reading the surface structure and not noticing that a term had
shifted sense. The renders were not at fault; one ran "entry by means of an
instrument" as a burglary tool in one premise and a forged financial instrument in
the next, which is as clean an equivocation as a textbook prints.

An open-ended 22-way question has almost no sensitivity here, so it cannot verify
the class. A targeted one does: the renderer declares which term it equivocated on,
a verifier is asked only "does any term carry two senses, and which", and the item
survives if the verifier independently names the same term. Both checks are run and
both recorded, because the gap between them measures something worth knowing --
how much of a fallacy's detectability depends on being asked the right question.

Consensus removes idiosyncratic error, not systematic error. If every model shares
a blind spot, this preserves it -- and one is already documented in this project,
where Gemini systematically misreads `straw man`. Two independent agreements is
the strongest claim available without a human, and it is reported as that rather
than as gold.

What is specified here is the INFERENTIAL MOVE, not the wording. The renderer is
told the shape of the reasoning and invents everything else, so the label follows
from the construction rather than from anyone's reading of the result.

TWO FALLACIES ARE INCLUDED, and the line matters. Constructing an `ad hominem` or
a `straw man` produces a caricature rather than a specimen, because what makes
those fallacious is the discourse context -- who is being answered, and what they
actually said. `hasty generalization` and `equivocation` are not like that. Both
are structural: the first is `generalization` with the sample made too small to
bear the leap, the second is one term carrying two senses across the premises.
"Two rude waiters in Paris, so Parisians are rude" is a real instance, not a
parody of one. So these two are constructible and the other five informal
fallacies are not.

They are also the two classes with nowhere else to turn. LOGIC's 387 unused
`faulty generalization` rows cannot fill the first: import_logic.py removed that
mapping because LOGIC uses the class for any bad inductive leap while ours is the
narrow one, and Gemini disagreed with the human label on 21 of 33 sampled. MAFALDA
has both classes but is CC BY-SA, so lifting rows out of it would put a ShareAlike
obligation on a 1435-record benchmark for the sake of 35.

    python pipeline/corpus/build_inductive.py --per-form 4
    python pipeline/corpus/build_inductive.py --per-form 40 --apply
"""

import argparse
import collections
import concurrent.futures
import json
import os
import random
import sys
import threading
import time

os.environ.setdefault("GOOGLE_CLOUD_PROJECT", "agentic-school-506719")

from google import genai
from google.genai import types

from lfx.schema import FORMS

LOCATION = os.environ.get("GOOGLE_CLOUD_LOCATION", "global")
RENDER_MODEL = "gemini-3.8-flash"
# Two verifiers from different families. Same-family agreement would mostly
# measure shared training, which is the correlation this is trying to avoid.
VERIFIERS = ["gemini-2.5-flash", "gemini-3.1-pro-preview"]

# The inferential move, described structurally. Each says what must be reasoned
# FROM and TO, and what must NOT be present -- the negative clause is what keeps
# neighbouring classes apart. `generalization` and `application of generalization`
# are the same premise material in opposite directions, and without the exclusion
# a renderer will write both at once.
# form -> (argument_type, specification). hasty generalization is a failed
# induction so the schema requires `inductive`; equivocation is written here as a
# syllogism whose middle term shifts sense, which is deductive in shape.
SPECS = {
    "generalization": ("inductive",
        "Reason FROM several specific observed instances sharing a property TO a "
        "general claim about the wider class they belong to. The sample must be "
        "described as reasonably sized or representative, so the leap is warranted. "
        "Do NOT then apply the general claim back to any individual case."),
    "application of generalization": ("inductive",
        "Reason FROM an established general claim about a class, plus the fact that "
        "a specific individual belongs to that class, TO a probable conclusion about "
        "that individual. The general claim must be stated as already known or "
        "well established -- do NOT derive it from a sample in this argument. The "
        "conclusion must be hedged as probable, not certain."),
    "inference to the best explanation": ("inductive",
        "Reason FROM a puzzling observation, THROUGH the comparative merits of rival "
        "explanations, TO accepting the one that best accounts for it. At least one "
        "competing explanation must be named and set aside as a worse fit. Do NOT "
        "present it as a simple cause-and-effect claim."),
    "analogy": ("inductive",
        "Reason FROM a similarity between two specific cases in named relevant "
        "respects, and from one case having some further property, TO the other case "
        "probably having that property too. Both cases must be particular, not "
        "classes. Do NOT reason from a sample to a population."),
    "causal": ("inductive",
        "Reason FROM evidence that one factor produces another -- correlation plus a "
        "plausible mechanism, or an intervention -- TO the claim that it does cause "
        "it. Do NOT present rival explanations for comparison, and do NOT infer the "
        "cause merely from temporal order."),
    "authority": ("inductive",
        "Reason FROM the considered judgement of a relevant, identified expert or "
        "body TO accepting the claim they assert. The expertise must be germane to "
        "the claim. Do NOT supply the underlying evidence itself -- the argument's "
        "weight rests on who is saying it."),
    "hasty generalization": ("inductive",
        "Reason FROM a sample that is EXPLICITLY too small or unrepresentative -- "
        "one or two cases, or a plainly skewed source -- TO a sweeping claim about "
        "the whole class. The inadequacy of the sample must be visible in the text "
        "(state how few cases, or how they were selected). The conclusion must be "
        "stated with unwarranted confidence, not hedged."),
    "equivocation": ("deductive",
        "Write an argument that LOOKS valid but turns on one word or phrase carrying "
        "TWO DIFFERENT SENSES in different premises. State the term once in its "
        "first sense and once in its second, so that the conclusion only follows if "
        "the two senses are confused. Do not flag the shift or explain it -- the "
        "argument must read as though it goes through."),
}

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

RENDER_PROMPT = """Write a short argument that performs exactly this inferential move.

THE MOVE
{spec}

Set it in {domain}, write it as {register}, ordered {order}.

Hard requirements:
- The argument must be a genuine instance of exactly this move, written the way a
  real author would write it -- not a textbook parody of it.
- Perform this move and no other. Do not add a second inferential step.
- Never name the form of reasoning, and avoid textbook giveaways ("by analogy",
  "generalising from", "the best explanation is", "appeal to authority").
- 2 to 5 sentences. Natural writing, not a logic exercise.

Return JSON with: text (the argument prose), premises (each premise as it appears
in the text), conclusion (the conclusion as it appears in the text). If the move
turns on a term carrying two senses, also return equivocal_term: that term exactly
as it appears in the text."""

EQUIVOCAL_PROMPT = """Does this argument rely on a word or phrase that carries two \
DIFFERENT meanings in different parts of it?

If so, name that single word or phrase exactly as it appears, and give the two \
senses. If no term shifts meaning, return term as an empty string.

ARGUMENT
{text}"""

EQUIVOCAL_SCHEMA = {
    "type": "OBJECT",
    "properties": {"term": {"type": "STRING"},
                   "sense_a": {"type": "STRING"},
                   "sense_b": {"type": "STRING"}},
    "required": ["term"],
}

CLASSIFY_PROMPT = """Identify the form of reasoning this argument uses.

Choose exactly one from: {forms}

Report the structure the author actually used, including any error in it.

ARGUMENT
{text}"""

RENDER_SCHEMA = {
    "type": "OBJECT",
    "properties": {"text": {"type": "STRING"},
                   "premises": {"type": "ARRAY", "items": {"type": "STRING"}},
                   "conclusion": {"type": "STRING"},
                   # equivocation only: what the renderer shifted, so a targeted
                   # verifier can be checked against it rather than trusted
                   "equivocal_term": {"type": "STRING"}},
    "required": ["text", "premises", "conclusion"],
}
CLASSIFY_SCHEMA = {
    "type": "OBJECT",
    "properties": {"form": {"type": "STRING", "enum": FORMS}},
    "required": ["form"],
}


def call(cl, model, prompt, schema, temperature, retries=6):
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
                raise ValueError("empty body")
            return json.loads(body)
        except Exception as e:
            if attempt == retries - 1 or not any(
                    t in str(e) for t in ("429", "RESOURCE_EXHAUSTED", "503",
                                          "UNAVAILABLE", "empty body")):
                raise
            time.sleep(delay + random.random() * 2)
            delay = min(delay * 2, 45)


def build_one(cl, args, form, i, seed):
    rng = random.Random(seed)
    arg_type, spec_text = SPECS[form]
    spec = dict(spec=spec_text, domain=rng.choice(DOMAINS),
                register=rng.choice(REGISTERS), order=rng.choice(ORDERS))
    for attempt in range(3):
        r = call(cl, args.render_model, RENDER_PROMPT.format(**spec), RENDER_SCHEMA, 1.0)
        if (r.get("text") or "").strip():
            break
        time.sleep(2 * (attempt + 1))
    else:
        raise ValueError("renderer returned empty text three times")

    votes = {}
    for m in args.verifiers:
        votes[m] = call(cl, m, CLASSIFY_PROMPT.format(forms=", ".join(FORMS),
                                                      text=r["text"]),
                        CLASSIFY_SCHEMA, 0.0)["form"]
    unanimous = len(set(votes.values())) == 1 and next(iter(votes.values())) == form

    targeted = None
    if form == "equivocation":
        # The open-ended question does not find these; a targeted one does. Both
        # are kept so the gap between them stays visible.
        term = (r.get("equivocal_term") or "").strip().lower()
        found = {}
        for m in args.verifiers:
            got = call(cl, m, EQUIVOCAL_PROMPT.format(text=r["text"]),
                       EQUIVOCAL_SCHEMA, 0.0)
            got_term = (got.get("term") or "").strip().lower()
            # a head-noun match is enough: "instrument" vs "an instrument"
            hit = bool(term) and bool(got_term) and (
                term in got_term or got_term in term)
            found[m] = {"term": got.get("term"), "senses":
                        [got.get("sense_a"), got.get("sense_b")], "matches": hit}
        targeted = {"renderer_term": r.get("equivocal_term"), "verifiers": found,
                    "unanimous": all(v["matches"] for v in found.values())}
        unanimous = targeted["unanimous"]
    return {"id": f"ind_{form.replace(' ', '_')}_{i:03d}",
            "text": r["text"], "origin": "constructed", "tier": "1b",
            "target_form": form,
            "spec": spec_text,
            "verifiers": votes,
            "targeted_check": targeted,
            "open_ended_unanimous": len(set(votes.values())) == 1
                                    and next(iter(votes.values())) == form,
            "unanimous": unanimous,
            "label": {"premises": r["premises"], "conclusion": r["conclusion"],
                      "argument_type": arg_type, "form": [form],
                      "suppressed_premise": None},
            "certain": []}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-form", type=int, default=40)
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--out")
    ap.add_argument("--rejects")
    ap.add_argument("--forms", nargs="*")
    ap.add_argument("--render-model", default=RENDER_MODEL)
    ap.add_argument("--verifiers", nargs="*", default=VERIFIERS)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--apply", action="store_true")
    args = ap.parse_args()

    stamp = time.strftime("%Y%m%d_%H%M%S")
    out_path = args.out or f"data/interim/inductive_tier1b_{stamp}.jsonl"
    rej_path = args.rejects or f"data/interim/inductive_tier1b_{stamp}_rejects.jsonl"

    cl = genai.Client(vertexai=True, project=os.environ["GOOGLE_CLOUD_PROJECT"],
                      location=LOCATION)
    targets = args.forms or list(SPECS)
    tasks = [(f, i) for f in targets for i in range(args.per_form)]
    kept, rejected, errors = [], [], []
    lock = threading.Lock()
    out_fh = open(out_path, "w") if args.apply else None
    rej_fh = open(rej_path, "w") if args.apply else None

    done = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        futs = {pool.submit(build_one, cl, args, f, i,
                            args.seed * 7919 + hash((f, i)) % 10**6): (f, i)
                for f, i in tasks}
        for fut in concurrent.futures.as_completed(futs):
            form, i = futs[fut]
            done += 1
            try:
                rec = fut.result()
            except Exception as e:
                errors.append(f"{form} #{i}: {type(e).__name__}: {str(e)[:90]}")
                continue
            with lock:
                target = kept if rec["unanimous"] else rejected
                target.append(rec)
                fh = out_fh if rec["unanimous"] else rej_fh
                if fh:
                    fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    fh.flush()
            if done % 25 == 0:
                print(f"  {done}/{len(tasks)}  kept {len(kept)}  "
                      f"rejected {len(rejected)}  errors {len(errors)}", flush=True)

    total = len(kept) + len(rejected)
    print(f"\n{len(kept)} unanimous, {len(rejected)} not, {len(errors)} errors "
          f"({len(kept) / max(total, 1):.0%} agreement)")
    print(f"{'form':34} {'kept':>5} {'rej':>4}")
    for form in targets:
        k = sum(1 for x in kept if x["target_form"] == form)
        r = sum(1 for x in rejected if x["target_form"] == form)
        print(f"  {form:34} {k:4} {r:4}")

    eq = [x for x in kept + rejected if x["target_form"] == "equivocation"]
    if eq:
        oe = sum(1 for x in eq if x["open_ended_unanimous"])
        tg = sum(1 for x in eq if (x.get("targeted_check") or {}).get("unanimous"))
        print(f"\nequivocation, open-ended vs targeted verification:")
        print(f"  open-ended 22-way question : {oe}/{len(eq)} found")
        print(f"  targeted 'which term shifts': {tg}/{len(eq)} found")

    print("\nwhere the verifiers went instead (rejected items):")
    conf = collections.Counter()
    for x in rejected:
        for m, v in x["verifiers"].items():
            if v != x["target_form"]:
                conf[(x["target_form"], v)] += 1
    for (g, p), n in conf.most_common(10):
        print(f"  {n:3}  {g}  ->  {p}")
    if errors:
        print(f"\n{len(errors)} errors, first few:")
        for e in errors[:3]:
            print(f"  {e}")

    if args.apply:
        out_fh.close(); rej_fh.close()
        print(f"\nwrote {out_path} ({len(kept)}) and {rej_path} ({len(rejected)})")
    else:
        print("\nDRY RUN — re-run with --apply to write")


if __name__ == "__main__":
    sys.exit(main())
