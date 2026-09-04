"""Generate prose to fill classes that sit under the per-class cap.

Reads the work order from `merge_corpus.py` — the classes short of the cap — and
generates natural-prose passages for each. NOT templates: the v2 run measured
template-trained forms at 18/18 on templated text and 2/6 on natural prose, so
templates give perfect labels and teach the wrong thing.

Passages only. `label_corpus.py` labels them blind afterwards and `triage.py`
compares the intended form against that blind label, so a passage that fails to
instantiate its target form surfaces as a disagreement rather than entering the
corpus mislabelled.

    python pipeline/corpus/gen_fill.py --plan
    python pipeline/corpus/gen_fill.py
"""

import argparse
import json
import os
import re
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from difflib import SequenceMatcher

from google.genai import types

from lfx.vertex import MODEL, client

# From the Phase 2 work order.
NEED = {
    "application of generalization": 42,
    "other": 36,
    "generalization": 33,
    "hasty generalization": 24,
    "analogy": 24,
    "authority": 23,
    "hypothetical syllogism": 22,
    "reductio ad absurdum": 22,
    "categorical syllogism": 13,
    "denying the antecedent": 11,
    "affirming the consequent": 10,
    "disjunctive syllogism": 10,
    "inference to the best explanation": 8,
}

DOMAINS = ["philosophical", "biblical", "historical", "scientific",
           "legal", "everyday", "editorial"]

FORM_BRIEF = {
    "application of generalization": (
        "From a population-level generalization DOWN to a particular case. "
        "CRITICAL: the generalization must be statistical, not universal — use "
        "'most', 'the majority of', 'usually', 'typically', 'roughly 80 percent'. "
        "NEVER 'all' or 'every'. The conclusion must be hedged: 'probably', "
        "'likely', 'in all likelihood'. Example shape: most members of a group "
        "have a property; this individual is a member; so this individual "
        "probably has it. If you write 'all' or an unhedged conclusion you have "
        "written a categorical syllogism instead, which is the wrong form."),
    "other": (
        "A passage chaining two or more DIFFERENT KINDS of inference, so no single "
        "form covers it. The test is the kind of step, not the number of steps: a "
        "long causal chain ending in a causal claim is still just causal. Examples "
        "that qualify: a causal chain that then terminates in a no-infinite-regress "
        "step; a statistical generalization applied to a case and THEN used to "
        "ground a normative 'therefore we should'; eliminating alternatives and "
        "THEN inferring a best explanation and THEN drawing a policy conclusion."),
    "generalization": (
        "From an observed SAMPLE to a broader population. The sample must be large "
        "or varied enough that the inference is reasonable, not hasty — say so in "
        "the passage (a number, a span of time, a range of places)."),
    "hasty generalization": (
        "Generalizes from a sample far too small or too biased to support it. Keep "
        "the sample conspicuously thin — one or two cases — and state the sweeping "
        "conclusion as though it followed."),
    "analogy": (
        "Two things share relevant features; one has a further property; conclude "
        "the other likely does too. NAME the shared features that carry the "
        "inference, otherwise it reads as a bare assertion."),
    "authority": (
        "A qualified, RELEVANT authority asserts a claim, so it is probably true. "
        "State the qualification and make it germane to the claim. This is the "
        "legitimate form, not an appeal to an irrelevant or unqualified authority."),
    "hypothetical syllogism": (
        "If P then Q; if Q then R; therefore if P then R. The conclusion must "
        "itself be CONDITIONAL — that is what distinguishes this from modus ponens."),
    "reductio ad absurdum": (
        "Assume a claim for the sake of argument, derive a contradiction or a plain "
        "absurdity from it, and reject the claim. The assumption must be stated as "
        "an assumption."),
    "categorical syllogism": (
        "Quantified premises about CLASSES: all/no/some A are B; s is an A; so s is "
        "a B. Use real quantifiers. The conclusion follows with certainty — do NOT "
        "hedge it with 'probably', which would make it an application of a "
        "generalization instead."),
    "denying the antecedent": (
        "If P then Q; not P; therefore not Q. Invalid. Present it as though it "
        "follows. P must not be the only route to Q, which is what makes it fail."),
    "affirming the consequent": (
        "If P then Q; Q; therefore P. Invalid. Present it as though it follows. "
        "P must not be the only possible cause of Q."),
    "disjunctive syllogism": (
        "Either P or Q; not P; therefore Q. The disjunction must be GENUINELY "
        "exhaustive — no third option — otherwise you have written a false dilemma."),
    "inference to the best explanation": (
        "An observation needs explaining; a hypothesis would explain it; no rival "
        "explains it as well; therefore it is probably true. This includes "
        "reasoning from an INDICATOR to the condition it evidences — symptoms to a "
        "diagnosis, residue to a past event. The indicator does not CAUSE the "
        "condition, it is evidence of it."),
}

SYSTEM = """\
You write short natural-language argument passages for a dataset used to train a
model that extracts argument structure.

Requirements for every passage:
- 2 to 5 sentences. One self-contained argument with premises and a conclusion.
- Ordinary prose a person would actually write. NOT a logic-textbook template with
  P and Q, and not labelled ("Premise 1:", "Conclusion:").
- NEVER name the argument form, and never use its vocabulary ("modus ponens",
  "this is a fallacy", "by analogy"). The structure must be implicit.
- No verbatim quotation of any real text. Original prose only.
- Each passage in a batch must be about a DIFFERENT subject.
- Vary the surface: sometimes conclusion-first, sometimes last; vary connectives;
  sometimes no connective at all.

Style — this matters as much as the structure:
- Plain, level declarative prose. State claims; do not oversell them.
- Avoid emphatic adverbs entirely: no "undeniably", "decisively", "invariably",
  "demonstrably", "fundamentally", "consistently", "truly", "ultimately",
  "clearly", "logically follows". They are a stylistic fingerprint.
- Vary sentence length.

Structural fidelity — the passage must genuinely instantiate the requested form:
- Every step the form requires must be ACTUALLY STATED, not gestured at.
- Do not chain extra inferences on top of the requested form, unless the requested
  form is "other", which is defined by chaining across kinds.

Where the requested form is a fallacy, the passage must actually commit it while
being presented as though the reasoning is sound. Do not signal it.
"""

SCHEMA = {
    "type": "OBJECT",
    "properties": {"passages": {"type": "ARRAY", "items": {
        "type": "OBJECT",
        "properties": {"topic": {"type": "STRING"}, "text": {"type": "STRING"}},
        "required": ["topic", "text"]}}},
    "required": ["passages"],
}

_lock = threading.Lock()


def norm(s):
    return re.sub(r"[^a-z0-9 ]", "", s.lower())


def too_similar(text, existing, thresh=0.75):
    n = norm(text)
    head = " ".join(n.split()[:10])
    for e in existing:
        if head and head == " ".join(e.split()[:10]):
            return True
        if SequenceMatcher(None, n, e).ratio() > thresh:
            return True
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/interim/fill_corpus.jsonl")
    ap.add_argument("--plan", action="store_true")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--batch", type=int, default=6)
    args = ap.parse_args()

    # spread each form's need across domains
    cells = []
    for form, n in NEED.items():
        per = max(1, n // len(DOMAINS))
        left = n
        for i, d in enumerate(DOMAINS):
            take = min(per if i < len(DOMAINS) - 1 else left, left)
            if take > 0:
                cells.append((form, d, take))
                left -= take
        if left > 0:
            cells.append((form, DOMAINS[0], left))

    if args.plan:
        print(f"{len(cells)} cells, {sum(c[2] for c in cells)} passages")
        for form, n in NEED.items():
            spread = [f"{d[:4]}:{k}" for f, d, k in cells if f == form]
            print(f"  {form:<34}{n:>4}   {' '.join(spread)}")
        return

    done, seen_text, seen_topics = {}, [], {}
    if os.path.exists(args.out):
        for line in open(args.out):
            r = json.loads(line)
            done[r["intended_form"]] = done.get(r["intended_form"], 0) + 1
            seen_text.append(norm(r["text"]))
            seen_topics.setdefault(r["intended_form"], []).append(r.get("topic", ""))

    todo = []
    for form, dom, n in cells:
        got = done.get(form, 0)
        want = NEED[form]
        if got < want:
            todo.append((form, dom, min(n, want - got)))
            done[form] = got + min(n, want - got)

    print(f"target {sum(NEED.values())} passages across {len(NEED)} forms; "
          f"{len(todo)} cells to run\n")

    cl = client()
    out = open(args.out, "a")
    stats = {"written": 0, "dupes": 0, "errors": 0}
    nid = [sum(1 for _ in open(args.out)) if os.path.exists(args.out) else 0]

    def work(cell):
        form, domain, need = cell
        got = 0
        while got < need:
            ask = min(args.batch, need - got)
            with _lock:
                avoid = list(seen_topics.get(form, []))[:14]
            prompt = (f"Write {ask} argument passages.\n\n"
                      f"FORM each must instantiate: {form}\n  {FORM_BRIEF[form]}\n\n"
                      f"DOMAIN: {domain}\n\n"
                      + (f"Avoid these subjects, already used: {', '.join(avoid)}.\n\n"
                         if avoid else "")
                      + "Give each passage a short `topic` (3-6 words).")
            try:
                resp = cl.models.generate_content(
                    model=MODEL, contents=prompt,
                    config=types.GenerateContentConfig(
                        system_instruction=SYSTEM, response_mime_type="application/json",
                        response_schema=SCHEMA, temperature=1.0, top_p=0.95))
                passages = json.loads(resp.text)["passages"]
            except Exception as exc:  # noqa: BLE001
                with _lock:
                    stats["errors"] += 1
                print(f"  ! {form}/{domain}: {type(exc).__name__}"[:120], file=sys.stderr)
                return
            with _lock:
                for p in passages:
                    if got >= need:
                        break
                    text = p["text"].strip()
                    if too_similar(text, seen_text):
                        stats["dupes"] += 1
                        continue
                    nid[0] += 1
                    slug = re.sub(r"[^a-z]+", "_", form.lower()).strip("_")
                    out.write(json.dumps({
                        "id": f"fill_{slug}_{nid[0]:04d}",
                        "category": f"generated_{domain}",
                        "source": f"generated fill ({domain}, target form: {form})",
                        "text": text, "topic": p.get("topic", ""),
                        "intended_form": form, "domain": domain,
                    }, ensure_ascii=False) + "\n")
                    out.flush()
                    seen_text.append(norm(text))
                    seen_topics.setdefault(form, []).append(p.get("topic", ""))
                    stats["written"] += 1
                    got += 1
                print(f"  {form[:30]:<30} {domain[:11]:<11} {got}/{need}")

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        list(pool.map(work, todo))
    out.close()
    print(f"\nwrote {stats['written']} -> {args.out} "
          f"({stats['dupes']} near-duplicates dropped, {stats['errors']} cell errors)")


if __name__ == "__main__":
    main()
