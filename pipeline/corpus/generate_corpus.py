"""Generate a stratified argument corpus with Gemini via Vertex AI.

Scales the seed set toward the 300-500 target by generating passages cell by cell
(form x domain) against an explicit target distribution, so `form` coverage is
designed rather than hoped for.

Passages only — no labels. `label_corpus.py` labels them blind afterwards, and
disagreement between the form a passage was generated to instantiate and the form
the labeler independently assigns becomes the human-review triage queue.

Usage:
    python generate_corpus.py --plan            # show the target distribution, generate nothing
    python generate_corpus.py --limit-cells 3   # smoke test
    python generate_corpus.py                   # full run, resumable
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
BATCH = 5           # passages per API call
WORKERS = 4

# Target counts per form. Floor of 20 keeps a stratified 70/15/15 split from
# handing the test set a single example of a class; the extras go to the forms
# that genuinely occur most often. `other` is held at ~15% as a real class —
# the seed corpus showed multi-step chained arguments are common in exactly the
# philosophical and historical material this tool is aimed at.
TARGETS = {
    "modus ponens": 26, "modus tollens": 20, "hypothetical syllogism": 20,
    "disjunctive syllogism": 20, "categorical syllogism": 25, "reductio ad absurdum": 20,
    "generalization": 25, "analogy": 20, "causal": 24, "sign": 20, "authority": 20,
    "affirming the consequent": 20, "denying the antecedent": 20,
    "ad hominem": 20, "hasty generalization": 20, "false dilemma": 20,
    "other": 60,
}

DOMAINS = ["philosophical", "biblical", "historical", "scientific",
           "legal", "everyday", "editorial"]

# Not every form sits naturally in every domain — sign reasoning is at home in
# medicine and detective work, not in scripture. Weights shape each form's spread.
W_DEFAULT = {"philosophical": 2, "biblical": 1, "historical": 2, "scientific": 2,
             "legal": 2, "everyday": 3, "editorial": 2}
W_FORMAL  = {"philosophical": 3, "biblical": 1, "historical": 1, "scientific": 2,
             "legal": 3, "everyday": 3, "editorial": 1}
W_CHAINED = {"philosophical": 4, "biblical": 2, "historical": 4, "scientific": 1,
             "legal": 2, "everyday": 1, "editorial": 2}
W_EMPIRIC = {"philosophical": 1, "biblical": 1, "historical": 2, "scientific": 4,
             "legal": 2, "everyday": 3, "editorial": 2}
W_RHETOR  = {"philosophical": 1, "biblical": 1, "historical": 2, "scientific": 1,
             "legal": 2, "everyday": 3, "editorial": 4}

WEIGHTS = {
    "modus ponens": W_FORMAL, "modus tollens": W_FORMAL,
    "hypothetical syllogism": W_FORMAL, "disjunctive syllogism": W_FORMAL,
    "categorical syllogism": W_FORMAL, "reductio ad absurdum": W_CHAINED,
    "generalization": W_EMPIRIC, "analogy": W_EMPIRIC, "causal": W_EMPIRIC,
    "sign": W_EMPIRIC, "authority": W_EMPIRIC,
    "affirming the consequent": W_FORMAL, "denying the antecedent": W_FORMAL,
    "ad hominem": W_RHETOR, "hasty generalization": W_RHETOR, "false dilemma": W_RHETOR,
    "other": W_CHAINED,
}

FORM_BRIEF = {
    "modus ponens": "If P then Q; P; therefore Q.",
    "modus tollens": "If P then Q; not Q; therefore not P.",
    "hypothetical syllogism": "If P then Q; if Q then R; therefore if P then R.",
    "disjunctive syllogism": "Either P or Q; not P; therefore Q. The disjunction must be genuinely exhaustive — otherwise you have written a false dilemma.",
    "categorical syllogism": "Quantified premises about classes: All A are B; s is an A; therefore s is a B. Use real quantifiers (all, some, no).",
    "reductio ad absurdum": "Assume a claim for the sake of argument, derive a contradiction or plain absurdity from it, and reject the claim.",
    "generalization": "From an observed sample to a broader population. The sample must be large or varied enough that the inference is reasonable, not hasty.",
    "analogy": "Two things share relevant features; one has a further property; conclude the other likely does too. Name the shared features that carry the inference.",
    "causal": "From evidence of correlation, mechanism, or controlled comparison to a causal conclusion. The conclusion must actually assert cause, not just association.",
    "sign": "An observed indicator is reliably associated with an underlying condition; conclude the condition is likely present. Sign is correlation-as-evidence, NOT a claim that the sign causes the condition.",
    "authority": "A qualified, relevant authority asserts a claim; conclude it is likely true. The authority must be genuinely relevant and its qualification stated.",
    "affirming the consequent": "If P then Q; Q; therefore P. Invalid. The passage must present it as though it follows.",
    "denying the antecedent": "If P then Q; not P; therefore not Q. Invalid. The passage must present it as though it follows.",
    "ad hominem": "Rejects a CLAIM OR ARGUMENT by attacking the character, motives, or circumstances of the person advancing it. There must be an actual claim being rejected — not merely an unflattering inference about someone's motives.",
    "hasty generalization": "Generalizes from a sample far too small or biased to support the conclusion. Keep the sample conspicuously thin (one or two cases).",
    "false dilemma": "Presents two options as exhaustive when others plainly exist, then eliminates one to force the other. The inference itself is valid; the FIRST PREMISE is what makes it a fallacy.",
    "other": "A multi-step argument that chains two or more DISTINCT inferences, so that no single named form describes it — for example a causal chain that terminates in a no-infinite-regress step, or a general principle plus a subsumption plus a normative conclusion. This is the shape of most real philosophical and political argument.",
}

DOMAIN_BRIEF = {
    "philosophical": "metaphysics, epistemology, ethics, philosophy of mind — in the register of a philosophy essay",
    "biblical": "theology, scripture, apologetics, church history — in the register of a serious theological argument, not a sermon",
    "historical": "political history, constitutional argument, historical causation — in the register of a historian or a statesman's address",
    "scientific": "empirical research, medicine, ecology, physics, methodology — in the register of a science writer",
    "legal": "statutory interpretation, precedent, liability, constitutional law — in the register of a legal brief or judicial opinion",
    "everyday": "practical reasoning about work, money, health, family, travel — plain contemporary prose",
    "editorial": "opinion writing on policy, technology, culture, education — in the register of a newspaper op-ed",
}

SYSTEM_INSTRUCTION = """\
You write short natural-language argument passages for a dataset used to train a
model that extracts argument structure.

Requirements for every passage:
- 2 to 5 sentences. One self-contained argument with premises and a conclusion.
- Written as ordinary prose a person would actually write. NOT a logic-textbook
  template with P and Q, and not labeled ("Premise 1:", "Conclusion:").
- NEVER name the argument form, and never use its vocabulary ("modus ponens",
  "this is a fallacy", "by analogy"). The structure must be implicit in the prose.
- No verbatim quotation of any real text. Original prose only.
- Each passage in a batch must be about a DIFFERENT subject from the others.
- Vary the surface: sometimes conclusion-first, sometimes conclusion-last,
  sometimes with a "therefore"/"so"/"which means", sometimes with none.

Style — this matters as much as the structure:
- Plain, level declarative prose. State claims; do not oversell them.
- Avoid emphatic adverbs and intensifiers entirely: no "undeniably", "decisively",
  "invariably", "demonstrably", "fundamentally", "consistently", "truly",
  "ultimately", "indeed", "clearly", "logically follows". They are a stylistic
  fingerprint, and a dataset full of them teaches the wrong cue.
- Vary sentence length. Not every passage should open with a general principle.

Structural fidelity — the passage must genuinely instantiate the requested form:
- Every step the form requires must be ACTUALLY STATED in the passage. For a form
  that requires affirming a conditional's antecedent, the passage must assert that
  antecedent outright, not merely gesture at evidence for it.
- Do not chain extra inferences on top of the requested form. One form, one
  argument — unless the requested form is "other", which is defined by chaining.

Where the requested form is a fallacy, the passage must actually commit that
fallacy while being presented as though the reasoning is sound. Do not signal it.
"""

SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "passages": {
            "type": "ARRAY",
            "items": {
                "type": "OBJECT",
                "properties": {
                    "topic": {"type": "STRING"},
                    "text": {"type": "STRING"},
                },
                "required": ["topic", "text"],
            },
        }
    },
    "required": ["passages"],
}

_lock = threading.Lock()


def plan():
    """Allocate each form's target across domains by weight. Largest-remainder,
    so the per-form totals come out exactly right."""
    cells = []
    for form, total in TARGETS.items():
        w = WEIGHTS[form]
        tw = sum(w.values())
        raw = {d: total * w[d] / tw for d in DOMAINS}
        base = {d: int(v) for d, v in raw.items()}
        short = total - sum(base.values())
        for d in sorted(DOMAINS, key=lambda d: raw[d] - base[d], reverse=True)[:short]:
            base[d] += 1
        for d in DOMAINS:
            if base[d]:
                cells.append((form, d, base[d]))
    return cells


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


def gen_cell(cl, form, domain, n, avoid):
    prompt = (
        f"Write {n} argument passages.\n\n"
        f"FORM each passage must instantiate: {form}\n"
        f"  {FORM_BRIEF[form]}\n\n"
        f"DOMAIN: {domain} — {DOMAIN_BRIEF[domain]}\n\n"
        + (f"Avoid these subjects, already used: {', '.join(avoid[:14])}.\n\n" if avoid else "")
        + "Give each passage a short `topic` (3-6 words) naming its subject."
    )
    resp = cl.models.generate_content(
        model=MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_INSTRUCTION,
            response_mime_type="application/json",
            response_schema=SCHEMA,
            temperature=1.0,          # diversity matters more than determinism here
            top_p=0.95,
        ),
    )
    return json.loads(resp.text)["passages"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/interim/generated_corpus.jsonl")
    ap.add_argument("--plan", action="store_true", help="print the target distribution and exit")
    ap.add_argument("--limit-cells", type=int, help="only run the first N cells (smoke test)")
    args = ap.parse_args()

    cells = plan()
    if args.plan:
        print(f"{len(cells)} cells, {sum(c[2] for c in cells)} passages\n")
        for form, total in TARGETS.items():
            spread = [f"{d[:4]}:{n}" for f, d, n in cells if f == form]
            print(f"  {form:<26} {total:>3}   {' '.join(spread)}")
        return

    done_counts, seen_text, seen_topics = {}, [], {}
    if os.path.exists(args.out):
        for line in open(args.out):
            r = json.loads(line)
            key = (r["intended_form"], r["domain"])
            done_counts[key] = done_counts.get(key, 0) + 1
            seen_text.append(norm(r["text"]))
            seen_topics.setdefault(r["intended_form"], []).append(r.get("topic", ""))

    todo = []
    for form, domain, n in cells:
        need = n - done_counts.get((form, domain), 0)
        if need > 0:
            todo.append((form, domain, need))

    have = sum(done_counts.values())
    print(f"target {sum(c[2] for c in cells)} passages; {have} already generated; "
          f"{sum(t[2] for t in todo)} to go across {len(todo)} cells")
    if args.limit_cells:
        todo = todo[: args.limit_cells]
    if not todo:
        return

    cl = client()
    out = open(args.out, "a")
    counter = {f: sum(1 for t in seen_text) for f in TARGETS}   # ids only need uniqueness
    nid = [have]
    stats = {"written": 0, "dupes": 0, "errors": 0}

    def work(cell):
        form, domain, need = cell
        got = 0
        while got < need:
            ask = min(BATCH, need - got)
            try:
                with _lock:
                    avoid = list(seen_topics.get(form, []))
                passages = gen_cell(cl, form, domain, ask, avoid)
            except Exception as exc:
                with _lock:
                    stats["errors"] += 1
                print(f"  ! {form}/{domain}: {type(exc).__name__} {exc}"[:160], file=sys.stderr)
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
                        "id": f"gen_{slug}_{nid[0]:04d}",
                        "category": f"generated_{domain}",
                        "source": f"generated ({domain}, target form: {form})",
                        "text": text,
                        "topic": p.get("topic", ""),
                        "intended_form": form,
                        "domain": domain,
                    }, ensure_ascii=False) + "\n")
                    out.flush()
                    seen_text.append(norm(text))
                    seen_topics.setdefault(form, []).append(p.get("topic", ""))
                    stats["written"] += 1
                    got += 1
                print(f"  {form[:22]:<22} {domain[:12]:<12} {got}/{need}")

    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        list(pool.map(work, todo))
    out.close()
    print(f"\nwrote {stats['written']} passages -> {args.out} "
          f"({stats['dupes']} near-duplicates dropped, {stats['errors']} cell errors)")


if __name__ == "__main__":
    main()
