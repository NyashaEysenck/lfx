# Project: Logical Form Extractor — Fine-Tuning Pipeline

## Goal
Learn the end-to-end ML fine-tuning pipeline (data → format → train → eval → deploy) by building a small, useful tool: a fine-tuned LLM that extracts the logical structure of a natural-language argument into a fixed JSON schema. Scope is intentionally small (extraction only, no evaluation/validity-checking layer — that's a deliberate v1 cut, may revisit later).

Interests driving topic choice: logic, philosophy, history, biblical history. Source material draws on Aquinas, Pascal, Plato, Descartes, Hume, C.S. Lewis, Lincoln-Douglas debates, the Federalist Papers, MLK, plus inductive-reasoning and fallacy examples.

## Task Definition
**Input:** a natural-language argument passage (1 paragraph).
**Output:** structured JSON matching the schema below.

This is a **pure extraction/parsing task** — deliberately decoupled from *evaluating* the argument (validity/soundness/strength), which was cut from v1 as a separate, harder reasoning task. May be added later as a downstream rule-based checker, not as something the fine-tuned model itself judges.

## Schema (v1.1)
```json
{
  "premises": ["...", "..."],
  "conclusion": "...",
  "argument_type": "deductive" | "inductive",
  "form": // valid deductive
          "modus ponens" | "modus tollens" | "hypothetical syllogism" | "disjunctive syllogism"
          | "categorical syllogism" | "reductio ad absurdum"
          // inductive
          | "generalization" | "analogy" | "causal" | "sign" | "authority"
          // formal fallacies
          | "affirming the consequent" | "denying the antecedent"
          // informal fallacies
          | "ad hominem" | "hasty generalization" | "false dilemma"
          | "other",
  "suppressed_premise": "..." | null
}
```
- Conceptual scope modeled on Duke's "Introduction to Logic and Critical Thinking" (Think Again) specialization: identifying arguments, breaking into premises/conclusion, suppressed premises, deductive validity forms, the five common inductive argument forms (generalization, analogy, causal, sign, authority), and fallacies.
- Fallacies are folded into the `form` field rather than a separate field.
- **v1.1 change (2026-09-02):** the pilot labeling run put 53% of the seed corpus (18/34) on `"other"`.
  The original enum covered only *propositional* deductive forms — even "All men are mortal;
  Socrates is a man; so Socrates is mortal" had no slot. Added `categorical syllogism` and
  `reductio ad absurdum` (valid forms), plus `ad hominem`, `hasty generalization`, and
  `false dilemma` (the three informal fallacies in the seed set that had no enum value).
- **Known open issue:** multi-step passages that chain several distinct inferences (Aquinas'
  Five Ways, Lincoln, MLK, Federalist 10/51) have no single correct `form` value and still land
  on `"other"`. A single-value `form` field cannot represent them. Revisit if `"other"` remains
  a large share after the enum extension — options are a separate `fallacy` field, a list-valued
  `form`, or accepting `"other"` as a legitimate minority class.

## Compute / Tooling Decisions
- **Local machine:** MacBook, M4 Pro, 24GB unified memory (MPS). Good for local *inference* on the final merged model (via MLX or llama.cpp), not ideal for training (MPS training support is weaker than CUDA).
- **Training:** Google Colab (free T4 to debug pipeline cheaply) → GCP (~$140 credits, via Colab Enterprise or a plain GCP VM with T4/L4) for the real run.
- **Google Colab CLI** (`pip install google-colab-cli`, Google official, released June 2026): drives a
  remote Colab runtime from the terminal — no browser notebook, sessions kept alive up to 24h.
  Auths off the same ADC as Vertex. Key commands: `colab new --gpu T4 -s NAME`, `colab install PKG`,
  `colab upload/download LOCAL REMOTE`, `colab exec -s NAME -f script.py`, `colab stop`.
  Colab compute bills through Colab (free tier / Pro), NOT the GCP credits — the free-tier/GCP
  split in the line above is unchanged.
- **Local env:** `.venv/` on Python 3.12 (Homebrew). Deps: `google-genai`, `python-dotenv`.
- **Framework:** Unsloth (QLoRA/LoRA) — CUDA-only, fast iteration, minimal boilerplate.
- **Base model:** Qwen2.5-1.5B-Instruct or 3B-Instruct (small enough for cheap LoRA runs, strong enough to bootstrap from).
- **Labeling:** Gemini (`gemini-2.5-flash`) via **Vertex AI + Application Default Credentials**, not
  an API key. NOTE: the `GEMINI_API_KEY` in `.env` is a Cloud-console key restricted to
  `generativelanguage.googleapis.com`, which returns 429 "prepayment credits depleted" — that
  surface bills against a prepaid balance separate from the project's Cloud credits. The
  `aiplatform` (Vertex) surface bills the ordinary Cloud way and works. Use Vertex; the key is unused.
  Used to auto-label the raw argument corpus into the schema above; human review/correction pass afterward (recommended: review all of it at this small scale, or minimum 30-40%).

## Data
- Seed corpus: `seed_corpus.jsonl` — 34 raw (unlabeled) argument passages across categories:
  - Philosophical (10): Aquinas' Five Ways (x3), Pascal's Wager, Kalam cosmological, Anselm's ontological argument, Epicurus (problem of evil), Plato's Euthyphro, Descartes' Cogito, Hume on induction
  - Biblical/apologetic (5): Lewis's Trilemma, design argument, argument from religious experience, moral argument, argument from prophecy
  - Historical (4): Lincoln (Declaration/slavery), Federalist No. 10, Federalist No. 51, MLK's Letter from Birmingham Jail
  - Inductive, one per form (5): generalization, analogy, causal, sign, authority
  - Fallacies (5): affirming the consequent, denying the antecedent, ad hominem, hasty generalization, false dilemma
  - Editorials (2), classical syllogisms (2)
- These are condensed paraphrases (not verbatim quotes) of public-domain source arguments, plus some original illustrative examples for the inductive/fallacy categories — deliberately short and clean so an extraction model can learn structure without needing to trim long rhetorical prose.
- **Target full corpus size: 300–500 examples.** The seed set of 34 is meant to pilot the labeling pipeline end-to-end (catch schema/prompt bugs cheaply) before scaling up.
- Split plan: ~70% train / 15% val / 15% test, stratified so all `form` values appear in each split.

## Pipeline / Build Sequence
1. ~~Environment setup~~ (pending: wire up Colab + GCP credits + Vertex AI access)
2. ~~Source collection~~ — DONE: `seed_corpus.jsonl` (34 examples)
3. **Auto-labeling** (NEXT STEP) — write a script that calls Gemini via Vertex AI with the schema + a few hand-labeled few-shot examples, loops over `seed_corpus.jsonl`, and outputs candidate JSON labels alongside the raw text for review.
4. Human review pass — correct Gemini's labels (recommend reviewing all 34 in the pilot batch).
5. Scale up corpus to 300-500 examples (more sourcing + labeling + review), then split train/val/test.
6. Baseline eval — prompt the *un-fine-tuned* base model (few-shot) on the test set, score with field-level structured metrics (JSON validity, premise/conclusion match, form classification accuracy).
7. LoRA fine-tune on Colab (Unsloth).
8. Post-tune eval — same test set, same metrics, direct before/after comparison. This comparison is the core deliverable of the learning exercise.
9. Package: merge LoRA weights, build a minimal CLI or Gradio tool, runnable locally on the M4 Pro (via MLX or llama.cpp) so the tool is actually usable day-to-day on real reading material.

## Status / Where We Left Off
Steps 1-3 done. Step 4 (human review) is next.

Files:
- `fewshot.jsonl` — 3 hand-labeled few-shot examples (modus ponens / generalization /
  affirming the consequent). Written fresh, NOT drawn from the seed corpus, so all 34
  corpus passages stay uncontaminated.
- `label_corpus.py` — Vertex + ADC, `gemini-2.5-flash`, temperature 0, schema enforced via
  Gemini structured-output mode so `form`/`argument_type` cannot come back off-enum.
  Resumable (skips ids already in the output), retries with backoff.
  Run: `.venv/bin/python label_corpus.py [--limit N] [--force]`
- `labeled_candidates.jsonl` — 34 candidate labels, each with `reviewed: false` and
  `review_notes: ""` for the human pass.
- `labeled_candidates.v1schema.jsonl` — the first run under the original enum, kept for comparison.
- `review_stats.py` — form distribution + auto-flags rows needing attention
  (type/form mismatch, <2 premises, empty conclusion, `form: "other"`).

- `labeled_reviewed.jsonl` — all 34 after the review pass. **Mixed provenance, carried per record
  in a `reviewed_by` field:** 13 are `human` (phil_001-010, bib_001-003); 21 are `claude-opus-5`.
  The model-reviewed 21 are NOT human-verified gold — they are a second model's pass over a first
  model's output. Treat them as provisional, and re-check them before they anchor the step-8
  eval, or those numbers partly measure agreement with Claude rather than with you.
- Review UI: https://claude.ai/code/artifact/e2e082c3-7010-484a-8492-057203cfcd9d
  (all 34 labels live in its `reviews` collection; pull them with `read_db`).

### Step 6-8 results (2026-09-02, Qwen2.5-1.5B-Instruct + LoRA r=16, 3 epochs, T4)

End-to-end form accuracy (schema-invalid counted as wrong — the honest number):

| test set  | base  | tuned | delta  |
|-----------|-------|-------|--------|
| test_real | 0.029 | 0.324 | +0.294 |
| test_gen  | 0.121 | 0.259 | +0.138 |

argument_type accuracy on test_real: 0.560 -> 0.923. Suppressed-premise
present/absent: 0.480 -> 0.769. Conclusion similarity: 0.868 -> 0.940.
Premise F1 REGRESSED on test_real: 0.778 -> 0.713.

**Defect FIXED by training longer (6 epochs).** The tuned model had been writing
`form` values into `argument_type` (`"argument_type": "modus tollens"`) — every
schema failure it made. Diagnosis was undertraining, not a structural problem:
3 epochs was not enough for a 1.5B model on 284 examples to learn that
`argument_type` is a two-value field distinct from `form`. No constrained decoding
needed.

  field-swap rate    base    3 epochs   6 epochs
  test_real          21%     24%        3%
  test_gen           22%     47%        5%

### 6-epoch results (the current best)

| metric (end-to-end)    | test_real base -> e6 | test_gen base -> e6 |
|------------------------|----------------------|---------------------|
| FORM ACC (end-to-end)  | 0.029 -> 0.529       | 0.121 -> 0.655      |
| schema valid           | 0.735 -> 0.971       | 0.655 -> 0.948      |
| argument_type accuracy | 0.560 -> 0.939       | 0.579 -> 0.891      |
| premise F1             | 0.778 -> 0.833       | 0.832 -> 0.845      |
| conclusion similarity  | 0.868 -> 0.951       | 0.835 -> 0.869      |

The 3-epoch premise-F1 regression on test_real (-0.065) reversed at 6 epochs (+0.055).

**Watch this:** at 3 epochs test_real improved MORE than test_gen (+0.294 vs +0.138);
at 6 epochs test_gen leads (+0.534 vs +0.500), a 0.126 gap in favour of the synthetic
distribution. That reversal is the start of the expected overfit-to-Gemini-prose
pattern. Both still improved, so 6 epochs is a win — but do not push epochs further
without growing test_real past 34 examples, or the gap becomes unmeasurable noise.

Adapters on the Colab VM: `lora_out` (3 epochs), `lora_e6` (6 epochs, current best).
NOTE these live on an ephemeral runtime — download or re-train before relying on them.

**Scorer caveat:** `form acc | schema ok` is conditional on schema validity and so
flatters a model that fails schema on its hard cases. `FORM ACC (end-to-end)` is
the number to quote. Both are reported.

**Sample sizes are small** (34 real / 58 generated). Per-form cells hold 1-5
examples, so the per-form table is directional at best, not measurement.

### Step 9 DONE — packaged and running locally (2026-09-02)

Pipeline: `merge_adapter.py` (LoRA -> plain weights, fp32 on CPU, with a
weight-delta assertion so a no-op merge cannot pass silently) -> `mlx_lm convert`
-> `logicalform` CLI.

| artifact           | size   | forms correct /34 | notes                     |
|--------------------|--------|-------------------|---------------------------|
| GPU fp16 (adapter) | 3.1 GB | 18                | the Colab number          |
| `mlx_model_8bit`   | 1.5 GB | 16                | CLI default; schema 1.000 |
| `mlx_model_4bit`   | 847 MB | 15                | 1 example below 8-bit     |

8-bit is the default: same speed class (~0.6 s/passage, ~215 tok/s), perfect schema
validity, better suppressed-premise accuracy. Quantization loss is 2-3 examples out
of 34 — real in direction, but NOT resolvable at this sample size. Do not quote it
as a precise figure.

Usage: `./logicalform "passage"` · `pbpaste | ./logicalform` · `--repl` · `--json`

### CRITICAL DEFECT — the model repairs fallacies into valid forms

Reproduced on every fallacy with a valid twin:

| passage            | gold                     | model says            |
|--------------------|--------------------------|-----------------------|
| rain / sidewalk    | affirming the consequent | modus ponens          |
| study / exam       | denying the antecedent   | modus tollens         |
| budget / bankruptcy| false dilemma            | disjunctive syllogism |
| Socrates           | categorical syllogism    | modus ponens          |

For a tool meant to support critical reading this is the worst failure mode: it
certifies invalid arguments as valid, confidently.

**Traceable cause.** The LABELER had this exact bias (triage found false dilemma ->
disjunctive syllogism on 10/20). The precedence-rule fix overcorrected (disjunctive
syllogism fell 20/20 -> 12/20), and `split_corpus.py` then routed all 83 contested
records into TRAIN. So the training set still teaches the confusion. Defect path:
labeler prompt -> labels -> training data -> model.

**Fix candidates, in order:** (1) hand-label the fallacy classes rather than trusting
Gemini on them — there are only ~120 such records and they are the highest-value
labels in the corpus; (2) upweight or oversample fallacy examples in training;
(3) reconsider folding fallacies into `form` at all — a separate `fallacy` field
would stop the two from competing for one slot (this was option 2 in the v1.1
schema decision, deferred).

### v2 corpus + retrain (2026-09-02)

Corpus grew 400 -> 840 by adding two sources with labels Gemini did NOT choose:
- `logic_labeled.jsonl` 413 real passages, HUMAN form labels from LOGIC
  (Jin et al. 2022, `data/logic/`). Only ad hominem + false dilemma map cleanly.
- `formal_generated.jsonl` 256 + `disjunctive_generated.jsonl` 64, template-built,
  labels correct by construction.
Merged and capped by `merge_corpus.py` (cap 60 on imported classes); 293 surplus
human-labeled records held out as `extra_test_human.jsonl`.
Splits: train 592 / val 124 / test_gen 124 / test_real 34 / extra_test_human 293.

**Headline result — 293 real passages, human labels, never trained on:**

| metric                    | v2    |
|---------------------------|-------|
| FORM ACC (end-to-end)     | 0.775 |
| schema valid              | 0.952 |
| argument_type accuracy    | 0.864 |
| ad hominem                | 171/225 (76%) |
| false dilemma             | 56/68  (82%) |

This is now the most trustworthy metric in the project: n=293, real prose, human
labels. `test_real` (n=34) stays useful for the non-fallacy forms but holds only ONE
example each of the fallacy classes, so it cannot measure them — it read 0.529 flat
across v1 and v2 for that reason, NOT because nothing improved.

### KEY FINDING — real data generalizes, templates do not

Controlled comparison from the same training run:

| form                     | trained on      | template phrasing | natural prose |
|--------------------------|-----------------|-------------------|---------------|
| affirming the consequent | templates       | 10/10             | 1/2           |
| denying the antecedent   | templates       | 8/8               | 1/4           |
| false dilemma            | real (LOGIC)    | —                 | 10/12         |

Templates scored 18/18 on templated text and 2/6 on prose: the model learned the
TEMPLATE, not the form. The risk was flagged when the approach was proposed and it
materialised. Classes backed by real passages generalised to real passages.

**Implication for the next round:** stop adding synthetic data. Affirming-the-
consequent and denying-the-antecedent need a REAL-PROSE source; LOGIC buries them
in its `fallacy of logic` class unlabelled. Candidates not yet examined: Argotario
(1,344 crowdsourced, 5 fallacy types), MAFALDA (unified taxonomy, span-level).
Verify class EXTENSIONS against passages before mapping — see the faulty
generalization error above.

Diagnostic cases after v2: denying the antecedent PASS, categorical syllogism PASS,
affirming the consequent FAIL (-> modus ponens), false dilemma FAIL (-> disjunctive
syllogism). 2 of 4 fixed.

**Measurement trap hit and recorded:** comparing `preds_e6_gen.jsonl` against the NEW
`test_gen.jsonl` produced JSON-parse 0.048 — an artifact of the test set changing
between runs, not a real score. Predictions are only comparable against the exact
split file they were generated from. Re-generate baselines when splits change.

### Scope limits worth remembering (2026-09-03)

The model classifies 17 forms and knows FIVE fallacies: affirming the consequent,
denying the antecedent, ad hominem, hasty generalization, false dilemma. Anything
else — equivocation, ad populum, straw man, circular reasoning, slippery slope,
appeal to emotion, false cause, red herring — has NO representation, and the model
does not fall back to `other`: it names the valid form the argument resembles.
Verified: three textbook equivocations all returned `modus tollens`.

Equivocation is additionally hard in principle — the structure is valid and the flaw
is semantic (a word shifts meaning between premises), so a structural classifier
cannot see it. LOGIC has 49 equivocation records in `data/logic/` if this is ever
worth pursuing, plus ~1,950 more across 9 other unmapped fallacy classes.

**type/form consistency.** `form` implies `argument_type`; the two contradicted each
other in 4 of 840 training records (Gemini slips), and the model learned the pairing
was possible — it emitted "deductive · authority" in use. `fix_type_form.py` repairs
those 4 (takes effect at next retrain). `logicalform.py` now flags any contradictory
output at inference time, in both human and `--json` modes, since an impossible
pairing reliably indicates the model is unsure of the form.

**Practical heuristic from real use:** when the form label is wrong, the
`suppressed_premise` usually reads as nonsense. A "false dilemma" call came with
"saving costs and being healthier are the only two possible outcomes" — a
conjunction of benefits, not alternatives. The field is a useful self-check.

**Next: step 5, scale corpus to 300-500 and split.** Before that, spot-check the two
model-reviewed form changes (`ind_causal_001` generalization→causal, `fallacy_005`
disjunctive syllogism→false dilemma) and the one flagged judgment call
(`editorial_002` ad hominem→other).

---

## Schema v1.2 — align with Duke "Think Again" (planned, 2026-09-04)

The v1 enum was built from a misattribution: the brief credited Duke with "the five
common inductive forms (generalization, analogy, causal, sign, authority)". That is
the argumentation-textbook list, not Duke's. Think Again III actually teaches
generalization, **application of a generalization**, **inference to the best
explanation**, analogy, and causal. Syllabi checked 2026-09-04.

### Enum changes: 17 -> 22 forms

REMOVE (1)
- `sign` — merge into inference to the best explanation. All 23 records are
  observed-indicator -> underlying-condition (spectral lines -> atmosphere,
  charcoal -> wildfire, evidence-destruction -> guilt), which is IBE in Duke's
  framing. Relabel, do not delete.

KEEP (reversing an earlier call)
- `authority` — Duke files Appeals to Authority under Think Again IV fallacies, but
  the lesson teaches legitimate vs illegitimate appeals rather than rejecting the
  form. v1 does extraction, not evaluation, so the form stands.

ADD — inductive (2)
- `inference to the best explanation` — observation needs explaining; hypothesis
  explains it; no rival explains it as well; therefore probably true. Expected to
  absorb the 23 `sign` records and part of `other`.
- `application of generalization` — "most Fs are G; a is an F; so a is probably G".
  The INVERSE of `generalization` (sample -> population). Distinguished from
  `categorical syllogism` by the quantifier: "all" is deductive and certain, "most"
  is inductive and probable. The current model has no slot for this — a test case
  returned `inductive` + `other`, correctly refusing categorical syllogism and then
  having nowhere to go.

ADD — fallacies from Duke Think Again IV (4), all available in `data/logic/`
- `ad populum` (Appeals to Popular Opinion) — 232 available
- `begging the question` (Circularity) — 171 available
- `straw man` (Attacking a Straw Man) — 141 available, LOGIC calls it
  `fallacy of extension`
- `equivocation` — 49 available, the only class where supply is the constraint

STILL OUT OF SCOPE
- Truth tables, Venn diagrams, probability/Bayes, decision theory (Think Again II
  and III): computation over a formalised argument, not extraction from prose.
- Assuring / guarding / discounting markers (Think Again I): Duke's most
  distinctive apparatus and a good fit for this model's strong half, but it is
  span-tagging, i.e. a schema SHAPE change rather than an enum edit. Revisit after
  v1.2.
- Slippery slope, vagueness, self-sealers, silencers: no clean source. LOGIC's
  `faulty generalization` contains slippery slopes but as an unlabelled mixture.

### Plan

Phase 1 — schema + relabel what exists. No new data, no GPU.
  Update `lfx/schema.py` and the labeler instruction in `lfx/vertex.py` with
  definitions and precedence for the new forms. Re-label the existing `sign`,
  `other` and `analogy` records (~80) under v1.2 and measure what redistributes.
  The question to answer: how much of `other` is actually IBE?

Phase 2 — import the 4 fallacy classes from LOGIC.
  Extend `KEEP` in `pipeline/corpus/import_logic.py`. VERIFY EXTENSIONS BY READING
  PASSAGES FIRST — the `faulty generalization` error cost a wrong recommendation.
  Then blind-label extraction fields via `label_logic.py` and check the
  gemini/human agreement per class the way ad hominem (87%) and false dilemma (79%)
  were checked.

Phase 3 — generate IBE and application-of-generalization.
  No dataset covers either. Use the stratified generator plus blind labelling plus
  triage, NOT templates: the v2 run showed templates score 18/18 on templated text
  and 2/6 on natural prose.

Phase 4 — rebalance, split, retrain, evaluate.
  WATCH THE FALLACY RATIO. The corpus is already 39% fallacies (326/840). Adding
  four classes at cap 60 would push it past 50%, and a corpus that is half
  fallacies will bias the model toward predicting them. Cap the imported classes
  lower — 40 rather than 60 — and put the surplus in `extra_test_human.jsonl`,
  which is the more valuable destination anyway.

Phase 5 — the part that needs a human.
  `test_real` has 12 `other` records out of 34. Under v1.2 several are probably
  IBE, and those gold labels have to be re-reviewed by hand: it is the only split
  with human-verified labels and real prose, and its value depends on that.

### Success criteria

- `other` shrinks on test_real (currently 12/34, and 7 of the 16 form errors)
- the 4 new fallacy classes reach the 76-82% that ad hominem and false dilemma hit
- `application of generalization` separates cleanly from `categorical syllogism`
- overall form accuracy on `extra_test_human` (n=293) does not regress below 0.775

---

## Phase 0 — source search, and what it settled (2026-09-04)

Searched for real-prose examples of the two formal fallacies, because five classes
in the corpus are 73-80% template data and templates were measured not to transfer
(18/18 on templated text, 2/6 on prose).

| source | size | formal fallacies | license | verdict |
|--------|------|------------------|---------|---------|
| LOGIC (Jin 2022) | 2,449 | none | NONE declared | already imported |
| LogicClimate | 1,079 | none | NONE declared | unused, real journalism |
| MAFALDA (Helwe 2024) | 200 texts / 272 spans | none | CC-BY-SA | ADOPT AS TEST SET |
| Argotario (Habernal 2017) | 1,344 | none | — | REJECT |
| LogiQA 2.0 | 8,678 | none | NONE | reject |
| ReClor | LSAT/GMAT MCQ | none | NONE | reject |
| Open logic textbooks | — | yes | CC-BY / CC-BY-NC-SA | optional, low value |

**The finding: no public dataset labels formal fallacies in natural prose, and this
is not an oversight.** Fallacy datasets are built from Reddit, news and
crowdsourcing, where annotators tag ad hominem, straw man, false dilemma and ad
populum. Affirming the consequent barely appears because people rarely commit it
cleanly in real writing — it is a pedagogical category that lives in textbooks,
which is why textbooks are the only source. The corpus is template-heavy on those
classes because the real-world signal is scarce, not because the search was lazy.

Why the rejects were rejected:
- **Argotario** records both the writer's intended fallacy and a second player's
  guess. Agreement is 31% against ~17% chance for six classes. Samples confirm it:
  "If you fight once you will never stop fighting" is labelled hasty generalization
  (it is a slippery slope); "Because gorillas may make people frightened" is not an
  argument at all.
- **LogiQA 2.0** annotates multi-label REASONING TYPE (Categorical / Sufficient
  Conditional / Disjunctive / Conjunctive), not argument form — "Sufficient
  Conditional Reasoning" covers modus ponens and affirming the consequent
  identically. Passages are constraint puzzles, not arguments.
- **ReClor** is LSAT/GMAT multiple choice with no license.

### Decisions taken

1. **Formal fallacies stay synthetic.** Cap their share instead of chasing prose
   that does not exist. They are currently 45% of training data for a phenomenon
   the real world rarely produces, which is backwards. Do not claim prose
   performance on `affirming the consequent` / `denying the antecedent`.
2. **Shift weight to Duke's informal fallacies, where real data exists** —
   ad populum (232), circular reasoning (171), straw man (141), equivocation (49)
   in `data/logic/`. These are also what actually turns up in op-eds.
3. **Adopt MAFALDA as an external test set**, not training data. 200 expert
   annotated real-prose passages under a real licence, independent of this
   project's generator, labeller and reviewer — the only genuinely external check
   available. Overlaps 7 of our classes.
4. **Optionally mine open logic textbooks** for formal-fallacy examples. Toy-sized
   like the templates, but different toys: modest syntactic variety, low effort,
   low priority.

### Phase 1 result — schema v1.2 applied, and a hypothesis refuted

Re-labelled the 176 records in candidate classes (sign, other, analogy, causal,
categorical syllogism, generalization). 70 moved.

**The IBE hypothesis was wrong.** It predicted `inference to the best explanation`
would absorb a large share of `other`. It absorbed 3 of 35. IBE's 37 records come
almost entirely from `sign` (21, the merge working as designed) and `causal` (10).

**`other` collapsed anyway, 35 -> 9, but scattered** — mostly to `causal` (12), plus
categorical syllogism, disjunctive syllogism, hypothetical syllogism, modus ponens.

**Root cause found: two precedence rules contradicted each other.** Rule 2 assigned
`causal` when the conclusion asserts causation; rule 5 assigned `other` for a
passage chaining several inferences. A multi-step causal chain satisfied both, so
the labeller was resolving a coin flip. Rules rewritten per decision: a chain of
ONE kind of inference takes that kind's name; `other` is reserved for chains that
cross KINDS (Aquinas: causal chain + no-infinite-regress + identification with God).

**Two failures worth recording, because they bound what prompt work can achieve:**

1. The rewritten rule did NOT fix the case it was written for. `gen_other_0364` is
   a causal chain ending in the normative "transparency is paramount for effective
   governance" — rule 6 explicitly names that pattern as `other`, and the labeller
   still returns `causal`.
2. **Gemini's labelling has a ~12% noise floor.** Re-running the same 176 records at
   temperature 0 after editing rules 5/6 flipped 21 labels (12%), including changes
   unrelated to the edit: causal -> generalization, other -> authority, categorical
   syllogism -> authority. Small prompt edits perturb labels far from the edit.

That is the third prompt-engineering misfire in this project (the false-dilemma
overcorrection, this rule, and the collateral flips). STOP TUNING THE LABELER
PROMPT. Design around a ~12% label noise floor instead: keep contested records out
of held-out splits, and prefer human or constructed labels wherever a class matters.

### State entering Phase 2

840 records across 18 of 22 forms. Four classes have ZERO records — ad populum,
begging the question, straw man, equivocation — all four sourced in Phase 2 from
`data/logic/`. Eight classes sit under 25 records and need Phase 3 generation,
most urgently `application of generalization` (3) and `other` (9).
