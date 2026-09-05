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

### Phase 2 result — four fallacy classes imported, composition fixed

Extensions were checked by READING PASSAGES before mapping (the faulty
generalization lesson). Quality varied: equivocation 6/6 clean, straw man ~5/6,
begging the question ~4/6, ad populum ~3/6 — half of ad populum was GLOSSARY
ENTRIES, not arguments ("Encourages the audience to become part of a group").

Added a non-argument filter to `import_logic.py` for glossary entries, quiz
questions and fragments. These are not label noise; they are text with no argument
in it, and training on them teaches the model to extract structure that is not
there. It cut ad populum 232 -> 181 and equivocation 49 -> 44.

**Gemini/human agreement, which inverted the prediction from the samples:**

  ad hominem            89%      straw man     66%
  ad populum            83%      equivocation  64%
  false dilemma         76%      begging the question  76%

ad populum came out FINE once the glossary entries were filtered. equivocation came
out LOWEST despite being the cleanest class sampled — and the disagreements turn out
to be LOGIC's labels being wrong, not Gemini failing: a bare claim ("Science shows
us that improved quality of life comes through research and invention") and a non
sequitur (the barbecue exchange) are both labelled equivocation. The 6/6 sampled
were the clean head of a noisy tail. straw man's 66% is mostly Gemini saying
`ad hominem` — genuinely adjacent categories, and some LOGIC labels are wrong there
too.

**Consequence for the splits:** a human label is not automatically trustworthy.
`split_corpus.py` now trusts a `logic-human` record only when Gemini's BLIND label
agrees with it, instead of trusting the origin unconditionally. Otherwise LOGIC's
noisy tail lands in the held-out sets, which is exactly where noise does most damage.

**Composition fix — two caps in `merge_corpus.py`:**
per-class cap 45, plus a sub-cap limiting TEMPLATE records to 40% of any class.
Template share fell from 73-80% to 41-53%. Where limiting templates leaves a class
short, the shortfall is REPORTED as a work order rather than backfilled with more
templates.

Result: 709 records, 630 surplus human-labelled records to `extra_test_human.jsonl`.

**Phase 3 work order — 281 records of prose generation:**

  application of generalization  need 42     hypothetical syllogism  need 22
  other                          need 36     reductio ad absurdum    need 22
  generalization                 need 33     categorical syllogism   need 13
  hasty generalization           need 24     denying the antecedent  need 11 (tmpl-heavy)
  analogy                        need 24     affirming the consequent need 10 (tmpl-heavy)
  authority                      need 23     disjunctive syllogism   need 10 (tmpl-heavy)
                                             inference to best expl. need  8

Fallacies are 51% of the corpus, still too high. Filling the work order fixes this
on its own: at 45 across all 22 classes, the 9 fallacy classes would be 41%.

### Phase 3 result — 278 prose records generated, corpus balanced

Generated prose (NOT templates) for the 13 under-filled classes via
`pipeline/corpus/gen_fill.py`, then blind-labelled and triaged.

**Intent vs blind label: 227/278 = 82%** (the original corpus managed 79%).

  application of generalization  42/42  100%     categorical syllogism   11/13  85%
  inference to best explanation   8/8   100%     generalization          26/33  79%
  analogy / hasty gen / authority       96%      disjunctive syllogism    7/10  70%
  hypothetical syllogism         21/22   95%     affirming the consequent 7/10  70%
  reductio ad absurdum           19/22   86%     denying the antecedent   6/11  55%
                                                 other                   12/36  33%

`application of generalization` scored 42/42. The class hinges entirely on a
quantifier — "most Swedes ... probably Lutheran" is inductive, "all men ... is
mortal" is a categorical syllogism — and spelling that out in the generation brief
("NEVER 'all' or 'every'; the conclusion must be hedged") produced a clean split
from `categorical syllogism` on the first attempt.

`other` remains the problem class at 33%, scattering to IBE (8), categorical
syllogism (5), application of generalization (5), causal (4). Under v1.2 it means
"chains crossing KINDS of inference", and neither the generator nor the labeller
hits that reliably. **`other` may not be a learnable class as defined** — worth
watching in Phase 4 rather than fixing with more generation.

### Corpus v1.2 final

  945 records, all 22 forms present, 18 classes at 40-45
  origins: gemini 597, logic-human 261, template 87
  TEMPLATE SHARE 9% overall, 36-45% in its heaviest classes (was 38% overall,
    73-80% in five classes — the composition problem that motivated Phases 0-3)
  fallacies 42% (was 51%)
  only `other` is materially short, at 24

Remaining work order is trivial (1-7 records for eight classes) except `other`,
which needs 21 and is 33%-reliable to generate.

### Phase 4 — v3 regressed, v4 fixed it. v4 is the default.

**v3 regressed on real prose** despite the corpus work, because of a design error:
the uniform per-class cap of 45 treated a human-labelled real-prose passage and a
template as interchangeable, and cut ad hominem and false dilemma from ~80 records
to 45. Those are the two classes with the most abundant human-labelled prose in the
corpus. Balance is worth having because it stops the model learning a prior instead
of a task — it is not worth paying for by discarding the best data available.

**Fix: origin-aware caps.** human-labelled <= 60, generated <= 45, template <= 18
(and templates only fill what prose could not). That restores the v2 recipe —
ad hominem 80, false dilemma 86 — while keeping all 22 classes and a 9% template
share.

**Also fixed: a silent evaluation bug.** `extra_test_human.jsonl` held 630 lines for
571 unique ids, and `evaluate.py` keys on id, so 59 records were dropped from every
score reported against it. The merge now deduplicates. All numbers below use the
corrected 512-record set, so the v3 figures here are lower than those first
reported for v3.

**Form accuracy on 512 real human-labelled passages, never trained on:**

| class                | v2 (old set) | v3    | v4    | n   |
|----------------------|--------------|-------|-------|-----|
| ad hominem           | 0.760        | 0.445 | 0.651 | 209 |
| false dilemma        | 0.824        | 0.321 | 0.774 |  53 |
| begging the question | -            | 0.600 | 0.863 |  80 |
| ad populum           | -            | 0.728 | 0.719 | 114 |
| straw man            | -            | 0.500 | 0.536 |  56 |
| OVERALL              | -            | 0.525 | 0.699 | 512 |

v2's column is not strictly comparable: it was measured on a 2-class set that still
carried the duplicate-id bug. v4 reaches near-v2 accuracy on the shared classes
while also separating `straw man`, which competes directly with ad hominem.

**Diagnostic cases — 4 of 5 pass, two of them for the first time:**

  PASS  affirming the consequent   (rain/sidewalk — wrong in every previous version)
  PASS  denying the antecedent
  PASS  categorical syllogism
  PASS  equivocation               (a class v2 could not express at all)
  FAIL  false dilemma -> disjunctive syllogism   (the project's most stubborn confusion)

**test_real (n=34) reads 0.412 against v3's 0.529.** That is 14/34 versus 18/34 — a
four-example difference on a 34-record set, which is not a signal. The 512-record
human set is the number to quote.

**CLI default switched to `models/mlx_v4_8bit`.**

### The finding that outlasts this phase

`test_real` form accuracy has now read 0.529, 0.529, 0.529 and 0.412 across four
training runs on corpora differing in size (284 to 741), balance, provenance and
schema. Corpus work has stopped being the lever. What remains:
  - the ~12% Gemini labelling noise floor (measured in Phase 1)
  - the capacity of a 1.5B base model
  - `form` being genuinely underdetermined for condensed philosophical prose —
    `other` has resisted every attempt to make it learnable, from the 53% pilot
    through to 12/36 generation agreement and 1/4 test accuracy
Those are three different projects. Choose deliberately rather than drifting.

### Phase 5 — model capacity was a real lever. Qwen2.5-3B beats v4.

Same corpus, same splits, same recipe (6 epochs, r=16, effective batch 8): only the
base model changed, 1.5B -> 3B. 558 steps, 36 minutes on a free T4.

**Form accuracy on the 512 human-labelled passages, never trained on:**

| form                 | n   | 3B    | v4    | delta  |
|----------------------|-----|-------|-------|--------|
| straw man            |  56 | 0.804 | 0.536 | +0.268 |
| false dilemma        |  53 | 0.887 | 0.774 | +0.113 |
| ad hominem           | 209 | 0.742 | 0.651 | +0.091 |
| ad populum           | 114 | 0.781 | 0.719 | +0.061 |
| begging the question |  80 | 0.700 | 0.863 | -0.163 |
| OVERALL              | 512 | 0.766 | 0.699 | +0.066 |

Schema validity 0.994 vs 0.955; argument_type 0.809 vs 0.771; premise F1 0.719 vs
0.688. Every aggregate metric moved the same direction.

`test_real` read 0.529 against v4's 0.412 -- 18/34 vs 14/34. Still not a signal.
That split has now read 0.529 in four of five runs across corpora differing in
size, balance, provenance and schema. **Stop quoting it.**

**What the capacity actually bought:** the confusion cluster, not general ability.
3B recovered 16 ad hominem that v4 called begging the question and 12 it called
straw man. Straw man was v4's worst class and competes directly with ad hominem;
3B nearly resolves the pair.

**What it cost:** begging the question -> equivocation, 10 cases, plus 6 to straw
man. `equivocation` is the class v1.2 introduced and v4 could barely express at
all. The larger model learned it well enough to over-apply it. That is a class
BOUNDARY problem, not a capacity one, and it is the single largest remaining
block: fixing it alone would put 3B above 0.80.

**Eval loss stayed misleading.** 3B bottoms at epoch 2 (0.173, below the 1.5B's
0.201) and rises to 0.337 by epoch 6 -- yet epoch 6 is the checkpoint that scores
0.766. Token-level loss and form accuracy diverge on this task in both model
sizes. Early stopping on eval loss would have cost the gain a third time.

### The free-tier Colab ceiling, measured

A free T4 lives about **55 minutes**. Two runs confirmed it (14:22->15:23,
15:35->16:30), both reclaimed mid-prediction.

  setup, deps, upload   ~10 min
  training               36 min
  predictions           >20 min   <- never fits

Predictions cannot complete on Colab free. The pipeline now downloads the adapter
the moment training ends (`pack_3b.py` + `run_3b.sh`), and predictions run locally
against the MLX build. One 3B adapter was lost to learning this.

Three further hardening fixes, each from a real failure this phase:
  - jobs run DETACHED on the VM (`lfjob.py`); the local side only polls, so a dead
    terminal no longer kills a run -- the failure that cost three earlier runs
  - the poller tolerates a transient `exec` error instead of exiting; that error
    pruned the session record and orphaned a VM we could no longer address
  - the job-control module is NOT named `joblib.py`: /content is on sys.path, and
    that name shadowed the real joblib, breaking sklearn and so transformers

### Phase 6 — auditing the labels, and checking the ruler

**The equivocation/begging-the-question leak was a data defect, not a boundary.**
Of 44 equivocation training records, about half were equivocation. The rest were
straw man, appeals to authority, a sorites, and passages that are not arguments.
The smallest class in the corpus was also the noisiest, so it acted as an
attractor for anything involving restatement or word-play -- which is what
circular arguments look like.

Blind-relabelled both classes (`audit_classes.py`): 75% agreement on begging the
question, 64% on equivocation. Every disagreement was adjudicated by a human;
Gemini's answer was adopted only where a human agreed. It was wrong on the two
cleanest equivocations in the corpus -- "only man is rational, no women is man"
and the rare-steak pun -- which is the argument for a human queue over an
automatic overwrite.

Result: 6 dropped as not arguments, 15 relabelled, 5 human labels confirmed.
equivocation 44 -> 31, begging the question 61 -> 54.

**Then the ruler itself was checked** -- a 100-record seeded sample of the 512
held-out human records, blind-labelled the same way. 23 disagreements, all 23
adjudicated:

  gold correct, Gemini wrong   11
  gold wrong                    8
  genuinely contested           4

So the held-out set carries roughly **8-12% label error**, and every score quoted
against it should be read as "against gold that is itself about 90% right".

Straw man read 50% agreement, which looked alarming because straw man is where
the 3B gained most (0.536 -> 0.804). Reading the seven disagreements reversed
that: four are genuine straw men that Gemini called `ad hominem` -- a predictable
confusion, since straw-manning usually attacks the person too -- and the 3B
agreed with the gold on all four. The low agreement measures Gemini's blind spot,
not the test set's.

**Two defects removed, both properties of the text or the schema rather than of
any prediction, so neither can flatter a model:**

  - 43 quiz prompts and dictionary definitions across the splits (5.3% of the
    held-out set): "Doing something because everyone else is doing it",
    "...What fallacy has Louise committed? http://www.funtrivia.com". No model can
    extract an argument from a question about arguments.
  - 2 records in test_real still labelled `sign`, deleted by schema v1.2. No model
    trained since has seen that label, so 6% of a 34-record split was unwinnable.

A shape-based detector was tried for the first and REMOVED: rejecting text that
starts lower-case or lacks terminal punctuation flagged 12.9% of the held-out set,
and most were real arguments that merely lack a full stop ("We know God exists
because he made everything"). Both properties are common here and carry no
signal. Definitions are distinguished by their PHRASING, so phrase and opener
matching does the work, at 5.3% with no false positives on inspection.

**Corrected scores on the cleaned sets:**

  extra_test_human (485)   3B 0.765   v4 0.700
  test_real (34)           3B 0.559   v4 0.471   -- still 19 vs 16, still noise

The held-out numbers barely moved, which is itself worth knowing: the models were
scoring on those quiz prompts, having learned the same prompt-to-label
association from equivalent items in training. Removing them costs nothing and
removes a way of being right for the wrong reason.

**A bias this set cannot escape:** the model trains on labels from the same
annotators it is tested against, so part of what 0.765 measures is agreement with
LOGIC's house style rather than correctness. A genuinely independent test set
would need different annotators.

### Two latent bugs, found by a guard rather than looked for

  - 5 records carry an inductive form with `argument_type: deductive`. Four are in
    train.jsonl, so every model from v2 through the 3B trained on labels that
    contradict themselves -- the same defect `inconsistency()` reports the model
    emitting at 0-2%.
  - 15 ids each name TWO DIFFERENT passages. `evaluate.py` keys on id and was
    silently dropping one of every pair; the splitter could put one id on both
    sides of the train/test line.

The second nearly caused a bad edit: keying decisions by id alone relabelled both
halves of logic_0096, including a passage nobody had audited. Decisions are now
matched on id AND text, and anything that cannot be pinned to exactly one record
is refused rather than applied to both.

### Phase 7 — schema v2.0: `form` becomes a list

A single-value `form` could not describe a multi-step argument. "Forty of the
Swedes I met were Lutheran, so most Swedes are; so Sven probably is" performs a
generalization AND an application of one; whichever the annotator wrote down
became the only correct answer, so a model reading the passage correctly scored
as wrong. Those records were pushed into `other`, which then meant two unrelated
things -- "several forms apply" and "no form applies". `equivocation` had already
shown what a double-meaning label does to a class.

  form = ["generalization", "application of generalization"]   several apply
  form = ["other"]                                             none applies

Nearly every record holds one form, where set equality is the same comparison
v1.2's string equality made. Re-scoring the 3B and v4 predictions after the
migration returned 0.765 and 0.672 -- identical to before. **The schema change is
measurement-neutral**, so every number from earlier phases stays comparable.

**A drift bug found on the way in.** `lfx/prompts.py`, whose own docstring says
"single source of truth -- change it in one place or not at all", still carried
the v1.0 enum written out by hand. Every model from v1.2 onward was trained,
evaluated and shipped with a prompt offering the deleted `sign` and never
mentioning `straw man`, `ad populum`, `begging the question`, `equivocation`,
`application of generalization` or `inference to the best explanation` -- four of
which are classes in the main test set. The models learned them from the training
targets while being told a label space that did not contain them. The prompt is
now BUILT from `FORMS` and cannot drift again.

**Two things the build itself taught us:**

  - The labeler cannot invent chains from prose rules. Given rules alone it
    returned single forms for every multi-step passage except the one written
    verbatim in its instruction. It needed few-shot examples -- the same lesson as
    "templates don't generalize", one level up.
  - The consistency rule was wrong as first written. Requiring argument_type to
    suit EVERY form in a chain immediately rejected a real argument: night shifts
    -> poor sleep -> clinical error is causal and inductive, while "a hospital must
    not adopt such a policy, so withdraw the rota" is a deductive modus tollens.
    Real chains mix families; forbidding that only moves the single-value
    bottleneck from `form` to `argument_type`. On a chain, argument_type now
    describes how the argument as a whole presents its conclusion.

### The `other` class is not labelable, measured

The 26 `other` passages were labelled twice at temperature 0, changing nothing but
adding two chain examples to the few-shot. **65% of the labels changed**, and only
5 of the 17 changes were the intended single->chain effect. The rest were single
form -> a DIFFERENT single form, which the change had nothing to do with:

  inference to the best explanation -> modus ponens
  categorical syllogism            -> authority
  authority                        -> other
  causal                           -> other  /  other -> causal

**The control shows this is specific to `other`, not the labeler.** The same test
over 60 records from six well-defined classes flipped 13%, and six of those eight
were the chain examples working as intended -- ~3% genuine instability.

  ad hominem 16/16    categorical syllogism 10/10    analogy 9/9
  generalization 4/5  modus ponens 8/11              causal 5/9

Sharply-defined classes are perfectly stable. The fuzzier inductive ones are less
so, which is exactly where the schema's own boundaries are softest. For `other`
the answer is driven by prompt context rather than by the text.

So the 24 GENERATED `other` records are dropped: re-labelling them would inject
noise, and keeping them teaches the model to answer `other` for arguments that are
not unnameable. The 2 logic-human ones stay -- adjudicated by a person in Phase 6,
an appeal to ignorance and a continuum fallacy, which is what `other` is now for.

Corpus v2.0: 1037 records. train 712, val 159. Test splits untouched: removing
held-out records on the strength of a labelling experiment would be adjusting the
ruler, not the model.

**Still open:** test_real holds 12 `other` records and now has almost no training
signal for that label, so they behave like the `sign` records did -- close to
unwinnable. test_real is already too small to resolve anything; this is one more
reason to retire it rather than repair it.

### The prompt now travels with the weights

Generating the prompt from `FORMS` fixed the drift, then immediately caused a
worse version of it. Editing `SYSTEM` silently re-specifies every model already
shipped. Measured on test_real, the packaged 3B given the new v2.0 wording:

  form accuracy   0.559 -> 0.265
  schema valid    1.000 -> 0.853

The weights did not change. A prompt is a single source of truth only for models
trained against that version of it.

So `package.py` writes `prompt.txt` into the model directory and
`lfx.prompts.for_model` reads it back; `cli.py` and `predict_mlx.py` use the
model's own prompt, not the repo's current one. Re-running the 3B through the
restored path returns 0.559 and 1.000 exactly, matching its stored predictions.

Packaging was three hand-typed commands before this, which is how the prompt came
adrift in the first place. It is now one script, and freezing the prompt is a step
inside it rather than something to remember.

### Work order before the retrain: `other` has no training signal

  train    other = 0 of 712
  test_gen other = 4
  test_real other = 12

Dropping the 24 unstable records took every `other` example out of training. A
model retrained today could not emit the label at all, so an appeal to ignorance
or a sorites would be forced into the nearest wrong class -- `other` becomes the
next `sign`, a label in the schema that no model can produce.

The escape hatch needs roughly 40 real arguments whose form genuinely is not among
the 22: appeal to ignorance, continuum fallacy, tu quoque, gambler's fallacy,
amphiboly. Two are already in the corpus (the leprechauns record and the
therapist/placebo one), both found in real prose, so this is collectable rather
than hypothetical.

### The escape hatch, rebuilt from real prose

`other` went from 0 training records to 53, drawn from the LOGIC classes Phase 2
deliberately did NOT import *because they had no counterpart in our enum* -- which
is exactly the property the escape hatch needs.

  appeal to emotion     "Professor, PLEASE reconsider my grade. I only answered 7%
                        of the questions correctly, but I need this to graduate."
  fallacy of relevance  "I know you want to imprison me for murdering my parents,
                        but judge, have mercy -- I'm an orphan!"
  intentional           "I believe in God because no one can prove a god doesn't
                        exist." (appeal to ignorance)

**`false causality` was excluded although it is the largest of the four (203
records).** Our `causal` is the VALID inductive form, so filing post hoc examples
under `other` would teach the model that causal-looking text is unnameable -- and
`causal` was already among the least stable classes in the Phase 7 control (5/9).
That is the shape of the `faulty generalization` error: a class whose NAME
suggests a mapping its EXTENSION does not support.

Two filters. The blind labeler had to agree the record was `other` -- 67 of 134
did, and the survivors came out evenly split across the three sources (22/23/22),
which is what matters: `other` must mean "none of the 22 apply", not "emotional
appeal". A single-source escape hatch would rebuild the double-meaning problem
that made the class unlearnable in the first place. Then a human read all 67 and
excluded 16: glossary definitions written as bare noun phrases that `is_argument`
cannot catch, descriptions of adverts rather than arguments, one quiz prompt, and
five near-duplicates (the moon short-ribs/spare-ribs pair, two invisible-unicorns,
two cat-sweaters, three phone-bill variants).

51 kept. Corpus 1088 records, `other` at 53 -- in line with the other classes,
where before it was 26 records nobody could label twice the same way.

### Phase 8 — the last four human-labelled classes

  ad hominem     81   98%      false dilemma  87   75%
  ad populum     60   80%      straw man      62   58%

`ad hominem` at 98% is the cleanest class in the corpus. `straw man` at 58% is
mostly the Gemini blind spot already measured on the test set -- 8 of its 26
disagreements are `ad hominem`, which is predictable because straw-manning
usually attacks the person too. The gold is right on those.

`false dilemma -> disjunctive syllogism` showed up 7 times, the project's most
stubborn confusion appearing in the LABELS rather than in a model. Those are
genuinely contested ("Either determinism is true, or human beings possess genuine
free will" is arguably exhaustive) and were left alone rather than churned.

**The useful find was 20 more non-arguments**, surfaced because the blind labeler
asked for `other` on them:

  "assumes that if many people act or believe a certain way, it must be the right way"
  "This persuasive technique is used to make you feel left out if you don't join the group."
  ", rather than on his arguments or opinions."
  "What is the name of the fallacy in question 4 about Marcus?"

Plus one real relabel: "Red had come up six times in a row on the roulette wheel,
so Greg knew it was close to certain black would come up next" was filed as
`ad populum`. It is a gambler's fallacy -- no form in the enum names it, so it is
`other`, which is what the escape hatch is for.

**The glossary filter is now scoped to LOGIC-sourced text.** Applied to the whole
corpus it discarded three real generated arguments, because its patterns are
ordinary English and only signal a definition in a source that mixes definitions
with instances:

  "Battery OCCURS WHEN one intentionally causes harmful contact... the defendant
   deliberately shoved the victim"        -> a categorical syllogism, not a definition
  "Overwatering often causes root rot for THIS TYPE OF plant"
                                          -> a real hasty generalization
  "Florentine artists quickly adopted THIS TECHNIQUE"   -> a real appeal to authority

A filter justified by one source's defects must be applied only to that source.

Corpus 1068 records, 22 classes, `other` at 54. train 707, val 159.

### Rebuilding the splits — three leaks found

**Do not re-run `merge_corpus.py`.** It rebuilds from the original interim
sources, which would discard every Phase 6-8 correction. `corpus_v20` is already
the capped, corrected corpus; it is split directly.

**Leak 1: 4 records were in both the corpus and the held-out human set** (3 unique
passages, one duplicated inside the corpus). All `straw man` -- LOGIC carries the
same passage under different ids, so it landed on both sides. Every straw man
score on that set was inflated by 3 of 56 memorised records, about 5%. The 3B's
0.804 is really nearer 0.75-0.78; still far above v4's 0.536, so the Phase 5
conclusion holds, but the number should be quoted with this attached.

**Leak 2: 2 passages landed in both train and val** on the first split, again the
same text under different ids. Text-level dedup now runs BEFORE the split rather
than after, since afterwards the duplicates are already on opposite sides.

**Leak 3: 46 ids were shared with the held-out set** with zero text overlap -- the
collision-renaming from Phase 6 meeting a set that kept its original ids.
Harmless today, silent mis-scoring tomorrow, because `evaluate.py` joins on id.
New split ids are now made unique against the held-out files.

**And the lesson that cost the most time:** `clean_splits.py` fixed the splits in
place, and the fixes did not survive a re-split. The splitter regenerates
`test_real` from `labeled_reviewed.jsonl`, so the retired `sign` labels came
straight back and the v2.0 migration was undone -- test_real briefly held 34
invalid records. **A fix belongs upstream of the generator, not on its output.**

Splits: train 741, val 160, test_gen 160, test_real 34, extra_test_human 485.
No text or id overlap between any pair; every label valid.
