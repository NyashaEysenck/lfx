# Logical Form Extractor — Research & Engineering Report

## Executive Summary

This report documents the end-to-end engineering, training dynamics, dataset curation, and deployment of **Logical Form Extractor (LFX)**: a specialized 3B-parameter language model fine-tuned to parse natural-language arguments into structured JSON.

### Current System Specification
- **Base Model:** `Qwen/Qwen2.5-3B-Instruct`
- **Fine-Tuning Technique:** LoRA ($r=16$, $\alpha=32$, 6 epochs, target modules: `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`)
- **Schema:** v2.0 (22 logical forms, list-valued `form`, `premises`, `conclusion`, `argument_type`, `suppressed_premise`)
- **Deployment Artifacts:**
  - **Ollama Registry:** `nyasha_stino/lfx` (`:3b-q8`, `:3b-q4`, `:latest`)
  - **Apple Silicon (MLX):** `models/mlx_v5_8bit` via the `./logicalform` CLI
- **Benchmark Performance (Held-Out Human Prose, $n=485$):**
  - **End-to-End Form Accuracy:** **76.5%** (`3b-q8`) / **75.5%** (`3b-q4`) / **76.3%** (MLX 8-bit)
  - **Schema Validity:** **99.4%**
  - **Argument Type Accuracy:** **83.8%**
  - **Premise F1:** **0.715**
  - **Suppressed Premise Match:** **78.4%**

---

## 1. Task Formulation & Production Schema (v2.0)

**Input:** A natural-language argument passage (typically one paragraph).
**Output:** A single, strictly validated JSON object describing the inferential structure.

The extraction task is deliberately decoupled from *argument evaluation* (soundness, epistemic validity, or truth-checking). The objective is structural parsing: identifying what claims the author treats as premises, what they assert as the conclusion, the inferential form employed, and what implicit bridge premises must be assumed to make the reasoning coherent.

```json
{
  "premises": [
    "All men are mortal.",
    "Socrates is a man."
  ],
  "conclusion": "Socrates is mortal.",
  "argument_type": "deductive",
  "form": [
    "categorical syllogism"
  ],
  "suppressed_premise": null
}
```

### Supported Taxonomy (22 Forms)
The taxonomy aligns with classical deductive logic and informal logic curricula (specifically modeled on Duke University's *Think Again* specialization):

1. **Valid Deductive (6):** `modus ponens`, `modus tollens`, `hypothetical syllogism`, `disjunctive syllogism`, `categorical syllogism`, `reductio ad absurdum`
2. **Inductive (6):** `generalization`, `application of generalization`, `inference to the best explanation`, `analogy`, `causal`, `authority`
3. **Formal Fallacies (2):** `affirming the consequent`, `denying the antecedent`
4. **Informal Fallacies (7):** `ad hominem`, `hasty generalization`, `false dilemma`, `ad populum`, `begging the question`, `straw man`, `equivocation`
5. **Escape Hatch (1):** `other` *(reserved for complex multi-step chaining or arguments outside this 21-form taxonomy)*

### Key Schema Design Decisions
- **`form` as a list (Schema v2.0):** In Schema v1, `form` was a single string. This created an artificial bottleneck on multi-step arguments where an annotator had to pick one step arbitrarily. In v2.0, `form` is an array of 1 to 3 strings, ordered primary move first.
- **Folding fallacies into `form`:** Fallacies are treated as inferential forms rather than an orthogonal boolean tag. A formal fallacy presents itself as deductive; an informal fallacy mimics a valid deductive or inductive form.
- **Suppressed Premise Reconstruction:** If an argument relies on an unstated enthymematic assumption to connect its stated premises to its conclusion, the model reconstructs it as a single declarative proposition.

---

## 2. Tooling & Compute Architecture

- **Local Development & Inference:** Apple Silicon (macOS, M4 Pro, 24 GB unified memory). Local inference runs on 8-bit quantized weights via MLX (`mlx-lm`) at ~0.6 s per passage (~55–63 tokens/sec).
- **Training Environment:** Google Colab (NVIDIA T4 / L4 GPUs) using [Unsloth](https://github.com/unslothai/unsloth) for accelerated LoRA fine-tuning.
- **Remote Orchestration:** Driven from the terminal via the Google Colab CLI (`google-colab-cli`) with detached background execution scripts (`pipeline/train/lfjob.py`), eliminating browser notebook dependencies and disconnections.
- **Corpus Labeling:** Google Gemini (`gemini-2.5-flash`) orchestrated through Vertex AI via Application Default Credentials (ADC), enforcing JSON schemas natively to prevent off-enum drift during data generation.
- **Model Packaging & GGUF Conversion:** Model weights merged on CPU (`pipeline/train/merge_adapter.py`), compiled to GGUF using `llama.cpp`'s conversion tools, and packaged into Ollama modelfiles (`pipeline/deploy/to_ollama.sh`).

---

## 3. Core Empirical Findings & Principles

### Finding 1: Real Data Generalizes; Templates Teach the Template
Early iterations used synthetic, template-generated formal fallacies (`affirming the consequent`, `denying the antecedent`).
- On template-generated test passages, the model scored **18/18 (100%)**.
- On natural human prose, the same checkpoint scored **2/6 (33%)**.
The model learned template phrasing rather than the logical structure. In contrast, classes trained on natural human-written passages from the LOGIC dataset (e.g., `false dilemma`) scored **10/12 (83%)** on unseen human prose.

### Finding 2: The Capacity Lever Operates on Generalization, Not Synthetic Task Learning
When comparing `Qwen2.5-1.5B` against `Qwen2.5-3B` on identical splits:
- On generated test prose (`test_gen`, $n=160$), accuracy was virtually unchanged: 0.781 vs 0.812 ($\Delta = +0.031$, not statistically significant, $z=0.66$).
- On natural human prose (`extra_test_human`, $n=485$), accuracy jumped from **0.647 to 0.763** ($\Delta = +0.115$, McNemar test $z=5.13$, $p < 10^{-6}$).
Scaling parameters did not teach the model the syntax of logical forms; it provided the rhetorical capacity to recognize those forms amidst the messiness and stylistic variance of human writing.

### Finding 3: Deduplication Belongs Upstream of Splitting
In multi-source datasets, duplicate texts often enter under different identifiers (e.g., cross-posts or re-scrapes in the source LOGIC dataset). Running text-level deduplication *after* splitting inevitably leaks duplicates across the train/validation/test boundary. Deduplication and structural validation must occur strictly upstream before any split generation.

### Finding 4: Quantization Profiles Differ by Field Complexity
Moving from 8-bit (`Q8_0`) to 4-bit (`Q4_K_M`) quantization costs only 0.010 in overall form accuracy (0.765 $\rightarrow$ 0.755). However, fine-grained tasks (extracting exact premise boundary spans and judging suppressed premises) suffer disproportionate drops ($-0.042$ on premise F1, $-0.052$ on suppressed premise match). High-level form classification is robust to coarse weights; textual reconstruction requires greater precision.

---

## 4. Detailed Engineering & Research Chronology

### Phase 0 — Empirical Source Search & Formal Fallacies in the Wild
A comprehensive survey of external corpora was conducted to find real-world natural-language examples of formal fallacies:

| Source | Size | Formal Fallacies | License | Decision |
|---|---|---|---|---|
| **LOGIC** (Jin et al., 2022) | 2,449 | None | MIT (Appendix A) | **Adopted as primary human corpus** |
| **LogicClimate** | 1,079 | None | MIT | Preserved for domain evaluation |
| **MAFALDA** (Helwe et al., 2024) | 200 texts | None | CC BY-SA 4.0 | Partially overlaps LOGIC; held-out check |
| **Argotario** (Habernal et al., 2017) | 1,344 | None | Unspecified | Rejected (low annotator agreement: 31%) |
| **LogiQA 2.0** | 8,678 | None | Proprietary | Rejected (constraint puzzles, not arguments) |
| **ReClor** | — | None | Proprietary | Rejected (LSAT multiple choice) |

**Key Finding:** No public NLP dataset labels formal fallacies in natural prose. Formal fallacies (*affirming the consequent*, *denying the antecedent*) are predominantly pedagogical constructs; in open web discourse and op-eds, informal fallacies (*ad hominem*, *straw man*, *false dilemma*) heavily dominate. Consequently, formal fallacies were represented via constrained synthesis, while informal fallacies were drawn from authentic human text.

### Phase 1 — Baseline, Pilot, and Undertraining Dynamics
- **Seed Corpus:** 34 diverse passages (Aquinas, Pascal, Descartes, Lincoln, MLK, textbook fallacies).
- **Initial Model:** `Qwen2.5-1.5B-Instruct` with LoRA ($r=16$, 3 epochs).
- **Observed Defect:** At 3 epochs, the model suffered from severe field-swapping (writing `form` strings into `argument_type`, e.g., `"argument_type": "modus tollens"`).

| Split | Base Model | 3 Epochs | 6 Epochs |
|---|---|---|---|
| **Field-Swap Rate (`test_real`)** | 21% | 24% | **3%** |
| **Field-Swap Rate (`test_gen`)** | 22% | 47% | **5%** |

Increasing training duration from 3 to 6 epochs completely stabilized schema compliance without needing constrained grammar decoding. End-to-end form accuracy improved from 0.029 (base) to 0.529 (6 epochs).

### Phase 2 — Fallacy Repair and Corpus Composition
- **Critical Failure Mode:** The early model frequently repaired fallacious arguments into valid deductive twins (e.g., *affirming the consequent* misidentified as *modus ponens*; *false dilemma* misidentified as *disjunctive syllogism*).
- **Root Cause:** The synthetic labeler exhibited an inherent charity bias, mapping subtle fallacies to valid structures. Contested records had been routed directly into training data.
- **Mitigation:** Imported real human-labeled fallacy instances from the LOGIC dataset. Introduced strict stratified caps to ensure fallacies did not exceed 40–45% of the total corpus.

### Phase 3 — Schema v1.2 Alignment (Duke "Think Again")
- Systematic alignment with formal logic curricula:
  - Removed `sign`; merged into `inference to the best explanation` (IBE).
  - Added `application of generalization` (inductive counterpart to categorical syllogism).
  - Added Duke informal fallacies: `ad populum`, `begging the question`, `straw man`, and `equivocation`.
- Evaluated inter-annotator agreement between Gemini and human labels on imported classes:
  - `ad hominem`: 89% agreement
  - `ad populum`: 83% agreement
  - `false dilemma`: 76% agreement
  - `begging the question`: 76% agreement
  - `straw man`: 66% agreement
  - `equivocation`: 64% agreement

### Phase 4 — Balancing Class Proportions
- Uniform per-class caps (45 examples each) initially caused performance regressions on human prose by discarding high-quality human examples of `ad hominem` and `false dilemma`.
- **Adopted Origin-Aware Caps:**
  - Human-labeled prose: $\le 60$ records per class
  - Generated prose: $\le 45$ records per class
  - Synthetic templates: capped at $\le 18$ records (restricted to filling severe data deficits)

| Class | v2 (n=293) | v3 (Uniform Caps) | v4 (Origin-Aware Caps) |
|---|---|---|---|
| **ad hominem** | 0.760 | 0.445 | **0.651** |
| **false dilemma** | 0.824 | 0.321 | **0.774** |
| **begging the question** | — | 0.600 | **0.863** |
| **ad populum** | — | 0.728 | **0.719** |
| **straw man** | — | 0.500 | **0.536** |
| **OVERALL (n=512)** | — | 0.525 | **0.699** |

### Phase 5 — Model Capacity Ablation (1.5B vs. 3B)
- Evaluated `Qwen2.5-3B-Instruct` using identical hyperparameters ($r=16$, $\alpha=32$, 6 epochs).
- Form accuracy on held-out human passages rose from 0.699 to 0.766.
- The 3B model resolved the primary confusion cluster between `straw man` and `ad hominem` (+0.268 accuracy gain on straw man).

| Form | n | 1.5B (v4) | 3B | Delta |
|---|---|---|---|---|
| **straw man** | 56 | 0.536 | **0.804** | +0.268 |
| **false dilemma** | 53 | 0.774 | **0.887** | +0.113 |
| **ad hominem** | 209 | 0.651 | **0.742** | +0.091 |
| **ad populum** | 114 | 0.719 | **0.781** | +0.061 |
| **begging the question** | 80 | 0.863 | **0.700** | -0.163 |
| **OVERALL** | 512 | 0.699 | **0.766** | **+0.067** |

### Phase 6 — Label Auditing & Ruler Verification
- Audited the held-out evaluation splits. Discovered that the human gold dataset itself contained an ~8–12% noise floor (misclassifications and non-argument trivia questions).
- Stripped 43 non-argument definitions and quiz prompts from test sets.
- Pinned deduplication logic to combined `(id, text)` tuples to prevent collisions.

### Phase 7 — Schema v2.0 List Migration & Prompt Freezing
- Migrated `form` from a scalar string to a JSON array (`list[str]`).
- Established the architectural rule: **The prompt travels with the model weights.** All inference scripts (`cli.py`, `evaluate.py`, `to_ollama.sh`) dynamically read `prompt.txt` directly from the model directory.

### Phase 8 — Hardening the Escape Hatch (`other`)
- Discovered that pure synthetic generation of `other` was unstable (65% label churn upon re-prompting).
- Rebuilt `other` using 51 human-written real passages representing fallacies outside the 22-form enum (e.g., appeals to emotion, irrelevant conclusions, and gambler's fallacies).

### Phase 9 — Rigorous Data Leak Audit & Final Capacity Verification
- Audited all splits for text and identifier overlap:
  - Eliminated 4 records present in both training and test sets.
  - Eliminated 2 passages present in both train and validation.
  - Resolved 46 identifier collisions with held-out validation sets.
- Re-evaluated the clean 1.5B vs 3B comparison across identical MLX runtimes:

| Split | n | 1.5B | 3B (v5) | Gap | Statistical Significance |
|---|---|---|---|---|---|
| **extra_test_human** | 485 | 0.647 | **0.763** | **+0.115** | McNemar $z=5.13$ ($p < 10^{-6}$) |
| **test_gen** | 160 | 0.781 | **0.812** | +0.031 | $z=0.66$ (not significant) |
| **test_real** | 34 | 0.382 | **0.412** | +0.029 | Sample size too small |

### Phase 10 — Quantization Profiling & Ollama Deployment
- **Safetensors Import Defect:** Identified that Ollama's internal safetensors conversion produced garbled outputs for merged Qwen2.5 checkpoints. Resolved by compiling through `llama.cpp` (`convert_hf_to_gguf.py`) before building the Ollama modelfile.
- **Quantization Comparison ($n=485$):**

| Metric | Ollama Q8 (`3b-q8`) | Ollama Q4 (`3b-q4`) | MLX 8-bit |
|---|---|---|---|
| **Form Accuracy (e2e)** | **0.765** | 0.755 | 0.763 |
| **Schema Valid** | **0.994** | 0.992 | 0.992 |
| **Argument Type Acc** | **0.838** | **0.838** | 0.842 |
| **Premise F1** | 0.715 | 0.701 | **0.743** |
| **Suppressed Present** | 0.784 | 0.748 | **0.800** |
| **Throughput (M4 Pro)** | ~63 tok/s | ~95 tok/s | ~55 tok/s |

Released both tags to Ollama (`nyasha_stino/lfx:3b-q8` and `nyasha_stino/lfx:3b-q4`).

---

## 5. Summary of Final Benchmark Results

### Held-Out Human Benchmark (`extra_test_human`, $n=485$)
All passages are human-authored argumentative texts from the LOGIC benchmark, never seen during fine-tuning:

| Metric | Base Model (3B) | 1.5B Tuned | Ollama Q4 (3B) | Ollama Q8 (3B) | MLX 8-bit (3B) |
|---|---|---|---|---|---|
| **Form Accuracy (e2e)** | ~0.03 | 0.647 | 0.755 | **0.765** | 0.763 |
| **Schema Validity** | 0.735 | 0.965 | 0.992 | **0.994** | 0.992 |
| **Argument Type Acc** | 0.560 | 0.816 | 0.838 | **0.838** | **0.842** |
| **Premise F1** | 0.778 | 0.718 | 0.701 | 0.715 | **0.743** |
| **Suppressed Premise Match** | 0.480 | 0.710 | 0.748 | 0.784 | **0.800** |

### Per-Class Form Accuracy on Human Prose (Q8_0):
- **False dilemma:** 82.7% (43 / 52)
- **Ad hominem:** 78.9% (157 / 199)
- **Ad populum:** 77.9% (81 / 104)
- **Straw man:** 75.9% (41 / 54)
- **Begging the question:** 64.5% (49 / 76)

---

## 6. Future Scope & Parked Directions

1. **Argument Overlays vs. Sequential Chains:**
   The v2.0 schema introduced `form` as a list to support sequential inference chains (e.g., generalization followed by application). However, empirical analysis showed natural prose exhibits *overlays* rather than sequences: single arguments that can be legitimately described by multiple valid perspectives (e.g., an argument that is structurally a `disjunctive syllogism` but pragmatically a `false dilemma`). Mining inter-annotator disagreements to train multi-label overlay classifiers represents a promising next research direction.
2. **Span-Level Premise Tagging:**
   Currently, premises are extracted as normalized text strings. Future iterations could adopt span-level token tagging (character start/end offsets) to guarantee verbatim alignment with source documents.
3. **Downstream Validity Checking:**
   Decoupled from extraction, a downstream symbolic or neuro-symbolic engine can take the extracted JSON structure and verify formal validity via automated theorem provers (e.g., Z3 or Lean).

---

## 7. Compliance & Licensing

- **Model Derivative:** Fine-tuned from `Qwen2.5-3B-Instruct`. Distributed under the Alibaba Cloud Qwen Research License Agreement (Non-Commercial Research Use Only).
- **Attribution:** Built with Qwen.
- **Dataset Attribution:** Fine-tuning data incorporates the LOGIC benchmark (Jin et al., EMNLP 2022), licensed under MIT.
- See repository [`LICENSE`](../LICENSE) and [`THIRD_PARTY_NOTICES.md`](../THIRD_PARTY_NOTICES.md) for full legal text.
