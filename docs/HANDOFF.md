# Handoff — LFX quality phase

State as of 2026-09-22. Read this first in a new conversation, then
`docs/PROJECT_BRIEF.md` for the longer history.

---

## The goal

Improve the model to real quality, as a **learning project whose output is also a
contribution**. A small fine-tuned model on its own is not a contribution —
anyone who wants one prompts a frontier model. The contributions are:

1. **LFX-Bench** — a 22-class argument-form benchmark with provenance on every
   record. No public dataset covers valid inductive forms; every corpus in this
   space labels fallacies only.
2. **The findings** below, which are about how this kind of system gets measured.

Working premise from the user: LLMs reason badly about logic. The approach
avoids relying on model judgement for ground truth wherever it can.

**Constraint:** the user has no capacity for human annotation. Anything that
would need a human in the loop has to be designed around, not delegated.

---

## What was built

| artifact | path | what it is |
|---|---|---|
| symbolic checker | `lfx/formal.py` | derives form from a logical skeleton by rule. 53 self-tests, incl. a 68,952-argument ambiguity scan. `python -m lfx.formal` |
| Tier 1 generator | `pipeline/corpus/build_formal.py` | label specified by construction, verified by blind re-formalisation through the checker. Two-stage (see findings) |
| Tier 1b generator | `pipeline/corpus/build_specified.py` | inductive forms + `hasty generalization` + `equivocation`; verified by two models from different families agreeing blind |
| benchmark | `data/benchmark/lfx_bench_v1.jsonl` | **1521 records, 22/22 classes**, `provenance` field on every row |
| assembler | `pipeline/build/assemble_benchmark.py` | builds the benchmark, checks train/test leaks |
| train assembler | `pipeline/build/assemble_train.py` | merges constructed items into train, checks against benchmark |
| formalisation ablation | `pipeline/eval/formalize_ablation.py` | DIRECT vs FORMALIZE within one model |
| remote eval | `pipeline/deploy/vertex_eval.sh` | scores a split on Vertex CPU; laptop runs nothing |
| chunked training | `scripts/run_chunked.sh` | one epoch per Colab session, adapter carried between |
| evaluator | `pipeline/eval/evaluate.py` | **now reports precision / recall / F1 / macro-F1**, flags over-predicted classes |

### Benchmark composition

| provenance | n | guarantee |
|---|---|---|
| constructed-symbolic | 693 | label by construction; `lfx.formal` re-derives it from a blind formalisation |
| external-human | 485 | LOGIC annotators, ~8–12% inherited noise |
| constructed-consensus | 309 | label by construction; two model families agree blind |
| project-authored | 34 | condensed from public-domain sources here |

**Report per tier and per class, never one average.** Tier accuracies are not
comparable to each other: `constructed-consensus` survived a two-model agreement
filter and is biased toward items models find easy.

---

## Results

### v5 vs v6 vs v7 (retrained with informal fallacy rebalance), full 1521-record benchmark

| tier | n | v5 | v6 | v7 | delta (v6→v7) |
|---|---|---|---|---|---|
| constructed-symbolic | 693 | 0.743 | 0.987 | 0.988 | +0.001 |
| constructed-consensus | 309 | 0.796 | 0.932 | 0.929 | −0.003 |
| **external-human** | **485** | **0.763** | **0.625** | **0.781** | **+0.157** (McNemar z=7.18) |
| project-authored | 34 | 0.412 | 0.441 | 0.412 | −0.029 |
| overall accuracy | 1521 | 0.753 | 0.848 | **0.897** | **+0.049** (McNemar z=6.32) |
| **macro-F1** | 22 cls | **0.705** | **0.836** | **0.861** | **+0.025** |

**Verdict on v7:** The dilution hypothesis was confirmed. By adding the 61 labeled
LOGIC rows and upsampling the five informal fallacies ×2 (restoring their training share
from 13% to 27%), **external-human accuracy fully rebounded to 0.781** (exceeding
v5's 0.763), while retaining the constructed-class gains (`reductio` 0.988,
`equivocation` 0.643, `categorical syllogism` 0.989). **v7 is a clean improvement
and is ready to become `latest` on Ollama.**

What v7 fixed:
- `straw man` precision rose from 0.316 → **0.568** (F1 0.466 → **0.681**; predictions dropped from 152 to 81).
- `ad hominem` recall restored from 0.430 → **0.750** (F1 0.593 → **0.843**; confusions to `straw man` collapsed from 93 to 29).
- Retained v6 fixes: `reductio` 0.988, `affirming the consequent` 0.989, `categorical syllogism` 0.989.


---

## Findings worth keeping

These are the contribution. Each is in a commit message with the numbers.

1. **The old evaluation could not see most of the task.** "0.765 on 485 held-out
   arguments" covered 5 of 22 classes. v5 was actually 0.025 on `reductio`,
   0.143 on `equivocation`, 0.000 on `other`.
2. **Same-generator held-out sets hide generator learning.** v5 scored 6/6 on
   `test_gen` reductio and 0/78 on constructed reductio. Three candidate
   mechanisms tested; two ruled out, one correlated but insufficient.
3. **Frontier models read conclusions positionally.** When the conclusion is
   stated first, `gemini-3.1-pro` misreads the argument 35% of the time vs 1.2%
   when stated last (13/37 vs 1/82, Fisher p = 5×10⁻⁷). Errors go both
   directions — not a bias toward validity.
4. **Equivocation detection depends on the question.** Open-ended 22-way
   classification finds 13% (6/45); asking "does any term shift sense?" finds
   93% (42/45). Replicated on a fresh seed: 7% vs 93%.
5. **Formalising is a confidence signal, not a replacement.** Formalise-instead-
   of-classify is worse (0.700 vs 0.769). But when direct naming and
   formalise-then-derive **agree**, accuracy is ~0.98 on two models 16 points
   apart; when they disagree it falls to 0.49–0.73. Needs no gold label.
6. **Recall-only per-class tables hide dumping-ground classes.** v6's `straw man`
   recall rose 0.759 → 0.889 while precision fell ~0.53 → ~0.32.

### Methodological lessons (recorded because they recurred)

- First runs at n≈160 with cells of 20 were repeatedly wrong: 56% repair rate
  (really 35%), "formalising is much worse" (a prompt-format confound — the model
  wrote English into formula fields), "agreement doesn't replicate" (drawn from a
  run with 48% non-random data loss). **Replicate before believing.**
- Filtering items on whether a model reads them correctly discards the hard
  ones. Verify *quality* and record *difficulty* separately.

---

## Infrastructure facts (these cost hours)

- **GPU quota is zero** on `agentic-school-506719` for every accelerator (L4,
  T4, A100) on both Vertex and Compute Engine. The user declined to request more.
  Vertex **CPU** jobs work.
- **Vertex CPU jobs run serially**, not in parallel — a concurrent-CPU cap. Four
  shards ≈ four × 40 min. Sharding gives no speedup here.
- **Gemini 3.x models are served from `global`**, not `us-central1`. A regional
  call returns 404, which reads like a permissions problem.
- **Gemini 3.x-lite models are quota-starved**; `gemini-2.5-flash` handles bulk
  work cleanly. 429 is routine and can return an empty body.
- **Free Colab: the laptop must stay awake and the lid open.** The CLI holds the
  session with a local keep-alive. `caffeinate -is` blocks idle sleep but not
  clamshell sleep. Colab also reclaims after ~60 min of *sustained GPU load* —
  idle sessions last far longer, which is why session histories showing 20–29 h
  were misleading.
- **Ollama no longer quantises at create time from a GGUF.** Quantise during
  conversion instead. `convert_hf_to_gguf.py` offers up to `q8_0`; q4_K_M needs
  `llama-quantize` built from source.
- **Never use `ollama create` directly on safetensors** — its importer produces a
  broken model that generates noise while reporting success. Use llama.cpp's
  converter (`pipeline/deploy/to_ollama.sh` does this).
- **Use `.venv/bin/python`**, not bare `python3`, which now resolves to Homebrew
  Python without `lfx` installed.
- **The user's network intermittently blocks `*.googleapis.com`** while
  `www.google.com` works — DNS resolves, TCP never connects. When active, every
  Gemini / Vertex / GCS / OAuth call hangs silently with no error. Test with:
  ```
  curl -s -o /dev/null -m 10 -w "%{http_code}\n" https://oauth2.googleapis.com/
  ```
  `000` = blocked. It has cleared by itself before.

---

## Current state

**Registry (`nyasha_stino/lfx`):** `v7-q8` is published and tested; `latest` and `3b-q8` still point to v5 (ready to be updated to v7).

**On disk:** `models/lora_3b` (v5 adapter), `models/lora_v6` (v6 adapter), `models/lora_v7` (v7 adapter), `models/merged_v7` (6.18 GB, regenerable — safe to delete), `results/preds_v7_bench_full.jsonl`.

**Completed in this phase:**
- 61 LOGIC rows labeled into `data/interim/logic_rebalance_labeled.jsonl`.
- v7 training data assembled in `data/splits/train.jsonl` (2069 examples; informal fallacies upsampled ×2 to 27%).
- 6-epoch chunked training completed via `scripts/run_chunked.sh` on Colab T4.
- Model packaged to GGUF q8_0 via `pipeline/deploy/to_ollama.sh` and pushed to `nyasha_stino/lfx:v7-q8`.
- Full 1521-item benchmark evaluated on Vertex AI CPU and scored with `pipeline/eval/evaluate.py`.

---

## Next steps, in order

1. **Promote v7 to `latest` and `3b-q8` on Ollama:**
   ```bash
   ollama cp v7 nyasha_stino/lfx:latest && ollama push nyasha_stino/lfx:latest
   ollama cp v7 nyasha_stino/lfx:3b-q8 && ollama push nyasha_stino/lfx:3b-q8
   ```

2. **Clean up disk space:**
   `models/merged_v6` and `models/merged_v7` are ~12 GB total and fully regenerable from adapters; delete them if disk space is needed.

### Later

- **Schema v3.0:** emit `form` **and** `formalization`, recompute with
  `lfx.formal`, surface disagreement as a confidence flag (finding 5).
- **Qwen2.5-7B** is Apache-2.0 (removes the non-commercial licence) — blocked on
  GPU; needs quota or longer chunked Colab runs.
- **Rebuild and push `3b-q4`** for faster CPU evaluation — needs
  `llama-quantize`.
- **README** should lead with the benchmark and findings, not the model.
- Update `docs/PROJECT_BRIEF.md`: its "~55-minute free T4 limit" is right for
  sustained GPU load but was contradicted and then reconfirmed this phase.
