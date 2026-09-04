#!/usr/bin/env bash
# Drive the baseline -> fine-tune -> re-eval cycle on a Colab T4 from this machine.
#
#   ./run_colab.sh          full cycle
#   ./run_colab.sh stop     release the runtime
set -euo pipefail

S=lf-train
C="${COLAB:-.venv/bin/colab} --auth adc"    # ADC: same credentials as the Vertex labeling

if [ "${1:-run}" = "stop" ]; then $C stop -s "$S"; exit 0; fi

if $C sessions 2>/dev/null | grep -q "\[$S\]"; then
  echo "==> reusing running session $S"
else
  echo "==> provisioning T4"
  $C new -s "$S" --gpu T4
fi

echo "==> installing deps (slow: unsloth pulls a lot)"
$C install -s "$S" unsloth trl peft datasets bitsandbytes accelerate

echo "==> uploading data + scripts"
for f in train.chat.jsonl val.chat.jsonl test_gen.chat.jsonl test_real.chat.jsonl \
         test_gen.jsonl test_real.jsonl; do
  $C upload -s "$S" "data/splits/$f" "/content/$f"
done
for f in train_lora.py predict.py; do
  $C upload -s "$S" "pipeline/train/$f" "/content/$f"
done

# Baseline runs FIRST, against the untouched base model. Half the deliverable is
# the before/after comparison, and a baseline measured after training has been
# near the environment is not a baseline.
step () {                       # step <marker> <file> <timeout> <label>
  echo "==> $4"
  out=$($C exec -s "$S" -f "$2" --timeout "$3" 2>&1)
  echo "$out"
  if ! grep -q "$1" <<<"$out"; then
    echo "FAILED at: $4  (no '$1' marker — see the traceback above)" >&2
    exit 1
  fi
}

step "BASELINE DONE" scripts/colab_steps/baseline.py 3600 "baseline (un-tuned model, both test sets)"

step "TRAIN DONE" scripts/colab_steps/train.py 7200 "LoRA fine-tune, 3 epochs"

step "TUNED DONE" scripts/colab_steps/tuned.py 3600 "tuned model, same script / prompts / sampling as the baseline"

echo "==> downloading predictions"
for f in preds_base_real preds_base_gen preds_tuned_real preds_tuned_gen; do
  $C download -s "$S" "/content/$f.jsonl" "./results/$f.jsonl"
done

echo
echo "========== test_real (real prose — the number that matters) =========="
.venv/bin/python pipeline/eval/evaluate.py data/splits/test_real.jsonl results/preds_base_real.jsonl results/preds_tuned_real.jsonl
echo
echo "========== test_gen (in-distribution) =========="
.venv/bin/python pipeline/eval/evaluate.py data/splits/test_gen.jsonl results/preds_base_gen.jsonl results/preds_tuned_gen.jsonl
echo
echo "Runtime still up (billed as Colab compute). Release: ./run_colab.sh stop"
