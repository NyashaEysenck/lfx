#!/usr/bin/env bash
# Defect fix run: does more training resolve the form-into-argument_type field swap?
set -euo pipefail
S=lf-train
C=".venv/bin/colab --auth adc"

step () { echo "==> $4"; out=$($C exec -s "$S" -f "$2" --timeout "$3" 2>&1); echo "$out"
  grep -q "$1" <<<"$out" || { echo "FAILED at: $4" >&2; exit 1; }; }

echo "==> installing deps"
$C install -s "$S" unsloth trl peft datasets bitsandbytes accelerate
echo "==> uploading"
for f in train.chat.jsonl val.chat.jsonl test_gen.chat.jsonl test_real.chat.jsonl \
         test_gen.jsonl test_real.jsonl; do
  $C upload -s "$S" "data/splits/$f" "/content/$f"
done
for f in train_lora.py predict.py; do
  $C upload -s "$S" "pipeline/train/$f" "/content/$f"
done
step "TRAIN DONE" scripts/colab_steps/train6.py 7200 "LoRA fine-tune, 6 epochs"
step "TUNED DONE" scripts/colab_steps/tuned6.py 3600 "predictions from the 6-epoch adapter"
for f in preds_e6_real preds_e6_gen; do $C download -s "$S" "/content/$f.jsonl" "./results/$f.jsonl"; done
echo "DOWNLOADS DONE"
