#!/usr/bin/env bash
# v1.2 corpus retrain. Run from the repo root.
set -euo pipefail
S=lf-train; C=".venv/bin/colab --auth adc"
step () { echo "==> $4"; out=$($C exec -s "$S" -f "$2" --timeout "$3" 2>&1); echo "$out"
  grep -q "$1" <<<"$out" || { echo "FAILED at: $4" >&2; exit 1; }; }
if $C status -s "$S" >/dev/null 2>&1 && $C ls -s "$S" /content >/dev/null 2>&1; then
  echo "==> reusing live session $S"
else
  echo "==> provisioning fresh T4"; $C stop -s "$S" >/dev/null 2>&1 || true
  $C new -s "$S" --gpu T4
fi
echo "==> deps"; $C install -s "$S" unsloth trl peft datasets bitsandbytes accelerate
echo "==> uploading v1.2 splits"
for f in train.chat.jsonl val.chat.jsonl test_gen.chat.jsonl test_real.chat.jsonl \
         test_gen.jsonl test_real.jsonl; do
  $C upload -s "$S" "data/splits/$f" "/content/$f"
done
for f in train_lora.py predict.py; do $C upload -s "$S" "pipeline/train/$f" "/content/$f"; done
step "TRAIN DONE" scripts/colab_steps/train_v4.py 7200 "LoRA fine-tune on 741 examples, 6 epochs"
step "TUNED DONE" scripts/colab_steps/tuned_v4.py 3600 "predictions from lora_v4"
for f in preds_v4_real preds_v4_gen; do $C download -s "$S" "/content/$f.jsonl" "./results/$f.jsonl"; done
echo "DOWNLOADS DONE"
