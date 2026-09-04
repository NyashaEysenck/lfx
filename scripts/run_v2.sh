#!/usr/bin/env bash
set -euo pipefail
S=lf-train; C=".venv/bin/colab --auth adc"
step () { echo "==> $4"; out=$($C exec -s "$S" -f "$2" --timeout "$3" 2>&1); echo "$out"
  grep -q "$1" <<<"$out" || { echo "FAILED at: $4" >&2; exit 1; }; }
# A listed session can still be a dead VM — Colab reclaims runtimes and the local
# session file goes stale. Probe the kernel before trusting it.
if $C status -s "$S" >/dev/null 2>&1 && $C ls -s "$S" /content >/dev/null 2>&1; then
  echo "==> reusing live session $S"
else
  echo "==> provisioning fresh T4"
  $C stop -s "$S" >/dev/null 2>&1 || true
  $C new -s "$S" --gpu T4
fi
echo "==> deps"; $C install -s "$S" unsloth trl peft datasets bitsandbytes accelerate
echo "==> uploading v2 data"
for f in train.chat.jsonl val.chat.jsonl test_gen.chat.jsonl test_real.chat.jsonl \
         test_gen.jsonl test_real.jsonl; do
  $C upload -s "$S" "data/splits/$f" "/content/$f"
done
for f in train_lora.py predict.py; do
  $C upload -s "$S" "pipeline/train/$f" "/content/$f"
done
step "TRAIN DONE" scripts/colab_steps/train_v2.py 7200 "LoRA fine-tune on 592 examples, 6 epochs"
step "TUNED DONE" scripts/colab_steps/tuned_v2.py 3600 "predictions from lora_v2"
for f in preds_v2_real preds_v2_gen; do $C download -s "$S" "/content/$f.jsonl" "./results/$f.jsonl"; done
echo "DOWNLOADS DONE"
