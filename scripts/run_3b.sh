#!/usr/bin/env bash
# Phase 5: does a bigger base model move the number? Qwen2.5-3B against the v4
# recipe, same corpus and splits, so the base model is the only variable.
#
# RESUMABLE. Every long step runs detached on the VM (see colab_steps/lfjob.py)
# and this script only polls. Killing it, closing the laptop or losing the
# network costs nothing: re-run it and it reattaches to the live session and
# picks up wherever the VM has got to. Three earlier runs were lost to exactly
# that, because `colab exec` used to block for the whole two hours.
set -euo pipefail
S=lf-train; C=".venv/bin/colab --auth adc"

cell () {  # cell <file> -> stdout, 120s is ample now that nothing blocks
  $C exec -s "$S" -f "$1" --timeout 120 2>&1
}

if $C status -s "$S" >/dev/null 2>&1 && $C ls -s "$S" /content >/dev/null 2>&1; then
  echo "==> reusing live session $S"
  FRESH=0
else
  echo "==> provisioning fresh T4"
  $C stop -s "$S" >/dev/null 2>&1 || true
  $C new -s "$S" --gpu T4
  FRESH=1
fi

# Uploads are cheap but not free; skip them when the VM already has the corpus.
if [ "$FRESH" = 1 ] || ! $C ls -s "$S" /content/train.chat.jsonl >/dev/null 2>&1; then
  echo "==> deps"
  $C install -s "$S" unsloth trl peft datasets bitsandbytes accelerate
  echo "==> uploading splits and pipeline"
  for f in train.chat.jsonl val.chat.jsonl test_gen.chat.jsonl test_real.chat.jsonl \
           test_gen.jsonl test_real.jsonl; do
    $C upload -s "$S" "data/splits/$f" "/content/$f"
  done
  for f in train_lora.py predict.py; do
    $C upload -s "$S" "pipeline/train/$f" "/content/$f"
  done
fi
# lfjob.py is the job-control runtime; always refresh it.
$C upload -s "$S" scripts/colab_steps/lfjob.py /content/lfjob.py

run_job () {  # run_job <launch-script> <label>
  echo "==> $2"
  cell "$1" | tail -3
  while true; do
    # A transient exec failure must not kill the poller: that is what pruned the
    # session record last time, orphaning a VM we could no longer address.
    out=$(cell scripts/colab_steps/poll_3b.py || true)
    if [ -z "$out" ] || grep -q "not found\|Traceback (most recent call last):.*colab" <<<"$out"; then
      echo "    ...transient poll failure, retrying  [$(date +%H:%M:%S)]"; sleep 60; continue
    fi
    if grep -q "JOB OK" <<<"$out"; then echo "    done: $2"; return 0; fi
    if grep -q "JOB FAILED\|DEAD" <<<"$out"; then
      echo "$out"; echo "FAILED at: $2" >&2; return 1
    fi
    # surface the trainer's own progress line, not just "RUNNING", so the
    # question "how far along is it?" has a real answer at any moment.
    prog=$(grep -oE "[0-9]+/[0-9]+ \[[^]]*\]|'"'"'loss'"'"': [0-9.]+.*'"'"'epoch'"'"': [0-9.]+" <<<"$out" | tail -1 || true)   # grep exits 1 before the first step; set -e would kill the poller
    echo "    ...$(tail -1 <<<"$out")  ${prog:+| $prog}  [$(date +%H:%M:%S)]"
    sleep 120
  done
}

run_job scripts/colab_steps/launch_3b.py       "fine-tune Qwen2.5-3B, 6 epochs on 741 examples"
# Pull the adapter down BEFORE the prediction step. Training is the expensive
# part and a lost runtime must never cost it again.
echo "==> securing the adapter"
cell scripts/colab_steps/pack_3b.py | tail -2
mkdir -p models
$C download -s "$S" /content/lora_3b.tar.gz ./models/lora_3b.tar.gz
tar xzf models/lora_3b.tar.gz -C models/ && rm models/lora_3b.tar.gz
echo "    adapter saved to models/lora_3b"

run_job scripts/colab_steps/launch_tuned_3b.py "predictions on test_real and test_gen"

echo "==> downloading"
for f in preds_3b_real preds_3b_gen; do
  $C download -s "$S" "/content/$f.jsonl" "./results/$f.jsonl"
done
echo "DOWNLOADS DONE"
