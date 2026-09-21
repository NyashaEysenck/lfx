#!/usr/bin/env bash
# The model-capacity lever, parameterised: same corpus, same splits, same recipe,
# base model as the ONLY variable. run_3b.sh did this for Qwen2.5-3B with the base
# name hardcoded in five places; comparing sizes meant copying five files and
# trusting that nothing else drifted. It is one argument now, because a capacity
# comparison is only worth reading if the recipe is provably identical.
#
#   scripts/run_size.sh 1_5b unsloth/Qwen2.5-1.5B-Instruct
#   scripts/run_size.sh 3b   unsloth/Qwen2.5-3B-Instruct
#
# RESUMABLE: every long step runs detached on the VM (see colab_steps/lfjob.py)
# and this script only polls, so killing the script or losing the network costs
# nothing -- re-run and it reattaches.
#
# **THE LAPTOP MUST STAY AWAKE.** This said "closing the laptop costs nothing" and
# that was wrong. The JOB is detached from the shell; the SESSION is not detached
# from the machine, because the Colab CLI holds it open with a local keep-alive.
# A v6 run died this way: lid closed at 22:16, keep_alive_error at 22:19, session
# gone at step 108 of 868. `colab log -s <session>` shows the sequence.
#
# So run it with the lid open, and prevent idle sleep:
#
#     caffeinate -is scripts/run_size.sh v6 unsloth/Qwen2.5-3B-Instruct 4
#
# caffeinate stops IDLE sleep but not clamshell sleep -- on battery, closing the
# lid sleeps regardless. Sessions on this account have run 20-29 hours when the
# machine stayed awake, so there is no short Colab limit to design around; the
# "~55 minute free T4 life" recorded earlier in this project is not a Colab limit
# and should not be treated as one.
set -euo pipefail

TAG="${1:?usage: run_size.sh <tag> <base-model> [epochs]}"
BASE="${2:?usage: run_size.sh <tag> <base-model> [epochs]}"
EPOCHS="${3:-6}"

S=lf-train; C=".venv/bin/colab --auth adc"
STEPS="${TMPDIR:-/tmp}/lf_steps_$TAG"; mkdir -p "$STEPS"

# The detached-job cells are three lines each and differ only in the base model,
# so they are generated rather than kept as near-duplicate files per size. This
# script is the record of what ran.
cat > "$STEPS/launch.py" <<PY
"""Detached fine-tune of $BASE. Idempotent: re-running while alive reports
RUNNING rather than starting a second trainer on the same GPU."""
JOB = "train_$TAG"
CMD = ("python train_lora.py --base $BASE "
       "--epochs $EPOCHS --out lora_$TAG")
exec(open("/content/lfjob.py").read())
PY

cat > "$STEPS/pack.py" <<PY
"""Tar the adapter so it comes off the VM in one download, BEFORE prediction
runs. A finished adapter was lost once to a VM reclaim during the longer
prediction step; training is the expensive part."""
import os, subprocess
if not os.path.isdir("/content/lora_$TAG"):
    print("PACK MISSING")
else:
    subprocess.run(["tar", "czf", "/content/lora_$TAG.tar.gz", "-C", "/content",
                    "lora_$TAG"], check=True)
    print("PACK DONE", os.path.getsize("/content/lora_$TAG.tar.gz"), "bytes")
PY

cat > "$STEPS/tuned.py" <<PY
"""Detached prediction pass for the $TAG adapter."""
JOB = "tuned_$TAG"
CMD = ("python predict.py --split test_real --base $BASE "
       "--adapter lora_$TAG --out preds_${TAG}_real.jsonl && "
       "python predict.py --split test_gen --base $BASE "
       "--adapter lora_$TAG --out preds_${TAG}_gen.jsonl")
exec(open("/content/lfjob.py").read())
PY

cell () { $C exec -s "$S" -f "$1" --timeout 120 2>&1; }

if $C status -s "$S" >/dev/null 2>&1 && $C ls -s "$S" /content >/dev/null 2>&1; then
  echo "==> reusing live session $S"; FRESH=0
else
  echo "==> provisioning fresh T4"
  $C stop -s "$S" >/dev/null 2>&1 || true
  $C new -s "$S" --gpu T4
  FRESH=1
fi

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
$C upload -s "$S" scripts/colab_steps/lfjob.py /content/lfjob.py

run_job () {  # run_job <local-cell> <label>
  local FAILS=0
  echo "==> $2"
  cell "$1" | tail -3
  while true; do
    # A transient exec failure must not kill the poller: that is what pruned the
    # session record once, orphaning a VM we could no longer address.
    out=$(cell scripts/colab_steps/poll.py || true)
    if [ -z "$out" ] || grep -q "not found\|Traceback (most recent call last):.*colab" <<<"$out"; then
      FAILS=$((FAILS + 1))
      # A blip and a dead session look identical here, so tolerating failures
      # forever means a reclaimed VM is polled until someone notices. One run
      # spent 11 hours and 39 retries on a session that had been gone since
      # minute 17. Ten consecutive failures is well past any real blip.
      if [ "$FAILS" -ge 10 ]; then
        echo "    session gone after $FAILS consecutive poll failures -- giving up" >&2
        $C status -s "$S" 2>&1 | head -2 >&2
        echo "FAILED at: $2 (session lost)" >&2; return 1
      fi
      echo "    ...poll failure $FAILS/10, retrying  [$(date +%H:%M:%S)]"; sleep 60; continue
    fi
    FAILS=0
    if grep -q "JOB OK" <<<"$out"; then echo "    done: $2"; return 0; fi
    if grep -q "JOB FAILED\|DEAD" <<<"$out"; then
      echo "$out"; echo "FAILED at: $2" >&2; return 1
    fi
    prog=$(grep -oE "[0-9]+/[0-9]+ \[[^]]*\]|'"'"'loss'"'"': [0-9.]+.*'"'"'epoch'"'"': [0-9.]+" <<<"$out" | tail -1 || true)   # grep exits 1 before the first step; set -e would kill the poller
    echo "    ...$(tail -1 <<<"$out")  ${prog:+| $prog}  [$(date +%H:%M:%S)]"
    sleep 120
  done
}

run_job "$STEPS/launch.py" "fine-tune $BASE, $EPOCHS epochs on the v2.0 corpus"

echo "==> securing the adapter"
cell "$STEPS/pack.py" | tail -2
mkdir -p models
$C download -s "$S" "/content/lora_$TAG.tar.gz" "./models/lora_$TAG.tar.gz"
tar xzf "models/lora_$TAG.tar.gz" -C models/ && rm "models/lora_$TAG.tar.gz"
echo "    adapter saved to models/lora_$TAG"

run_job "$STEPS/tuned.py" "predictions on test_real and test_gen"

echo "==> downloading"
mkdir -p results
for f in "preds_${TAG}_real" "preds_${TAG}_gen"; do
  $C download -s "$S" "/content/$f.jsonl" "./results/$f.jsonl"
done
echo "DOWNLOADS DONE"
