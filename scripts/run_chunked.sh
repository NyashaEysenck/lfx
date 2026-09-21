#!/usr/bin/env bash
# Train across as many Colab sessions as it takes, one epoch per session.
#
#   caffeinate -is scripts/run_chunked.sh v6 unsloth/Qwen2.5-3B-Instruct 6
#
# WHY. A free Colab VM cannot be relied on for a long run, and the failure is not
# one thing to engineer around. Three consecutive attempts at this model died
# differently: a laptop lid closed and stopped the CLI's keep-alive (17 min);
# Colab reclaimed the VM after ~60 minutes of sustained GPU load; and a keep-alive
# failed at 8 minutes with the machine awake and quota intact, for no reason that
# could be recovered from the logs.
#
# Each of those lost the entire run. train_lora.py writes a checkpoint every epoch
# with save_strategy="epoch", but onto the VM -- so the checkpoint dies with the
# thing it was protecting against.
#
# So: one epoch per session, adapter pulled down after each, next epoch started
# from it with --init-adapter. Any failure now costs one epoch, and a failed
# epoch is simply retried. Sizing the job to fit inside the hour, which is what
# the previous attempt did, treats the symptom.
#
# The adapter is ~300 MB for a 3B. A full HF checkpoint with optimizer state
# would be several times that, and the optimizer state is what we choose to drop:
# each chunk restarts Adam's moments and the LR schedule. For LoRA at this scale
# that is a mild, known cost, and strictly better than losing the run.
set -uo pipefail

TAG="${1:?usage: run_chunked.sh <tag> <base> <epochs> [max-attempts-per-epoch]}"
BASE="${2:?}"
EPOCHS="${3:?}"
MAX_TRIES="${4:-3}"

S=lf-chunk-$TAG
C=".venv/bin/colab --auth adc"
ADAPTER="models/lora_$TAG"
mkdir -p models

cell () { $C exec -s "$S" -f "$1" --timeout 180 2>&1; }

ensure_session () {
  if $C status -s "$S" >/dev/null 2>&1 && $C ls -s "$S" /content >/dev/null 2>&1; then
    return 0
  fi
  echo "    provisioning a fresh T4"
  $C stop -s "$S" >/dev/null 2>&1
  $C new -s "$S" --gpu T4 >/dev/null 2>&1 || return 1
  $C install -s "$S" unsloth trl peft datasets bitsandbytes accelerate >/dev/null 2>&1 || return 1
  for f in train.chat.jsonl val.chat.jsonl; do
    $C upload -s "$S" "data/splits/$f" "/content/$f" >/dev/null 2>&1 || return 1
  done
  $C upload -s "$S" pipeline/train/train_lora.py /content/train_lora.py >/dev/null 2>&1 || return 1
  $C upload -s "$S" scripts/colab_steps/lfjob.py /content/lfjob.py >/dev/null 2>&1 || return 1
  # carry the adapter back up so this session continues rather than restarts
  if [ -d "$ADAPTER" ]; then
    echo "    uploading adapter from the previous epoch"
    tar czf "/tmp/$TAG.tar.gz" -C models "lora_$TAG"
    $C upload -s "$S" "/tmp/$TAG.tar.gz" "/tmp/$TAG.tar.gz" >/dev/null 2>&1 || return 1
    cat > /tmp/untar.py <<PY
import subprocess
subprocess.run(["tar","xzf","/tmp/$TAG.tar.gz","-C","/content"], check=True)
print("adapter restored")
PY
    cell /tmp/untar.py >/dev/null 2>&1 || return 1
  fi
  return 0
}

for ep in $(seq 1 "$EPOCHS"); do
  echo "==> epoch $ep/$EPOCHS"
  for try in $(seq 1 "$MAX_TRIES"); do
    ensure_session || { echo "    session setup failed, retry $try"; sleep 30; continue; }

    INIT=""
    [ -d "$ADAPTER" ] && INIT="--init-adapter lora_$TAG"
    cat > /tmp/launch_$TAG.py <<PY
JOB = "chunk_${TAG}_$ep"
CMD = ("python train_lora.py --base $BASE --epochs 1 "
       "--out lora_$TAG $INIT")
exec(open("/content/lfjob.py").read())
PY
    cell /tmp/launch_$TAG.py | tail -2

    FAILS=0; OK=0
    while true; do
      out=$(cell scripts/colab_steps/poll.py || true)
      if [ -z "$out" ] || grep -q "not found\|Traceback (most recent call last):.*colab" <<<"$out"; then
        FAILS=$((FAILS + 1))
        if [ "$FAILS" -ge 8 ]; then echo "    session lost mid-epoch"; break; fi
        sleep 45; continue
      fi
      FAILS=0
      if grep -q "JOB OK" <<<"$out"; then OK=1; break; fi
      if grep -q "JOB FAILED\|DEAD" <<<"$out"; then echo "$out" | tail -4; break; fi
      prog=$(grep -oE "[0-9]+/[0-9]+ \[[^]]*\]" <<<"$out" | tail -1 || true)
      echo "    ...${prog:-running}  [$(date +%H:%M:%S)]"
      sleep 90
    done

    if [ "$OK" = 1 ]; then
      echo "    epoch $ep trained; pulling the adapter down before anything else"
      cat > /tmp/pack_$TAG.py <<PY
import os, subprocess
d = "/content/lora_$TAG"
print("PACK MISSING" if not os.path.isdir(d) else
      ("PACK DONE " + str(subprocess.run(["tar","czf","/content/$TAG.tar.gz",
       "-C","/content","lora_$TAG"], check=True) or os.path.getsize("/content/$TAG.tar.gz"))))
PY
      cell /tmp/pack_$TAG.py | tail -1
      if $C download -s "$S" "/content/$TAG.tar.gz" "./models/$TAG.tar.gz" >/dev/null 2>&1; then
        rm -rf "$ADAPTER"
        tar xzf "models/$TAG.tar.gz" -C models/ && rm -f "models/$TAG.tar.gz"
        echo "    adapter saved -> $ADAPTER"
        break
      fi
      echo "    download failed, retry $try"
    fi
    echo "    epoch $ep attempt $try failed; starting over on a fresh session"
    $C stop -s "$S" >/dev/null 2>&1
    sleep 20
  done
  [ -d "$ADAPTER" ] || { echo "epoch $ep never produced an adapter; stopping" >&2; exit 1; }
done

echo
echo "all $EPOCHS epochs done -> $ADAPTER"
$C stop -s "$S" >/dev/null 2>&1
echo "package it:  python3 pipeline/train/package.py --adapter $ADAPTER --base $BASE --name $TAG"
