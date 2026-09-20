#!/usr/bin/env bash
# Score a split on Vertex AI instead of on the laptop.
#
#   pipeline/deploy/vertex_eval.sh data/benchmark/lfx_bench_v1.jsonl \
#       results/preds_remote.jsonl 4
#
# Why CPU. This project has ZERO GPU quota on every accelerator type -- L4, T4 and
# A100 all reject a custom job instantly with RESOURCE_EXHAUSTED, and
# GPUS_ALL_REGIONS is 0 for Compute Engine too. Raising it is a console request we
# are not making. A quantised 3B on 16 vCPUs is slower than the laptop's GPU in
# wall-clock terms, which is the whole point: it is not the laptop.
#
# Sharding is free speed. Cost is per machine-hour, so four machines for a quarter
# of the time costs the same as one for all of it.
#
# The model comes from the Ollama registry rather than a bucket, so publishing
# turned out to be infrastructure: no weight storage, no container build, no auth.
# Note `:3b-q4` does NOT exist -- that push was killed mid-upload and never
# retried, so `:3b-q8` is the only usable tag despite being the worse choice for
# CPU inference.
#
# Verified equivalent: on a shared sample the remote run agreed with the local MLX
# run on 8 of 8 items. It is a drop-in replacement, not an approximation.
set -euo pipefail

GOLD="${1:?usage: vertex_eval.sh <gold.jsonl> <out.jsonl> [shards] [model] [machine]}"
OUT="${2:?}"
SHARDS="${3:-4}"
MODEL="${4:-nyasha_stino/lfx:3b-q8}"
MACHINE="${5:-n2-standard-16}"

P="${GOOGLE_CLOUD_PROJECT:-agentic-school-506719}"
R="${GOOGLE_CLOUD_LOCATION:-us-central1}"
B="gs://lfx-$P"
STAMP=$(date +%Y%m%d_%H%M%S)
NAME=$(basename "$GOLD" .jsonl)
WORK="${TMPDIR:-/tmp}/lfx_vertex_$STAMP"; mkdir -p "$WORK"

echo "==> uploading inputs"
gcloud storage cp "$GOLD" "$B/bench/${NAME}_$STAMP.jsonl" >/dev/null
gcloud storage cp pipeline/deploy/vertex_runner.sh "$B/code/vertex_runner.sh" >/dev/null

JOBS=()
for i in $(seq 0 $((SHARDS - 1))); do
  cat > "$WORK/shard$i.yaml" <<YAML
workerPoolSpecs:
  - machineSpec:
      machineType: $MACHINE
    replicaCount: 1
    containerSpec:
      imageUri: us-docker.pkg.dev/deeplearning-platform-release/gcr.io/base-cpu
      command: ["/bin/bash","-c"]
      args:
        - "gsutil cp $B/code/vertex_runner.sh /tmp/r.sh && chmod +x /tmp/r.sh && /tmp/r.sh $B/bench/${NAME}_$STAMP.jsonl $B/preds/${NAME}_${STAMP}_$i.jsonl $MODEL $i $SHARDS"
YAML
  out=$(gcloud ai custom-jobs create --project="$P" --region="$R" \
        --display-name="lfx-eval-$STAMP-$i" --config="$WORK/shard$i.yaml" 2>&1)
  J=$(grep -oE "customJobs/[0-9]+" <<<"$out" | head -1 | cut -d/ -f2)
  [ -n "$J" ] || { echo "submit failed:"; echo "$out" | tail -5; exit 1; }
  JOBS+=("$J")
  echo "    shard $i -> job $J"
done

echo "==> waiting (each shard provisions for a few minutes before it starts)"
for J in "${JOBS[@]}"; do
  while true; do
    S=$(gcloud ai custom-jobs describe "projects/$P/locations/$R/customJobs/$J" \
        --project="$P" --format="value(state)" 2>/dev/null || echo UNKNOWN)
    case "$S" in
      JOB_STATE_SUCCEEDED) echo "    $J done"; break;;
      JOB_STATE_FAILED|JOB_STATE_CANCELLED)
        echo "    $J $S -- logs:"
        gcloud logging read "resource.type=\"ml_job\" AND resource.labels.job_id=\"$J\"" \
          --project="$P" --limit=10 --format="value(textPayload)" | tail -6
        exit 1;;
      *) sleep 45;;
    esac
  done
done

echo "==> downloading and merging"
: > "$OUT"
for i in $(seq 0 $((SHARDS - 1))); do
  gcloud storage cp "$B/preds/${NAME}_${STAMP}_$i.jsonl" "$WORK/p$i.jsonl" >/dev/null
  cat "$WORK/p$i.jsonl" >> "$OUT"
done
echo "    $(wc -l < "$OUT") predictions -> $OUT"
echo
echo "score it locally (seconds of CPU):"
echo "  python3 pipeline/eval/evaluate.py $GOLD $OUT"
