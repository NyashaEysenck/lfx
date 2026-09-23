#!/usr/bin/env bash
# End-to-end v7 pipeline: chunked training, Ollama packaging, push, Vertex eval, scoring.
#
# Usage:
#   caffeinate -is scripts/run_v7_pipeline.sh
#
# NOTE: Free Colab requires the laptop lid to remain open and the machine awake.
set -euo pipefail

TAG="v7"
BASE="unsloth/Qwen2.5-3B-Instruct"
EPOCHS=6
REGISTRY="nyasha_stino/lfx:v7-q8"
BENCH="data/benchmark/lfx_bench_v1.jsonl"
PREDS="results/preds_v7_bench_full.jsonl"

echo "=================================================="
echo "==> Step 1: Chunked training ($EPOCHS epochs on Colab T4)"
echo "=================================================="
scripts/run_chunked.sh "$TAG" "$BASE" "$EPOCHS"

echo "=================================================="
echo "==> Step 2: Packaging adapter to Ollama (q8_0)"
echo "=================================================="
pipeline/deploy/to_ollama.sh "$TAG" "models/lora_$TAG" "$BASE" \
  models/mlx_v5_8bit/prompt.txt q8_0

echo "=================================================="
echo "==> Step 3: Register and push to Ollama registry"
echo "=================================================="
ollama cp "$TAG" "$REGISTRY"
ollama push "$REGISTRY"

echo "=================================================="
echo "==> Step 4: Remote evaluation on Vertex AI"
echo "=================================================="
pipeline/deploy/vertex_eval.sh "$BENCH" "$PREDS" 4 "$REGISTRY"

echo "=================================================="
echo "==> Step 5: Scoring on benchmark"
echo "=================================================="
.venv/bin/python pipeline/eval/evaluate.py "$BENCH" "$PREDS"

echo "=================================================="
echo "==> v7 pipeline complete!"
echo "=================================================="
