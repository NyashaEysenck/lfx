#!/usr/bin/env bash
# Package a trained adapter as an Ollama model, end to end.
#
#   pipeline/deploy/to_ollama.sh lfx-3b models/lora_3b \
#       unsloth/Qwen2.5-3B-Instruct models/mlx_v5_8bit/prompt.txt
#
# **Do not use `ollama create` directly on a merged safetensors directory.** Its
# built-in importer produces a broken model for this checkpoint: it reports
# success, `ollama show --system` returns the right prompt, the architecture and
# parameter count are right, and generation is pure noise. The same directory
# scores perfectly under transformers. llama.cpp's convert_hf_to_gguf.py converts
# it correctly, so the GGUF is built first and Ollama is pointed at that.
#
# Finding that cost four builds. Ruled out along the way, none of them the cause:
# q4 being too aggressive (F16 was equally broken), tied embeddings with no
# lm_head (llama.cpp handles tying natively -- verified by rebuilding without the
# workaround and scoring identically), and missing BPE vocab files. The test that
# actually split the problem was pulling a registry model and watching it work,
# which separates "our model is broken" from "Ollama is broken here".
#
# Quantisation is cheap for this task and was measured, not assumed: q4_K_M costs
# 0.008 form accuracy against MLX 8-bit over 485 records. It costs considerably
# more on premise spans (-0.042) and the suppressed-premise call (-0.052).
set -euo pipefail

NAME="${1:?usage: to_ollama.sh <name> <adapter> <base> <prompt.txt> [quant]}"
ADAPTER="${2:?}"; BASE="${3:?}"; PROMPT="${4:?}"; QUANT="${5:-q4_K_M}"

MERGED="models/merged_${NAME}"
WORK="${TMPDIR:-/tmp}/lfx_gguf_${NAME}"; mkdir -p "$WORK"
GGUF="$WORK/${NAME}-f16.gguf"
LLAMA="${LLAMA_CPP:-$WORK/llama.cpp}"

# Ollama.app ships its own CLI; a Homebrew `ollama` can be many versions behind
# the running server, which prints a warning and is worth avoiding here.
OLLAMA="${OLLAMA_BIN:-/Applications/Ollama.app/Contents/Resources/ollama}"
[ -x "$OLLAMA" ] || OLLAMA=ollama

if [ ! -d "$LLAMA" ]; then
  echo "==> fetching llama.cpp converter (sparse, ~4 MB)"
  git clone --depth 1 --filter=blob:none --sparse \
      https://github.com/ggml-org/llama.cpp.git "$LLAMA"
  git -C "$LLAMA" sparse-checkout set --skip-checks gguf-py conversion
fi

if [ ! -d "$MERGED" ]; then
  echo "==> merging $ADAPTER into $BASE"
  python3 pipeline/train/merge_adapter.py --base "$BASE" --adapter "$ADAPTER" --out "$MERGED"
fi

echo "==> converting to GGUF (llama.cpp, NOT ollama's importer)"
python3 "$LLAMA/convert_hf_to_gguf.py" "$MERGED" --outfile "$GGUF" --outtype f16

echo "==> writing Modelfile"
python3 pipeline/deploy/make_modelfile.py --model "$GGUF" --prompt "$PROMPT" \
    --base "$BASE" --adapter "$ADAPTER" --out "Modelfile.$NAME"

echo "==> ollama create $NAME ($QUANT)"
$OLLAMA create "$NAME" -f "Modelfile.$NAME" --quantize "$QUANT"

echo
echo "built $NAME. Verify it before trusting it:"
echo "  python3 pipeline/eval/predict_ollama.py --split test_real --model $NAME \\"
echo "      --out results/preds_ollama_${NAME}_real.jsonl"
echo "  python3 pipeline/eval/evaluate.py data/splits/test_real.jsonl \\"
echo "      results/preds_ollama_${NAME}_real.jsonl"
echo "The GGUF in $WORK can be deleted once the model is created."
