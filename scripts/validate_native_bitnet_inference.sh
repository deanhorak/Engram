#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 4 ]]; then
  echo "usage: $0 PACKAGE MLP_LIBRARY ATTENTION_LIBRARY OUTPUT_DIRECTORY" >&2
  exit 2
fi

package=$1
mlp_library=$2
attention_library=$3
output_directory=$4
mkdir -p "$output_directory"
export PYTHONPATH=src

python -m engram.cli evaluate-native-bitnet-attention \
  --model "$package" --dataset tests/fixtures/confirmation_expanded.jsonl \
  --out "$output_directory/combined_frozen_confirmation.json" \
  --library "$mlp_library" --attention-library "$attention_library" \
  --threads 12 --native-projections --sequence-count 8 \
  --prediction-positions 256 --record-offset 8 --modes native_streaming \
  --local-window 16 --retrieval-candidates 8 --retrieval-top-k 4 --sink-tokens 2

python -m engram.cli evaluate-native-bitnet-generation \
  --model "$package" --prompts tests/fixtures/inference_prompts.jsonl \
  --out "$output_directory/sustained_generation.json" --max-tokens 16 \
  --mlp-library "$mlp_library" --attention-library "$attention_library" \
  --threads 12

python -m engram.cli benchmark-native-bitnet-generation \
  --model "$package" --out "$output_directory/extended_context.json" \
  --lengths 512 2048 --max-tokens 1 --mlp-library "$mlp_library" \
  --attention-library "$attention_library" --threads 12 --native-projections
