#!/bin/zsh

set -euo pipefail

repo_root=${0:A:h}
cd "$repo_root"

mkdir -p logs

timestamp=$(date +"%Y%m%d_%H%M%S")
run_id="run_${timestamp}"

echo "RUN_ID=${run_id}"
echo "Log file: logs/${run_id}.txt"

RUN_ID="$run_id" \
DATA_PATH=./data/datasets/fineweb10B_sp1024/ \
TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model \
VOCAB_SIZE=1024 \
torchrun --standalone --nproc_per_node=8 train_gpt.py "$@"
