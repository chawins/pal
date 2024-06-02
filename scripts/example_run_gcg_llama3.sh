#!/bin/bash
export WANDB_MODE=disabled
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

# Set path to tokenizer.model downloaded from https://llama.meta.com/llama-downloads/.
# Tokenizer from transformers (4.42) is not correct.
export LLAMA3_TOKENIZER_PATH="$HOME/data/models/llama3_tokenizer.model"

ATTACK="gcg"

python -u main.py \
    --config="./configs/${ATTACK}.py" \
    --config.add_space=False \
    --config.batch_size=512 \
    --config.num_steps=10 \
    --config.log_freq=1 \
    --config.fixed_params=True \
    -- \
    --scenario "AdvBenchAll" --behaviors 0 --system_message "helpful" \
    --model "llama-3@~/data/models/Meta-Llama-3-8B-Instruct" --verbose

echo "Finished."
