#!/bin/bash
export WANDB_MODE=disabled
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

ATTACK="gcg"

python -u main.py \
    --config="./configs/${ATTACK}.py" \
    --config.batch_size=512 \
    --config.num_steps=30 \
    --config.log_freq=1 \
    -- \
    --scenario "Toxicity" --behaviors 0 --system_message "llama_default" \
    --model llama-2@~/data/models/Llama-2-7b-chat-hf --verbose

echo "Finished."
