#!/bin/bash
export WANDB_MODE=disabled
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

ATTACK="gcg"

python -u main.py \
    --config="./configs/${ATTACK}.py" \
    --config.add_space=True \
    --config.batch_size=512 \
    --config.mini_batch_size=256 \
    --config.num_steps=10 \
    --config.log_freq=1 \
    --config.fixed_params=True \
    -- \
    --scenario "AdvBenchAll" --behaviors 0 --system_message "llama_default" \
    --model llama-2@~/data/models/Llama-2-7b-chat-hf --verbose

echo "Finished."
