#!/bin/bash
export WANDB_MODE=disabled
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

# Run GCG++
python -u main.py \
    --config="./configs/gpp.py" \
    --config.add_space=True \
    --config.batch_size=512 \
    --config.num_steps=500 \
    --config.log_freq=1 \
    -- \
    --scenario "AdvBenchAll" --behaviors 0 --system_message "llama_default" \
    --model "llama-2@~/data/models/Llama-2-7b-chat-hf" --verbose
