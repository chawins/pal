#!/bin/bash
export WANDB_MODE=disabled
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

# Run PAL with fine-tuning against Llama-2-7B-chat on behavior 0.
# Requires two A100 GPUs.
python -u main.py \
    --config="./configs/pal.py" \
    --config.add_space=True \
    --config.adv_suffix_init="! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! !" \
    --config.batch_size=128 \
    --config.cw_margin=1.0 \
    --config.early_stop=False \
    --config.finetune=False \
    --config.fixed_params=True \
    --config.init_suffix_len=20 \
    --config.log_freq=1 \
    --config.log_target_loss=True \
    --config.loss_func="cw-one" \
    --config.max_queries=25000 \
    --config.mini_batch_size=128 \
    --config.num_steps=2000 \
    --config.num_target_query_per_step=32 \
    --config.peft="noembed" \
    --config.proxy_device="(1,)" \
    --config.proxy_dtype="bfloat16" \
    --config.proxy_model="('vicuna_v1.1@~/data/models/vicuna-7b-v1.5-16k',)" \
    --config.proxy_system_message="('vicuna_default',)" \
    --config.proxy_tune_bs=32 \
    --config.proxy_tune_period=1 \
    --config.sample_mode="rnd" \
    --config.seq_len=50 \
    --config.skip_mode="visited" \
    --config.tune_on_past_samples=False \
    --config.tune_seq_len=50 \
    -- \
    --scenario "AdvBenchAll" \
    --log_dir "./results/" \
    --model "llama-2@~/data/models/Llama-2-7b-chat-hf" \
    --seed 20 \
    --init_suffix_path "data/init_suffix/AdvBenchAll/vicuna-7b-v1.5-16k_gcg_len20.jsonl" \
    --behaviors 0 \
    --system_message "helpful" \
    --custom_name "init"

echo "Finished."
