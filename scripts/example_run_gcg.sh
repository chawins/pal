#!/bin/bash
export WANDB_MODE=disabled
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

ATTACK="gcg"
# ATTACK="ral"

python -u main.py \
    --config="./configs/${ATTACK}.py" \
    --config.batch_size=512 \
    --config.num_steps=30 \
    --config.log_freq=1 \
    -- \
    --scenario "Toxicity" --behaviors 0 --system_message "llama_default" \
    --model llama-2@~/data/models/Llama-2-7b-chat-hf --verbose

echo "Finished."


# Run PAL without fine-tuning against GPT-3.5-turbo-0613 on behavior 0.
python -u main.py \
    --config="./configs/bproxy.py" \
    --config.add_space=False \
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
    --config.proxy_device="(0,)" \
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
    --scenario AdvBenchAll \
    --log_dir "./results/" \
    --model "chatgpt@gpt-3.5-turbo-0613" \
    --seed 20 \
    --init_suffix_path "data/init_suffix/AdvBenchAll/vicuna.jsonl" \
    --behaviors 0 \
    --system_message helpful \
    --custom_name init


# Run GCG++
python -u main.py \
    --config="./configs/gcg.py" \
    --config.add_space=True \
    --config.batch_size=512 \
    --config.num_steps=500 \
    --config.skip_mode="visited" \
    --config.log_freq=1 \
    -- \
    --scenario AdvBenchAll --behaviors 0 --system_message llama_default \
    --model llama-2@~/data/models/Llama-2-7b-chat-hf