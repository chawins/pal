#!/bin/bash
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
export TOKENIZERS_PARALLELISM=true
export OPENAI_API_KEY="YOUR_OPENAI_API_KEY"
export TIKTOKEN_CACHE_DIR="$HOME/.cache/tiktoken"

python test_suffixes_main.py \
    "results/Llama-2-7b-chat-hf/ral/len20_100000query_bs64_seed20_l50_cw-one-0.001_static_visited_uniform_t1.0_c1" \
    --max_tokens 256 --max_queries 25000 --behaviors 9
