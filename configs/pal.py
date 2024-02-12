from ml_collections import config_dict

from configs.gpp import get_config as default_config


def get_config():
    # PAL attack (black-box)
    config = default_config()
    config.name = "pal"
    config.sample_mode = "rand"
    config.skip_mode = "visited"
    # "ce-one" and "cw-one" are black-box attack loss functions through API.
    # It computes the loss one token at a time. Options: "ce-all", "cw-all".
    config.loss_func = "cw-one"
    config.cw_margin = 1e-3
    config.num_coords = (1, 1)  # Number of coordinates to change in one step
    config.mu = 0.0  # Momentum parameter
    # Name and/or path to proxy models. Must be a tuple. Specify multiple models
    # will use ensemble of models (experimental).
    config.proxy_model = ("llama-2@~/data/models/Llama-2-7b-chat-hf",)
    # System message to use for proxy models
    config.proxy_system_message = ("llama_default",)
    # Device to place the proxy models on. Can be ["cuda:1", "cuda:2"], etc.
    config.proxy_device = ("cuda",)
    config.proxy_tune_period = 10  # Frequency of querying target model
    config.log_target_loss = False
    config.num_target_query_per_step = -1
    config.use_mp = True  # Use multiprocessing
    # For black-box attacks, log every step. One successful attack is enough.
    config.log_freq = 1
    # Loss function for proxy model only (default: <same as config.loss_func>)
    config.proxy_loss_func = config_dict.FieldReference(None, field_type=str)

    # ========================== Fine-tuning params ========================== #
    # Whether to fine-tune the proxy model. If False, all the following params
    # are ignored.
    config.finetune = False
    # Parameter-efficient fine-tuning. Options: "none" (full FT), "noembed"
    # (full FT, freeze token embedding), "lora" (LoRA)
    config.peft = "noembed"
    config.proxy_dtype = "bfloat16"
    config.proxy_optimizer = "adamw"
    config.proxy_lr = 2e-5
    config.proxy_wd = 1e-1
    config.gradient_clipping = 1.0
    config.lr_schedule = "constant"
    config.proxy_tune_bs = 32  # Batch size for fine-tuning
    # Number of steps to fine-tune. -1 means 1 epoch (each sample is seen once).
    config.proxy_tune_steps = -1
    # Loss function for fine-tuning. Options: "ce", "cw"
    config.proxy_loss = "ce"
    # Whether to also tune on past queries, aggregated from all steps.
    config.tune_on_past_samples = False
    # Output sequence length to compute loss for fine-tuning, similar to seq_len
    # but for fine-tuning.
    config.tune_seq_len = 50
    config.tune_temperature = 1.0
    return config
