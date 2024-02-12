from ml_collections import config_dict


def get_config():
    # Base attack config
    config = config_dict.ConfigDict()
    # Attack name
    config.name: str = "base"
    config.log_freq: int = 10
    config.adv_suffix_init: str = "! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! !"
    # Init suffix length (auto-generated from adv_suffix_init)
    config.init_suffix_len: int = -1
    config.num_steps: int = 500
    # Used fixed scenario params in each iteration
    config.fixed_params: bool = True
    config.allow_non_ascii: bool = False
    # Number of candidates to evaluate in each step
    config.batch_size: int = 512
    # Split a batch of candidates into minibatch to save memory.
    # -1 for full batch (config.batch_size)
    config.mini_batch_size: int = -1
    config.seq_len: int = 50  # Max sequence length for computing loss
    config.loss_temperature: float = 1.0  # Temperature for computing loss
    config.loss_func: str = "ce-all"  # "ce-all", "ce-one", "cw-all", "cw-one"
    config.cw_margin: float = 1e-3  # Margin for CW loss
    # Max number of queries (default: -1 for no limit)
    config.max_queries: float = -1
    # Whether to skip candidates: "none", "visited" (skip candidates that
    # optimization steps have reached), "seen" (skip candidates that we have
    # computed loss on -- superset of "visited")
    config.skip_mode: str = "none"  # "none", "visited", "seen"
    # Add metaspace in front of target string. Set to True if target model is
    # Llama-2. Otherwise, it's generally better to set to False.
    config.add_space: bool = True
    config.monotonic: bool = False  # Only update if loss improves
    config.early_stop: bool = False  # Stop if loss does not improve
    return config
