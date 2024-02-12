from configs.base import get_config as default_config


def get_config():
    # Original white-box GCG attack from Zou et al. (2023)
    config = default_config()
    config.name = "gcg"
    # Top k tokens to sample from
    config.topk = 256
    config.num_coords = (1, 1)  # Number of coordinates to change in one step
    config.mu = 0.0  # Momentum parameter
    config.sample_mode = "orig"
    config.skip_mode = "none"  # "none", "visited", "seen"
    return config
