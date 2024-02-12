from configs.base import get_config as default_config


def get_config():
    # GCC++ (white-box)
    config = default_config()
    config.name = "gcg"
    # Top k tokens to sample from
    config.topk = 256
    config.num_coords = (1, 1)  # Number of coordinates to change in one step
    config.mu = 0.0  # Momentum parameter
    config.sample_mode = "rand"
    config.skip_mode = "visited"
    config.loss_func = "cw-all"
    config.cw_margin = 1e-3
    return config
