from configs.base import get_config as default_config


def get_config():
    # RAL attack (black-box)
    config = default_config()
    config.name = "ral"
    config.token_dist = "uniform"
    config.token_probs_temp = 1.0
    config.num_coords = (1, 1)
    # For black-box attacks, log every step. One successful attack is enough.
    config.log_freq = 1
    config.sample_mode = "rand"
    config.skip_mode = "visited"
    return config
