from ml_collections import config_dict


def get_config():
    config = config_dict.ConfigDict()
    config.SEED = 42
    config.LR = 0.008
    config.GAMMA = 0.95
    config.EPS = 1
    config.NUM_EPISODES = 100000  # 5000

    config.EPS_DECAY = 0.95

    # config.WANDB = "disabled"  # "online" if want it to work
    config.WANDB = "online"

    config.WANDB_ENTITY = "jamesr-j"  # change this to your wandb username

    config.PLOT = True

    return config
