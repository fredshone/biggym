from ml_collections import config_dict


def get_config():
    config = config_dict.ConfigDict()
    config.SEED = 42
    config.LR = 0.008
    config.GAMMA = 0.99  # 0.95
    config.EPS = 1
    config.NUM_EPISODES = 5000  # 100000  # 5000

    config.EPS_DECAY = 0.95

    # config.WANDB = "disabled"  # "online" if want it to work
    config.WANDB = "online"

    config.WANDB_ENTITY = "jamesr-j"  # change this to your wandb username

    config.PLOT = True

    # FOR PSRL
    config.MU = 0.0
    config.LAMBDA = 1.0
    config.ALPHA = 1.0
    config.BETA = 0.9
    config.KAPPA = 1.0
    config.TAU = 20
    config.MAX_ITER = 100

    return config
