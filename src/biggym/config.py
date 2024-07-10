from ml_collections import config_dict


def get_config():
    config = config_dict.ConfigDict()
    config.SEED = 42
    config.LR = 0.008
    config.GAMMA = 0.99  # 0.95
    config.EPS_START = 1
    config.EPS_END = 0
    config.NUM_EPISODES = 5000  # 100000  # 5000

    config.EPS_DELAY = 0.95
    config.EPS_DECAY = 100000

    config.WANDB = "disabled"  # "online" if want it to work
    # config.WANDB = "online"

    config.WANDB_ENTITY = "jamesr-j"  # change this to your wandb username

    config.PLOT = True

    # For R2D2
    config.HIDDEN_SIZE = 64
    config.LOOKUP_STEP = 20  # TODO what is this?
    config.RANDOM_UPDATE = True  # If you want to do random update instead of sequential update
    config.MIN_EP_NUM = 20
    config.EP_BATCH_SIZE = 8

    # For D3QN
    config.BATCH_SIZE = 128
    config.BUFFER_SIZE = 2 ** 13

    # FOR PSRL
    config.MU = 0.0
    config.LAMBDA = 1.0
    config.ALPHA = 1.0
    config.BETA = 0.9
    config.KAPPA = 1.0
    config.TAU = 20
    config.MAX_ITER = 100

    return config
