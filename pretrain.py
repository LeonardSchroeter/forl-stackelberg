from training.ppo.inner_outer import train_inner_outer_contextual
from training.ppo.pretrain import pretrain_contextual
from training.rl2.pretrain_or_inner_outer import pretrain_or_inner_outer_rl2
from utils.config_util import load_config

if __name__ == "__main__":
    config = load_config()
    if config.algo == "rl2":
        pretrain_or_inner_outer_rl2(config)
    elif config.algo == "ppo":
        if config.inner_outer:
            train_inner_outer_contextual(config)
        else:
            pretrain_contextual(config)
