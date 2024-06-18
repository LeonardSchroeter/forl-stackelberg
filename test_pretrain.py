from testing.test_pretrain_ppo import test_pretrain_contextual
from testing.test_pretrain_rl2 import test_pretrain_rl2
from utils.config_util import load_config

if __name__ == "__main__":
    config = load_config()
    if config.algo == "rl2":
        test_pretrain_rl2(config)
    elif config.algo == "ppo":
        test_pretrain_contextual(config)
