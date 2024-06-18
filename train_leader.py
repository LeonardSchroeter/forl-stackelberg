from training.ppo.train_leader import train_leader_contextual
from training.rl2.train_leader import train_leader_rl2
from utils.config_util import load_config

if __name__ == "__main__":
    config = load_config()
    if config.algo == "rl2":
        train_leader_rl2(config)
    else:
        train_leader_contextual(config)