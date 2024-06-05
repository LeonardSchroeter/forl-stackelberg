"""
Script for training stateful meta-reinforcement learning agents
"""

from utils.config_util import load_config_args_overwrite
from utils.evaluate import evaluate

def test():
    
    config = load_config_args_overwrite("configs/rl2.yml")
    
    if config.env.name == "matrix_game":
        evaluate(config, verbose=True, leader_policy=[1, 0, 0, 1, 1])
    elif config.env.name == "drone_game":
        evaluate(
            config,
            verbose=True,
            # leader_policy=[1, 1, 1, 1] + [np.random.randint(2) for _ in range(2**10-4)],
            leader_policy=[3 for _ in range(2**4)]
            # leader_policy= [0, 3, 0, 3, 3, 0, 3, 0, 0, 0, 3, 3, 0, 3, 0, 3]
            # leader_policy= [1, 3, 1, 3, 3, 1, 3, 1, 1, 1, 3, 3, 1, 3, 1, 3]
        )


if __name__ == "__main__":
    test()
