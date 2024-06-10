"""
Script for training stateful meta-reinforcement learning agents
"""
import torch

from utils.config_util import load_config_args_overwrite
from utils.evaluate import evaluate

class LeaderModelContTest:
    def __init__(self) -> None:
        pass
    
    def predict(self, _, deterministic):
        return [0.7]

def test_pretrain():
    
    config = load_config_args_overwrite("configs/rl2.yml")
    
    if config.env.name == "matrix_game":
        evaluate(config, verbose=True, leader_policy=[1, 0, 0, 1, 1])
    elif config.env.name == "drone_game":
        if config.drone_game.leader_cont:
            leader_policy = LeaderModelContTest()
        else:
            leader_policy=[0 for _ in range(2**4)]
        evaluate(
            config,
            verbose=True,
            leader_policy=leader_policy,
            # leader_policy=[1, 1, 1, 1] + [np.random.randint(2) for _ in range(2**10-4)],
            # leader_policy=[0 for _ in range(2**4)]
            # leader_policy= [0, 3, 0, 3, 3, 0, 3, 0, 0, 0, 3, 3, 0, 3, 0, 3]
            # leader_policy= [1, 3, 1, 3, 3, 1, 3, 1, 1, 1, 3, 3, 1, 3, 1, 3]
        )


if __name__ == "__main__":
    test_pretrain()