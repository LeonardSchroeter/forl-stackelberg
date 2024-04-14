from typing import Union
import numpy as np

from envs.matrix_game import IteratedMatrixGame
from wrappers.follower import FollowerWrapper


def pretrain(matrix: Union[np.ndarray, str]):
    for epoch in range(100):
        env = IteratedMatrixGame(matrix, episode_length=10, memory=2)
        random_leader_responses = [env.action_space("leader").sample() for _ in range(5)]
        env = FollowerWrapper(env, num_queries=5, leader_response=random_leader_responses)

if __name__ == "__main__":
    pretrain("prisoners_dilemma")
