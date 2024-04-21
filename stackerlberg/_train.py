import os
from typing import Union
import numpy as np
import torch

from envs.matrix_game import IteratedMatrixGame
from wrappers.follower import FollowerWrapper
from ppo import PPOLeaderFollower

if __name__ == "__main__":
    env = IteratedMatrixGame(matrix="prisoners_dilemma", episode_length=10, memory=2)
    env = FollowerWrapper(env=env, num_queries=5)
    learner = PPOLeaderFollower(env=env)
    learner.pretraining(iterations=100000)

    file_dir = os.path.abspath(os.path.dirname(__file__))
    torch.save(learner.follower_actor.state_dict(), os.path.join(file_dir, "follower_oracle.pth"))