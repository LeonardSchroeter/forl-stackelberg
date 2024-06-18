import os

import wandb
import numpy as np
from gymnasium import spaces

from envs.matrix_game import IteratedMatrixGame

from envs.drone_game import DroneGame, DroneGameEnv
from wrappers.single_agent_follower import FollowerWrapperInfoSample, SingleAgentFollowerWrapper

from wrappers.follower import ContextualPolicyWrapper

from utils.checkpoint_util import maybe_load_checkpoint_ppo
from utils.config_util import load_config


def build_matrix_game_contextual(config):
    return ContextualPolicyWrapper(
        IteratedMatrixGame(
            matrix="prisoners_dilemma",
            episode_length=config.episode_len,
            memory=config.memory,
        ),
        num_queries=5,
    )

def build_drone_game_contextual(config):
    env = DroneGameEnv(
        width=config.width, height=config.height, drone_dist=config.drone_dist, agent_view_size=config.agent_view_size
    )
    env = DroneGame(
        env=env,
        headless=True,
        leader_cont=config.leader_cont,
        follower_blind=config.follower_blind,
    )
    if isinstance(env.observation_space("leader"), spaces.MultiBinary):
        num_queries = 2 ** env.observation_space("leader").n
    elif isinstance(env.observation_space("leader"), spaces.MultiDiscrete):
        num_queries = np.prod(env.observation_space("leader").nvec)
    return ContextualPolicyWrapper(env=env, num_queries=num_queries)

def build_follower_env_contextual(config):
    if config.env == "matrix_game":
        follower_env = build_matrix_game_contextual(config.env_config)
    elif config.env == "drone_game":
        follower_env = build_drone_game_contextual(config.env_config)

    if config.inner_outer:
        follower_env = FollowerWrapperInfoSample(follower_env)
    else:
        follower_env = SingleAgentFollowerWrapper(follower_env)

    return follower_env


def pretrain_contextual(config, follower_env=None):
    if follower_env is None:
        follower_env = build_follower_env_contextual(config)

    if config.log_wandb:
        wandb.init(project="stackelberg-ppo-follower", sync_tensorboard=True)

    follower_model, callback_list = maybe_load_checkpoint_ppo(
        os.path.join(config.checkpoint_path, "follower"),
        follower_env,
        config.log_wandb,
        config.algo_config.follower,
    )

    follower_model.learn(
        total_timesteps=300_000,
        reset_num_timesteps=False,
        callback=callback_list,
    )

    return follower_model


if __name__ == "__main__":
    config = load_config("ppo")
    pretrain_contextual(config)
