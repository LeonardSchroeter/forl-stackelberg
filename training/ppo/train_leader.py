import os

import wandb

from wrappers.single_agent_leader import *

from utils.checkpoint_util import maybe_load_checkpoint_ppo
from utils.config_util import load_config_args_overwrite
from utils.drone_leader_observation import decimal_to_binary

from pretrain import build_follower_env

config = load_config_args_overwrite("configs/ppo.yml")


def build_leader_env():
    follower_env = build_follower_env()

    follower_model, _ = maybe_load_checkpoint_ppo(
        os.path.join(config.training.checkpoint_path, "follower"), follower_env
    )

    if config.env.name == "matrix_game":
        leader_env = SingleAgentQueryLeaderWrapper(
            follower_env.env,
            queries=[0, 1, 2, 3, 4],
            follower_model=follower_model,
        )
    elif config.env.name == "drone_game":
        num_queries = 2 ** follower_env.env.observation_space("leader").n
        queries = [
            decimal_to_binary(o, width=follower_env.env.observation_space("leader").n)
            for o in range(num_queries)
        ]
        leader_env = SingleAgentQueryLeaderWrapper(
            follower_env.env,
            queries=queries,
            follower_model=follower_model,
        )

    return leader_env


def train():
    leader_env = build_leader_env()

    if config.training.log_wandb:
        run = wandb.init(project="stackelberg-ppo-leader", sync_tensorboard=True)

    leader_model, callback_list = maybe_load_checkpoint_ppo(
        os.path.join(config.training.checkpoint_path, "leader"),
        leader_env,
        config.training.log_wandb,
        run_id=run.id,
    )

    leader_model.learn(
        total_timesteps=300_00,
        reset_num_timesteps=False,
        callback=callback_list,
    )

    return leader_model


if __name__ == "__main__":
    train()
