import os

import wandb

from wrappers.single_agent_leader import *

from utils.checkpoint_util import maybe_load_checkpoint_ppo
from utils.config_util import load_config_args_overwrite
from utils.drone_leader_observation import decimal_to_binary

from training.ppo.pretrain import build_follower_env


def build_leader_env(config, follower_env=None, follower_model=None):
    if follower_env is None:
        follower_env = build_follower_env(config)

    if follower_model is None:
        follower_model, _ = maybe_load_checkpoint_ppo(
            os.path.join(config.training.checkpoint_path, "follower"), follower_env
        )

    if config.env.name == "matrix_game":
        if config.training.no_initseg:
            leader_env = LeaderWrapperNoInitialSegment(
                follower_env.env,
                queries=[0, 1, 2, 3, 4],
                follower_model=follower_model,
            )
        else:
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
        if config.training.no_initseg:
            leader_env = LeaderWrapperNoInitialSegment(
                follower_env.env,
                queries=queries,
                follower_model=follower_model,
            )
        else:
            leader_env = SingleAgentQueryLeaderWrapper(
                follower_env.env,
                queries=queries,
                follower_model=follower_model,
            )

    return leader_env


def train(config):
    leader_env = build_leader_env(config)

    if config.training.log_wandb:
        run = wandb.init(project="stackelberg-ppo-leader", sync_tensorboard=True)

    leader_model, callback_list = maybe_load_checkpoint_ppo(
        os.path.join(config.training.checkpoint_path, "leader_noinitseg")
        if config.training.no_initseg
        else os.path.join(config.training.checkpoint_path, "leader"),
        leader_env,
        config.training.log_wandb,
        run_id=run.id,
    )

    if config.training.no_initseg:
        leader_env.set_leader_model(leader_model)

    leader_model.learn(
        total_timesteps=30_000,
        reset_num_timesteps=False,
        callback=callback_list,
    )


if __name__ == "__main__":
    config = load_config_args_overwrite("configs/ppo.yml")
    train(config)
