import os

import wandb
import numpy as np

from wrappers.single_agent_leader import LeaderWrapperNoInitialSegment, SingleAgentQueryLeaderWrapper

from utils.checkpoint_util import maybe_load_checkpoint_ppo
from utils.config_util import load_config
from utils.drone_leader_observation import decimal_to_binary, repr_to_coord

from training.ppo.pretrain import build_follower_env


def build_leader_env_ppo(config, follower_env=None, follower_model=None):
    if follower_env is None:
        follower_env = build_follower_env(config)

    if follower_model is None:
        follower_model, _ = maybe_load_checkpoint_ppo(
            os.path.join(config.checkpoint_path, "follower"),
            follower_env,
        )
    if config.env.name == "matrix_game":
        queries = [0, 1, 2, 3, 4]
    elif config.env.name == "drone_game":
        if config.drone_game.leader_cont:
            num_queries = np.prod(follower_env.env.observation_space("leader").nvec)
            queries = [
                repr_to_coord(
                    o,
                    base=follower_env.env.observation_space("leader").nvec[0],
                    width=len(follower_env.env.observation_space("leader")),
                )
                for o in range(num_queries)
            ]
        else:
            num_queries = 2 ** follower_env.env.observation_space("leader").n
            queries = [
                decimal_to_binary(
                    o, width=follower_env.env.observation_space("leader").n
                )
                for o in range(num_queries)
            ]
    if config.no_initseg:
        wrapper = LeaderWrapperNoInitialSegment
    else:
        wrapper = SingleAgentQueryLeaderWrapper
    leader_env = wrapper(
        follower_env.env,
        queries=queries,
        follower_model=follower_model,
    )

    return leader_env


def train(config):
    leader_env = build_leader_env_ppo(config)

    if config.training.log_wandb:
        run = wandb.init(project="stackelberg-ppo-leader", sync_tensorboard=True, monitor_gym=True)

    leader_model, callback_list = maybe_load_checkpoint_ppo(
        os.path.join(config.checkpoint_path, "leader"),
        leader_env,
        config.training.log_wandb,
        run_id=run.id,
    )

    if config.no_initseg:
        leader_env.set_leader_model(leader_model)

    leader_model.learn(
        total_timesteps=300_000,
        reset_num_timesteps=False,
        callback=callback_list,
    )


if __name__ == "__main__":
    config = load_config("ppo")
    train(config)
