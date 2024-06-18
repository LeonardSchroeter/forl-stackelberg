import os

import wandb

from utils.checkpoint_util import maybe_load_checkpoint_ppo
from utils.config_util import load_config

from training.ppo.pretrain import build_follower_env_contextual
from training.ppo.train_leader import build_leader_env_ppo


def maybe_load_model(config):
    follower_env = build_follower_env_contextual(config)

    follower_model, follower_callback_list = maybe_load_checkpoint_ppo(
        os.path.join(config.checkpoint_path, "follower"),
        follower_env,
        config.log_wandb,
        config.algo_config.follower,
    )

    leader_env = build_leader_env_ppo(
        config=config, follower_env=follower_env, follower_model=follower_model
    )

    leader_model, leader_callback_list = maybe_load_checkpoint_ppo(
        os.path.join(config.checkpoint_path, "leader"),
        leader_env,
        log_wandb=False,
        config=config.algo_config.leader,
    )

    return (
        follower_env,
        leader_env,
        follower_model,
        leader_model,
        follower_callback_list,
        leader_callback_list,
    )


def train_inner_outer_contextual(config):
    
    if config.log_wandb:
        wandb.init(project="stackelberg-ppo-inner-outer", sync_tensorboard=True)
    
    (
        follower_env,
        leader_env,
        follower_model,
        leader_model,
        follower_callback_list,
        leader_callback_list,
    ) = maybe_load_model(config)

    leader_env.set_follower_model(follower_model)
    follower_env.set_leader_model(leader_model)

    for _ in range(100):
        
        follower_model.learn(
            total_timesteps=5000,
            reset_num_timesteps=False,
            callback=follower_callback_list,
        )
        
        leader_model.learn(
            total_timesteps=3000,
            reset_num_timesteps=False,
            callback=leader_callback_list,
        )


if __name__ == "__main__":
    config = load_config("ppo")
    train_inner_outer_contextual(config)
