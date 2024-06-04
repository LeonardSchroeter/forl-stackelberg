import os

from utils.config_util import load_config_args_overwrite
from utils.setup_experiment import create_env, get_policy_net_for_inference
from utils.checkpoint_util import maybe_load_checkpoint_ppo

from wrappers.rl2.trial_wrapper import TrialWrapper
from wrappers.rl2.leader import SingleAgentLeaderWrapperMetaRL

import wandb

def train(env, config):
    run = wandb.init(project="stackelberg-rl2-leader", sync_tensorboard=True)

    model, callback_list = maybe_load_checkpoint_ppo(
        os.path.join(config.training.checkpoint_path, "leader"),
        env,
        config.training.log_wandb,
        run_id=run.id,
    )

    total_timesteps = 1000_0000

    model.learn(
        total_timesteps=total_timesteps,
        callback=callback_list,
        reset_num_timesteps=False,
    )

if __name__ == "__main__":

    config = load_config_args_overwrite("configs/rl2.yml")

    follower_env = create_env(config=config)

    policy_net = get_policy_net_for_inference(follower_env, config)

    env = TrialWrapper(follower_env._env, num_episodes=3)
    env = SingleAgentLeaderWrapperMetaRL(env, follower_policy_net=policy_net)

    train(env, config)
