import numpy as np
from stable_baselines3 import PPO
from sb3_contrib import RecurrentPPO

from wrappers.follower import FollowerWrapperMetaRL
from wrappers.single_agent import (
    SingleAgentFollowerWrapper,
    SingleAgentLeaderWrapper,
)
from envs.matrix_game import IteratedMatrixGame

import wandb
from wandb.integration.sb3 import WandbCallback

run = wandb.init(
    project="forl-stackelberg-rnn",
    sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
)


def build_follower_env():
    follower_env = IteratedMatrixGame(
        matrix="prisoners_dilemma", episode_length=30, memory=2
    )
    follower_env = FollowerWrapperMetaRL(
        follower_env,
        num_episodes=3,
        zero_leader_reward=True,
        zero_follower_reward=True,
        min_reward=-1.5,
        max_reward=1.5,
    )
    follower_env = SingleAgentFollowerWrapper(
        follower_env, recursively_set_leader_response=False
    )
    return follower_env


def pretrain(follower_env):
    follower_model = RecurrentPPO(
        "MlpLstmPolicy",
        env=follower_env,
        verbose=1,
        # learning_rate= 1e-3,
        learning_rate=lambda progress: 1e-3 * progress + 1e-5 * (1 - progress),
        tensorboard_log=f"runs/{run.id}",
    )
    follower_model.learn(
        total_timesteps=100_000,
        progress_bar=True,
        callback=WandbCallback(
            gradient_save_freq=100,
            model_save_path=f"models/{run.id}",
            verbose=2,
        ),
    )
    follower_model.save("checkpoints/follower_ppo_rnn_matrix")


def test_pretrain(follower_env):
    follower_model = RecurrentPPO.load(
        "checkpoints/follower_ppo_rnn_matrix", env=follower_env
    )

    # cell and hidden state of the LSTM
    lstm_states = None
    num_envs = 1
    # Episode start signals are used to reset the lstm states
    episode_starts = np.ones((num_envs,), dtype=bool)

    obs, _ = follower_env.reset(leader_response=[0, 0, 0, 0, 0])

    while True:
        action, lstm_states = follower_model.predict(
            obs, state=lstm_states, episode_start=episode_starts, deterministic=True
        )
        new_obs, reward, done, _, _ = follower_env.step(action)

        print(obs, action, reward)

        obs = new_obs

        if done:
            episode_starts = np.ones((num_envs,), dtype=bool)
            break
        else:
            episode_starts = np.zeros((num_envs,), dtype=bool)


def build_leader_env(follower_env):

    follower_model = RecurrentPPO.load("checkpoints/follower_ppo_rnn_matrix", env=follower_env)
    leader_env = SingleAgentLeaderWrapper(
        follower_env.env,
        queries=[0, 1, 2, 3, 4],
        follower_model=follower_model,
    )

    return leader_env

def train(leader_env):

    model = PPO(
        "MlpPolicy", leader_env, verbose=1, tensorboard_log=f"runs/{run.id}"
    )
    model.learn(
        total_timesteps=30_000,
        callback=WandbCallback(gradient_save_freq=100, verbose=2),
    )
    model.save("checkpoints/leader_ppo_rnn_matrix")

def test_train(leader_env):

    leader_model = PPO.load("checkpoints/leader_ppo_rnn_matrix", env=leader_env)
    # play a single episode to check learned leader and follower policies
    obs, _ = leader_env.reset()
    while True:
        action = leader_model.predict(obs, deterministic=True)[0]
        new_obs, rewards, terminated, truncated, _ = leader_env.step(action)
        print(obs, action, rewards)
        obs = new_obs

        if terminated or truncated:
            break


if __name__ == "__main__":
    follower_env = build_follower_env()
    # pretrain(follower_env=follower_env)
    test_pretrain(follower_env=follower_env)

    # leader_env = build_leader_env(follower_env=follower_env)
    # train(leader_env=leader_env)
    # test_train(leader_env=leader_env)
