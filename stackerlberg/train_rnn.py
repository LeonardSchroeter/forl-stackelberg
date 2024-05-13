import numpy as np
from sb3_contrib import RecurrentPPO

from wrappers.follower import FollowerWrapperMetaRL
from wrappers.single_agent import SingleAgentFollowerWrapper
from envs.matrix_game import IteratedMatrixGame

import wandb
from wandb.integration.sb3 import WandbCallback

run = wandb.init(
    project="forl-stackelberg-rnn",
    sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
)

if __name__ == "__main__":
    env = IteratedMatrixGame(matrix="prisoners_dilemma", episode_length=4, memory=2)
    env = FollowerWrapperMetaRL(
        env,
        num_episodes=10,
        zero_leader_reward=True,
        zero_follower_reward=False,
        min_reward=-1.5,
        max_reward=1.5,
    )
    env = SingleAgentFollowerWrapper(env, recursively_set_leader_response=False)

    # model = RecurrentPPO.load("checkpoints/follower_ppo_rnn_matrix", env=env)
    model = RecurrentPPO(
        "MlpLstmPolicy",
        env=env,
        verbose=1,
        learning_rate=lambda progress: 1e-3 * progress + 1e-5 * (1 - progress),
        tensorboard_log=f"runs/{run.id}",
    )
    model.learn(total_timesteps=30_000, 
                progress_bar=True, 
                callback=WandbCallback(
                    gradient_save_freq=100,
                    model_save_path=f"models/{run.id}",
                    verbose=2,)
                )
    model.save("checkpoints/follower_ppo_rnn_matrix")
    # env = model.get_env()

    # cell and hidden state of the LSTM
    lstm_states = None
    num_envs = 1
    # Episode start signals are used to reset the lstm states
    episode_starts = np.ones((num_envs,), dtype=bool)

    obs, _ = env.reset(leader_response=[1, 0, 0, 1, 1])

    while True:
        action, lstm_states = model.predict(
            obs, state=lstm_states, episode_start=episode_starts, deterministic=True
        )
        new_obs, reward, done, _, _ = env.step(action)

        print(obs, action, reward)

        obs = new_obs

        if done:
            episode_starts = np.ones((num_envs,), dtype=bool)
            break
        else:
            episode_starts = np.zeros((num_envs,), dtype=bool)
