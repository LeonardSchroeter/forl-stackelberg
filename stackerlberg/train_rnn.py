import numpy as np
from sb3_contrib import RecurrentPPO

from wrappers.follower import FollowerWrapperMetaRL
from wrappers.single_agent import SingleAgentFollowerWrapper
from envs.matrix_game import IteratedMatrixGame


if __name__ == "__main__":
    env = IteratedMatrixGame(matrix="prisoners_dilemma", episode_length=10, memory=2)
    env = FollowerWrapperMetaRL(env, num_episodes=2, zero_leader_reward=True)
    env = SingleAgentFollowerWrapper(env, recursively_set_leader_response=False)

    model = RecurrentPPO(
        "MlpLstmPolicy", env=env, verbose=1, policy_kwargs={"n_lstm_layers": 2}
    )
    model.learn(total_timesteps=25_000, progress_bar=True)
    model.save("checkpoints/follower_ppo_rnn")
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
