import numpy as np
from sb3_contrib import RecurrentPPO

from wrappers.follower import FollowerWrapperMetaRL
from wrappers.single_agent import SingleAgentFollowerWrapper
from envs.matrix_game import IteratedMatrixGame


if __name__ == "__main__":
    env = IteratedMatrixGame(matrix="prisoners_dilemma", episode_length=4, memory=2)
    env = FollowerWrapperMetaRL(
        env,
        num_episodes=3,
        zero_leader_reward=True,
        zero_follower_reward=True,
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
    )
    model.learn(total_timesteps=15_000)
    model.save("checkpoints/follower_ppo_rnn_matrix")
    # env = model.get_env()

    def evaluate(leader_response):
        # cell and hidden state of the LSTM
        lstm_states = None
        num_envs = 1
        # Episode start signals are used to reset the lstm states
        episode_starts = np.ones((num_envs,), dtype=bool)

        obs, _ = env.reset(leader_response=leader_response)

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

    evaluate([1, 0, 0, 1, 1])
    evaluate([0, 0, 0, 0, 0])
