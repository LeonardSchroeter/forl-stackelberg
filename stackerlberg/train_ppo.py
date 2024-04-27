from stable_baselines3 import PPO

from envs.matrix_game import IteratedMatrixGame
from wrappers.single_agent import (
    SingleAgentFollowerWrapper,
    SingleAgentLeaderWrapper,
)
from wrappers.follower import FollowerWrapper

if __name__ == "__main__":
    env_follower = FollowerWrapper(
        IteratedMatrixGame(matrix="prisoners_dilemma", episode_length=10, memory=2),
        num_queries=5,
    )
    env_follower = SingleAgentFollowerWrapper(env_follower)

    model = PPO("MlpPolicy", env_follower, verbose=1)
    model.learn(total_timesteps=50_000)
    model.save("checkpoints/follower_ppo")

    env_leader = FollowerWrapper(
        IteratedMatrixGame(matrix="prisoners_dilemma", episode_length=10, memory=2),
        num_queries=5,
    )
    env_leader = SingleAgentLeaderWrapper(
        env_leader, queries=[0, 1, 2, 3, 4], follower_model=model
    )

    leader_model = PPO("MlpPolicy", env_leader, verbose=1)
    leader_model.learn(total_timesteps=50_000)
    leader_model.save("checkpoints/leader_ppo")

    # play a single episode to check learned leader and follower policies
    obs, _ = env_leader.reset()
    while True:
        action = leader_model.predict(obs, deterministic=True)[0]
        new_obs, rewards, terminated, truncated, _ = env_leader.step(action)
        print(obs, action, rewards)
        obs = new_obs

        if terminated or truncated:
            break
