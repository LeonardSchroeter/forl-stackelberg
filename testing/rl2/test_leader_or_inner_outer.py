import os

from utils.drone_leader_observation import decimal_to_binary
from utils.config_util import load_config_args_overwrite
from utils.checkpoint_util import maybe_load_checkpoint_ppo
from utils.setup_experiment import create_env, get_policy_net_for_inference

from wrappers.rl2.trial_wrapper import TrialWrapper
from wrappers.rl2.leader import SingleAgentLeaderWrapperMetaRL


def test(env, config):
    model, _ = maybe_load_checkpoint_ppo(
        os.path.join(config.training.checkpoint_path, "inner_outer", "leader")
        if config.training.rl2_inner_outer
        else os.path.join(config.training.checkpoint_path, "leader"),
        env,
    )

    if config.env.name == "drone_game":
        env.plant.headless = False
        env.plant.sleep_time = 0.5

    obs, _ = env.reset()
    while True:
        action = model.predict(obs, deterministic=True)[0]
        new_obs, reward, terminated, truncated, _ = env.step(action)
        print(obs, action, reward)
        obs = new_obs
        if terminated or truncated:
            break

    if config.env.name == "matrix_game":
        leader_policy = [
            model.predict(obs, deterministic=True)[0].item() for obs in range(5)
        ]
        print(leader_policy)
    elif config.env.name == "drone_game":
        for o in range(2**env.observation_space.n):
            o_bin = decimal_to_binary(o, width=env.observation_space.n)
            action = model.predict(o_bin, deterministic=True)[0]
            print(f"obs: {o_bin}, act: {action}")


if __name__ == "__main__":
    config = load_config_args_overwrite("configs/rl2.yml")

    follower_env = create_env(config=config)

    policy_net = get_policy_net_for_inference(follower_env, config)

    env = TrialWrapper(follower_env._env, num_episodes=3)
    env = SingleAgentLeaderWrapperMetaRL(env, follower_policy_net=policy_net)

    test(env, config)
