import os

from utils.drone_leader_observation import decimal_to_binary
from utils.checkpoint_util import maybe_load_checkpoint_ppo
from utils.config_util import load_config

from training.ppo.train_leader import build_leader_env_ppo
from training.rl2.train_leader import build_leader_env_rl2


def test_leader(config):
    if config.algo == "ppo":
        env = build_leader_env_ppo(config)
    elif config.algo == "rl2":
        env = build_leader_env_rl2(config)

    model, _ = maybe_load_checkpoint_ppo(os.path.join(config.checkpoint_path, "leader"), env)

    if config.no_initseg:
        env.set_leader_model(model)

    if config.env == "drone_game":
        env.plant.headless = False
        env.plant.sleep_time = 0.2

    if config.leader_test_env:
        # play a single episode to check learned leader and follower policies
        obs, _ = env.reset()
        while True:
            action = model.predict(obs, deterministic=True)[0]
            new_obs, rewards, terminated, truncated, _ = env.step(action)
            # print(obs, action, rewards)
            obs = new_obs

            if terminated or truncated:
                break
        if config.env == "drone_game":
            env.plant.close(video_name="complete_round.avi")

    if config.env == "matrix_game":
        leader_policy = [
            model.predict(obs, deterministic=True)[0].item() for obs in range(5)
        ]
        print(leader_policy)
    elif config.env == "drone_game" and not config.env_config.leader_cont:
        for o in range(2**env.observation_space.n):
            o_bin = decimal_to_binary(o, width=env.observation_space.n)
            action = model.predict(o_bin, deterministic=True)[0]
            print(f"obs: {o_bin}, act: {action}")


if __name__ == "__main__":
    config = load_config("rl2")
    test_leader(config)
