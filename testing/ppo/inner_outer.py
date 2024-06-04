from wrappers.single_agent_follower import *

from utils.config_util import load_config_args_overwrite

from training.ppo.inner_outer import maybe_load_model

def test(config):
    (
        _,
        leader_env,
        _,
        leader_model,
        _,
        _,
    ) = maybe_load_model(config, follower_training_config={})

    if config.env.name == "drone_game":
        leader_env.plant.headless = False
        leader_env.plant.sleep_time = 0.5

    # play a single episode to check learned leader and follower policies
    obs, _ = leader_env.reset()
    while True:
        action = leader_model.predict(obs, deterministic=True)[0]
        new_obs, rewards, terminated, truncated, _ = leader_env.step(action)
        print(obs, action, rewards)
        obs = new_obs

        if terminated or truncated:
            break
    if config.env.name == "drone_game":
        leader_env.plant.close(video_name="complete_round.avi")


if __name__ == "__main__":
    config = load_config_args_overwrite("configs/ppo.yml")
    test(config)
