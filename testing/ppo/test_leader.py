import os

from utils.checkpoint_util import maybe_load_checkpoint_ppo
from utils.config_util import load_config_args_overwrite

from training.ppo.train_leader import build_leader_env

def test_leader(config):
    env = build_leader_env(config)

    model, _ = maybe_load_checkpoint_ppo(
        os.path.join(config.training.checkpoint_path, "leader_noinitseg")
        if config.training.no_initseg
        else os.path.join(config.training.checkpoint_path, "leader"), env
    )

    if config.training.no_initseg:
        env.set_leader_model(model)

    if config.env.name == "drone_game":
        env.plant.headless = False
        env.plant.sleep_time = 0.5

    # play a single episode to check learned leader and follower policies
    obs, _ = env.reset()
    while True:
        action = model.predict(obs, deterministic=True)[0]
        new_obs, rewards, terminated, truncated, _ = env.step(action)
        print(obs, action, rewards)
        obs = new_obs

        if terminated or truncated:
            break
    if config.env.name == "drone_game":
        env.plant.close(video_name="complete_round.avi")


if __name__ == "__main__":
    config = load_config_args_overwrite("configs/ppo.yml")
    test_leader(config)
