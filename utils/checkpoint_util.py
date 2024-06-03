"""
Utility module for saving and loading checkpoints.
"""

import os

import torch as tc

from stable_baselines3 import PPO

from utils.constants import DEVICE

from stable_baselines3.common.callbacks import (
    BaseCallback,
    CheckpointCallback,
    CallbackList,
)
from wandb.integration.sb3 import WandbCallback


def _format_name(kind, steps):
    filename = f"{kind}_{steps}.pth"
    return filename


def _parse_name(filename):
    parts = filename.split(".")[0].split("_")
    kind, steps = parts[0], parts[1]
    steps = int(steps)
    return {"kind": kind, "steps": steps}


def _latest_n_checkpoint_steps(base_path, n=5):
    steps = set(map(lambda x: _parse_name(x)["steps"], os.listdir(base_path)))
    latest_steps = sorted(steps)
    latest_n = latest_steps[-n:]
    return latest_n


def _latest_step(base_path):
    return _latest_n_checkpoint_steps(base_path, n=1)[-1]


def save_checkpoint_rl2(steps, checkpoint_dir, model_name, model, optimizer, scheduler):
    """
    Saves a checkpoint of the latest model, optimizer, scheduler state.
    Also tidies up checkpoint_dir/model_name/ by keeping only last 5 ckpts.

    Args:
        steps: num steps for the checkpoint to save.
        checkpoint_dir: checkpoint dir for checkpointing.
        model_name: model name for checkpointing.
        model: model to be updated from checkpoint.
        optimizer: optimizer to be updated from checkpoint.
        scheduler: scheduler to be updated from checkpoint.

    Returns:
        None
    """
    base_path = os.path.join(checkpoint_dir, model_name)
    os.makedirs(base_path, exist_ok=True)

    model_path = os.path.join(base_path, _format_name("model", steps))
    optim_path = os.path.join(base_path, _format_name("optimizer", steps))
    sched_path = os.path.join(base_path, _format_name("scheduler", steps))

    # save everything
    tc.save(model.state_dict(), model_path)
    tc.save(optimizer.state_dict(), optim_path)
    if scheduler is not None:
        tc.save(scheduler.state_dict(), sched_path)

    # keep only last n checkpoints
    latest_n_steps = _latest_n_checkpoint_steps(base_path, n=5)
    for file in os.listdir(base_path):
        if _parse_name(file)["steps"] not in latest_n_steps:
            os.remove(os.path.join(base_path, file))


def maybe_load_checkpoint_rl2(
    checkpoint_dir, model_name, model, optimizer, scheduler, steps
):
    """
    Tries to load a checkpoint from checkpoint_dir/model_name/.
    If there isn't one, it fails gracefully, allowing the script to proceed
    from a newly initialized model.

    Args:
        checkpoint_dir: checkpoint dir for checkpointing.
        model_name: model name for checkpointing.
        model: model to be updated from checkpoint.
        optimizer: optimizer to be updated from checkpoint.
        scheduler: scheduler to be updated from checkpoint.
        steps: num steps for the checkpoint to locate. if none, use latest.

    Returns:
        number of env steps experienced by loaded checkpoint.
    """
    base_path = os.path.join(checkpoint_dir, model_name)
    try:
        if steps is None:
            steps = _latest_step(base_path)

        model_path = os.path.join(base_path, _format_name("model", steps))
        optim_path = os.path.join(base_path, _format_name("optimizer", steps))
        sched_path = os.path.join(base_path, _format_name("scheduler", steps))

        model.load_state_dict(tc.load(model_path, map_location=DEVICE))
        if optimizer is not None:
            optimizer.load_state_dict(tc.load(optim_path, map_location=DEVICE))
        if scheduler is not None:
            scheduler.load_state_dict(tc.load(sched_path, map_location=DEVICE))

        print(f"Loaded checkpoint from {base_path}, with step {steps}.")
        print("Continuing from checkpoint.")
    except FileNotFoundError:
        print(f"Bad checkpoint or none at {base_path} with step {steps}.")
        print("Running from scratch.")
        steps = 0

    return steps


def maybe_load_checkpoint_ppo(
    checkpoint_path, env, log_wandb=False, training_config=None, run_id=0
):
    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)

    latest_step = _latest_step(checkpoint_path) if os.listdir(checkpoint_path) else 0

    checkpoint_callback = CheckpointCallback(
        save_freq=1000,
        save_path=checkpoint_path,
        name_prefix="model",
        save_replay_buffer=True,
        save_vecnormalize=True,
    )
    rmckp_callback = RmckpCallback(ckp_path=checkpoint_path)

    callback_list = [checkpoint_callback, rmckp_callback]
    if log_wandb:
        wandb_callback = WandbCallback(gradient_save_freq=100, verbose=2)
        callback_list.append(wandb_callback)

    # Resuming
    if os.listdir(checkpoint_path):
        checkpoint_file = checkpoint_path + f"model_{latest_step}_steps.zip"
        print(f"Loading model from: {checkpoint_file}")
        model = PPO.load(
            checkpoint_path + f"model_{latest_step}_steps.zip",
            env=env,
        )
    # Starting from scratch
    else:
        print(f"Starting from scratch. Checkpoint path: {checkpoint_path}")
        model = PPO(
            "MlpPolicy",
            env,
            verbose=1,
            tensorboard_log=f"runs/{run_id}",
            **training_config,
        )
    return model, CallbackList(callback_list)


class RmckpCallback(BaseCallback):
    """
    A custom callback that derives from ``BaseCallback``.

    :param verbose: Verbosity level: 0 for no output, 1 for info messages, 2 for debug messages
    """

    def __init__(self, verbose: int = 0, ckp_path=None):
        super().__init__(verbose)
        # Those variables will be accessible in the callback
        # (they are defined in the base class)
        # The RL model
        # self.model = None  # type: BaseAlgorithm
        # An alias for self.model.get_env(), the environment used for training
        # self.training_env # type: VecEnv
        # Number of time the callback was called
        # self.n_calls = 0  # type: int
        # num_timesteps = n_envs * n times env.step() was called
        # self.num_timesteps = 0  # type: int
        # local and global variables
        # self.locals = {}  # type: Dict[str, Any]
        # self.globals = {}  # type: Dict[str, Any]
        # The logger object, used to report things in the terminal
        # self.logger # type: stable_baselines3.common.logger.Logger
        # Sometimes, for event callback, it is useful
        # to have access to the parent object
        # self.parent = None  # type: Optional[BaseCallback]
        self.ckp_path = ckp_path

    def _on_training_start(self) -> None:
        """
        This method is called before the first rollout starts.
        """
        pass

    def _on_rollout_start(self) -> None:
        """
        A rollout is the collection of environment interaction
        using the current policy.
        This event is triggered before collecting new samples.
        """
        pass

    def _on_step(self) -> bool:
        """
        This method will be called by the model after each call to `env.step()`.

        For child callback (of an `EventCallback`), this will be called
        when the event is triggered.

        :return: If the callback returns False, training is aborted early.
        """
        return True

    def _on_rollout_end(self) -> None:
        """
        This event is triggered before updating the policy.
        """
        # keep only last n checkpoints
        latest_n_steps = _latest_n_checkpoint_steps(self.ckp_path, n=5)
        for file in os.listdir(self.ckp_path):
            if _parse_name(file)["steps"] not in latest_n_steps:
                os.remove(os.path.join(self.ckp_path, file))

    def _on_training_end(self) -> None:
        """
        This event is triggered before exiting the `learn()` method.
        """
        pass
