"""
Script for training stateful meta-reinforcement learning agents
"""

import os

from functools import partial

import torch as tc

from rl2_agents.architectures.gru import GRU
from rl2_agents.architectures.lstm import LSTM
from rl2_agents.architectures.snail import SNAIL
from rl2_agents.architectures.transformer import Transformer
from rl2_agents.heads.policy_heads import LinearPolicyHead
from rl2_agents.heads.value_heads import LinearValueHead
from algos.ppo import training_loop

from utils.config_util import load_config_args_overwrite
from utils.checkpoint_util import (
    maybe_load_checkpoint_rl2,
    save_checkpoint_rl2,
    maybe_load_checkpoint_ppo,
)
from utils.comm_util import get_comm, sync_state
from utils.constants import ROOT_RANK, DEVICE
from utils.optim_util import get_weight_decay_param_groups
from utils.setup_experiment import create_env, create_net

from wrappers.rl2.trial_wrapper import TrialWrapper
from wrappers.rl2.leader import SingleAgentLeaderWrapperMetaRL


def create_architecture(architecture, input_dim, num_features, context_size):
    if architecture == "gru":
        return GRU(
            input_dim=input_dim,
            hidden_dim=num_features,
            forget_bias=1.0,
            use_ln=True,
            reset_after=True,
        )
    if architecture == "lstm":
        return LSTM(
            input_dim=input_dim, hidden_dim=num_features, forget_bias=1.0, use_ln=True
        )
    if architecture == "snail":
        return SNAIL(
            input_dim=input_dim,
            feature_dim=num_features,
            context_size=context_size,
            use_ln=True,
        )
    if architecture == "transformer":
        return Transformer(
            input_dim=input_dim,
            feature_dim=num_features,
            n_layer=9,
            n_head=2,
            n_context=context_size,
        )
    raise NotImplementedError


def create_head(head_type, num_features, num_actions):
    if head_type == "policy":
        return LinearPolicyHead(num_features=num_features, num_actions=num_actions)
    if head_type == "value":
        return LinearValueHead(num_features=num_features)
    raise NotImplementedError


def main():
    print("Using device:", DEVICE)

    config = load_config_args_overwrite("configs/rl2.yml")

    comm = get_comm()

    # create env.
    env = create_env(config=config)

    # create learning system.
    policy_net = create_net(
        net_type="policy",
        env=env,
        architecture=config.model.architecture,
        num_features=config.model.num_features,
        context_size=0,
    )

    value_net = create_net(
        net_type="value",
        env=env,
        architecture=config.model.architecture,
        num_features=config.model.num_features,
        context_size=0,
    )

    policy_net = policy_net.to(DEVICE)
    value_net = value_net.to(DEVICE)

    policy_optimizer = tc.optim.AdamW(
        get_weight_decay_param_groups(policy_net, config.training.adam_wd),
        lr=config.training.adam_lr,
        eps=config.training.adam_eps,
    )
    value_optimizer = tc.optim.AdamW(
        get_weight_decay_param_groups(value_net, config.training.adam_wd),
        lr=config.training.adam_lr,
        eps=config.training.adam_eps,
    )

    policy_scheduler = None
    value_scheduler = None

    model_name = "follower"
    if config.inner_outer:
        model_name = os.path.join("inner_outer", model_name)
    elif config.env.name == "drone_game" and config.drone_game.leader_cont:
        model_name = os.path.join("leader_cont", model_name)
    policy_model_name = os.path.join(model_name, "policy_net")
    value_model_name = os.path.join(model_name, "value_net")

    # load checkpoint, if applicable.
    pol_iters_so_far = 0
    if comm.Get_rank() == ROOT_RANK:
        a = maybe_load_checkpoint_rl2(
            checkpoint_dir=config.training.checkpoint_path,
            model_name=policy_model_name,
            model=policy_net,
            optimizer=policy_optimizer,
            scheduler=policy_scheduler,
            steps=None,
        )

        b = maybe_load_checkpoint_rl2(
            checkpoint_dir=config.training.checkpoint_path,
            model_name=value_model_name,
            model=value_net,
            optimizer=value_optimizer,
            scheduler=value_scheduler,
            steps=None,
        )

        if a != b:
            raise RuntimeError(
                "Policy and value iterates not aligned in latest checkpoint!"
            )
        pol_iters_so_far = a

    # sync state.
    pol_iters_so_far = comm.bcast(pol_iters_so_far, root=ROOT_RANK)
    sync_state(
        model=policy_net,
        optimizer=policy_optimizer,
        scheduler=policy_scheduler,
        comm=comm,
        root=ROOT_RANK,
    )
    sync_state(
        model=value_net,
        optimizer=value_optimizer,
        scheduler=value_scheduler,
        comm=comm,
        root=ROOT_RANK,
    )



    # make callback functions for checkpointing.
    policy_checkpoint_fn = partial(
        save_checkpoint_rl2,
        checkpoint_dir=config.training.checkpoint_path,
        model_name=policy_model_name,
        model=policy_net,
        optimizer=policy_optimizer,
        scheduler=policy_scheduler,
    )

    value_checkpoint_fn = partial(
        save_checkpoint_rl2,
        checkpoint_dir=config.training.checkpoint_path,
        model_name=value_model_name,
        model=value_net,
        optimizer=value_optimizer,
        scheduler=value_scheduler,
    )

    if config.inner_outer:
        leader_env = TrialWrapper(env._env, num_episodes=3)
        leader_env = SingleAgentLeaderWrapperMetaRL(
            leader_env, follower_policy_net=policy_net
        )

        training_config = {"n_steps": 128}
        leader_model, leader_callback_list = maybe_load_checkpoint_ppo(
            os.path.join(config.training.checkpoint_path, "inner_outer", "leader"),
            leader_env,
            training_config=training_config,
            save_freq=10,
        )
        env.set_leader_model(leader_model)
    else:
        leader_model = None
        leader_callback_list = None,

    training_loop(
        env=env,
        policy_net=policy_net,
        value_net=value_net,
        policy_optimizer=policy_optimizer,
        value_optimizer=value_optimizer,
        policy_scheduler=policy_scheduler,
        value_scheduler=value_scheduler,
        meta_episodes_per_policy_update=config.training.meta_episodes_per_policy_update,
        meta_episodes_per_learner_batch=config.training.meta_episodes_per_learner_batch,
        num_meta_episodes=config.env.num_meta_episodes,
        ppo_opt_epochs=config.training.ppo_opt_epochs,
        ppo_clip_param=config.training.ppo_clip_param,
        ppo_ent_coef=config.training.ppo_ent_coef,
        discount_gamma=config.training.discount_gamma,
        gae_lambda=config.training.gae_lambda,
        standardize_advs=bool(config.training.standardize_advs),
        max_pol_iters=config.training.max_pol_iters,
        pol_iters_so_far=pol_iters_so_far,
        policy_checkpoint_fn=policy_checkpoint_fn,
        value_checkpoint_fn=value_checkpoint_fn,
        comm=comm,
        log_wandb=config.training.log_wandb,
        inner_outer=config.inner_outer,
        leader_callback_list=leader_callback_list,
        leader_model=leader_model,
    )


if __name__ == "__main__":
    main()
