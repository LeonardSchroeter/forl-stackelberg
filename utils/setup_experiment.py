import os

from envs.rl2.mat_game_follower_env import (
    MatGameFollowerEnv,
    IteratedMatrixGame,
)
from envs.rl2.drone_game_follower_env import (
    DroneGameFollowerEnv,
    DroneGameFollowerInfoSample,
    DroneGame,
)
from envs.drone_game import DroneGameEnv

from rl2_agents.preprocessing.tabular import (
    MDPPreprocessing,
    DGFPreprocessing,
)
from rl2_agents.architectures.gru import GRU
from rl2_agents.architectures.lstm import LSTM
from rl2_agents.architectures.snail import SNAIL
from rl2_agents.architectures.transformer import Transformer
from rl2_agents.heads.policy_heads import LinearPolicyHead
from rl2_agents.heads.value_heads import LinearValueHead
from rl2_agents.integration.policy_net import StatefulPolicyNet
from rl2_agents.integration.value_net import StatefulValueNet

from utils.constants import DEVICE
from utils.checkpoint_util import maybe_load_checkpoint_rl2


def create_env(config):
    if config.env.name == "matrix_game":
        return MatGameFollowerEnv(
            env=IteratedMatrixGame(
                matrix="prisoners_dilemma",
                episode_length=config.matrix_game.episode_len,
                memory=config.matrix_game.memory,
            )
        )
    if config.env.name == "drone_game":
        env = DroneGame(
            env=DroneGameEnv(
                width=config.drone_game.width,
                height=config.drone_game.height,
                drone_dist=config.drone_game.drone_dist,
            ),
            headless=config.drone_game.headless,
            leader_cont=config.drone_game.leader_cont,
        )
        env = (
            DroneGameFollowerInfoSample(env)
            if config.inner_outer
            else DroneGameFollowerEnv(env)
        )
        if config.drone_game.leader_cont:
            env.inject_rand_noise()
        return env

    raise NotImplementedError


def create_preprocessing(env):
    if env.name == "matrix_game":
        return MDPPreprocessing(num_states=env.num_states, num_actions=env.num_actions)
    if env.name == "drone_game":
        return DGFPreprocessing(
            num_states=env.num_states,
            dim_states=env.dim_states,
            num_actions=env.num_actions,
            env_height=env._env.env.height,
            leader_cont=env.leader_cont,
        )
    raise NotImplementedError


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


def create_net(
    net_type,
    env,
    architecture,
    num_features,
    context_size,
):
    preprocessing = create_preprocessing(
        env=env,
    ).to(DEVICE)
    architecture = create_architecture(
        architecture=architecture,
        input_dim=preprocessing.output_dim,
        num_features=num_features,
        context_size=context_size,
    ).to(DEVICE)
    head = create_head(
        head_type=net_type,
        num_features=architecture.output_dim,
        num_actions=env.num_actions,
    ).to(DEVICE)

    if net_type == "policy":
        return StatefulPolicyNet(
            preprocessing=preprocessing, architecture=architecture, policy_head=head
        )
    if net_type == "value":
        return StatefulValueNet(
            preprocessing=preprocessing, architecture=architecture, value_head=head
        )
    raise NotImplementedError


def get_policy_net_for_inference(env, config):
    # create learning system.
    policy_net = create_net(
        net_type="policy",
        env=env,
        architecture=config.model.architecture,
        num_features=config.model.num_features,
        context_size=0,
    )

    policy_net = policy_net.to(DEVICE)

    # load checkpoint, if applicable.
    if config.inner_outer:
        folder = "inner_outer"
    elif (config.drone_game.leader_cont) and (config.env.name == "drone_game"):
        folder = "leader_cont"
    else:
        folder = ""
    model_name = os.path.join(folder, "follower", "policy_net")
    maybe_load_checkpoint_rl2(
        checkpoint_dir=config.training.checkpoint_path,
        model_name=model_name,
        model=policy_net,
        optimizer=None,
        scheduler=None,
        steps=None,
    )

    return policy_net
