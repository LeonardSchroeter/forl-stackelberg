"""
Implements the Tabular MDP environment(s) from Duan et al., 2016
- 'RL^2 : Fast Reinforcement Learning via Slow Reinforcement Learning'.
"""

from typing import Tuple

import torch
from torch import nn
import numpy as np

from envs.rl2.abstract import MetaEpisodicEnv
from envs.drone_game import DroneGame
from utils.drone_leader_observation import binary_to_decimal, decimal_to_binary
from utils.constants import DEVICE

from stable_baselines3 import PPO
from wrappers.rl2.trial_wrapper import TrialWrapper
from wrappers.rl2.leader import SingleAgentLeaderWrapperMetaRL


class DroneGameFollowerEnv(MetaEpisodicEnv):
    def __init__(self, env: DroneGame):
        self._env = env
        self._state = 0
        # if self.leader_cont:
        #     # config = load_config_args_overwrite("configs/rl2.yml")
        #     # env = build_leader_env_rl2(config)
        #     # self._leader_model, _ = maybe_load_checkpoint_ppo(
        #     #     os.path.join(config.training.checkpoint_path, "leader_cont", "leader"),
        #     #     env,
        #     # )
        #     self._leader_model = nn.Sequential(
        #         nn.Linear(2 * self._env.drone_life_span, 256),
        #         nn.Tanh(),
        #         nn.Linear(256, 2),
        #         # nn.Linear(256, self._env.env.height - 4),
        #         nn.Softmax()
        #     ).to(DEVICE)
        #     self.rand_noise = False

    def inject_rand_noise(self):
        self.rand_noise = True

    def set_follower_policy_net(self, follower_policy_net):
        self._follower_policy_net = follower_policy_net
    
    @property
    def name(self):
        return "drone_game"

    @property
    def max_episode_len(self):
        return self._env.episode_length

    @property
    def num_actions(self):
        return self._env.action_space("follower").n

    @property
    def num_states(self):
        return self._env.observation_space("follower").nvec.tolist()

    @property
    def leader_cont(self):
        return self._env.leader_cont

    @property
    def dim_states(self):
        dim_states = [
            2,
            self._env.env.agent_view_size * self._env.env.agent_view_size,
            2 * self._env.drone_life_span
            if self.leader_cont
            else self._env.env.num_divisions,
            1,
        ]
        # assert sum(dim_states) == len(self._env.observation_space("follower").nvec)
        return dim_states

    def _new_leader_policy(self):
        if self._env.leader_cont:
            leader_env = TrialWrapper(self._env, num_episodes=3)
            leader_env = SingleAgentLeaderWrapperMetaRL(
                leader_env, follower_policy_net=self._follower_policy_net
            )

            self._leader_model = PPO(
            "MlpPolicy",
                leader_env,
                verbose=1,
            )
            # for _, param in self._leader_model.named_parameters():
            #     param.data = torch.FloatTensor(
            #         0.1 * np.random.uniform(-1, 1, param.size())
            #     ).to(DEVICE)
        else:
            self._leader_response = [
                self._env.action_space("leader").sample()
                for _ in range(2 ** self._env.observation_space("leader").n)
            ]
        # self._leader_response = [
        #     3 \
        #     for _ in range(2 ** self._env.observation_space("leader").n)
        # ]
        # self._leader_response = [0, 3, 3, 0, 3, 0, 3, 0, 0, 3, 0, 3, 0, 3, 0, 3]

    def new_env(self) -> None:
        """
        Sample a new MDP from the distribution over MDPs.

        Returns:
            None
        """
        self._new_leader_policy()
        self._state = 0

    def reset(self) -> int:
        """
        Reset the environment.

        Returns:
            initial state.
        """
        self._state = self._env.reset()["follower"]
        return self._state

    def step(self, action, auto_reset=True) -> Tuple[int, float, bool, dict]:
        """
        Take action in the MDP, and observe next state, reward, done, etc.

        Args:
            action: action corresponding to an arm index.
            auto_reset: auto reset. if true, new_state will be from self.reset()

        Returns:
            new_state, reward, done, info.
        """

        ol = self._env.get_leader_observation()

        if self._env.leader_cont:
            # al_probs = self._leader_model(torch.FloatTensor(ol).to(DEVICE))
            # al = Categorical(al_probs).sample((1,)).cpu().item()
            # al = (
            #     0.1 * self._leader_model(torch.FloatTensor(ol).to(DEVICE)).item()
            #     # + 0.5 + 0.2 * np.random.normal()
            #     + 0.5
            # )
            # al = self._leader_model(torch.FloatTensor(ol).to(DEVICE)).cpu().numpy()[0]
            al = self._leader_model.predict(ol, deterministic=False)[0]
            # print(al)
            # if self.rand_noise:
            #     al += 0.1 * np.random.normal()
            # al = np.clip(al, 0, 1, dtype=float)
            # print(al)
        else:
            ol = binary_to_decimal(ol)
            al = self._leader_response[ol]
        a_ts = {"leader": al, "follower": action}

        s_tp1s, r_ts, done_ts, _, _ = self._env.step(a_ts)
        s_tp1 = s_tp1s["follower"]
        self._state = s_tp1

        r_t = r_ts["follower"]

        done_t = done_ts["follower"]
        if done_t and auto_reset:
            s_tp1 = self.reset()

        return s_tp1, r_t, done_t, {}


class DroneGameFollowerInfoSample(DroneGameFollowerEnv):
    def __init__(self, env: DroneGame):
        super().__init__(env)

    def set_leader_model(self, leader_model):
        self._leader_model = leader_model

    def _preprocess_observation(self, obs):
        return decimal_to_binary(obs, width=self._env.observation_space("leader").n)

    def _new_leader_policy(self):
        self._leader_response = [
            self._leader_model.predict(
                self._preprocess_observation(o), deterministic=False
            )[0]
            for o in range(2 ** self._env.observation_space("leader").n)
        ]
