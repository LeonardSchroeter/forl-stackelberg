"""
Implements the Tabular MDP environment(s) from Duan et al., 2016
- 'RL^2 : Fast Reinforcement Learning via Slow Reinforcement Learning'.
"""

from typing import Tuple

from envs.rl2.abstract import MetaEpisodicEnv
from envs.drone_game import DroneGame
from utils.drone_game_obs import binary_to_decimal

class DroneGameFollowerEnv(MetaEpisodicEnv):
    def __init__(self, env: DroneGame):
        
        self._env = env
        self.new_env()
        self._state = 0

    @property
    def name(self):
        return "drone_game_follower"

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
    def dim_states(self):
        dim_states = [
            2,
            self._env.env.agent_view_size * self._env.env.agent_view_size,
            self._env.env.num_divisions,
            1,
        ]
        assert sum(dim_states) == len(self._env.observation_space("follower").nvec)
        return dim_states

    def _new_leader_policy(self):
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

        ol = binary_to_decimal(self._env.get_leader_observation())

        a_ts = {"leader": self._leader_response[ol], "follower": action}

        s_tp1s, r_ts, done_ts, _, _ = self._env.step(a_ts)
        s_tp1 = s_tp1s["follower"]
        self._state = s_tp1

        r_t = r_ts["follower"]

        done_t = done_ts["follower"]
        if done_t and auto_reset:
            s_tp1 = self.reset()

        return s_tp1, r_t, done_t, {}
