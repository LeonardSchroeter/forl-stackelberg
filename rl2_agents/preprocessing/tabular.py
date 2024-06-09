"""
Implements preprocessing for tabular MABs and MDPs.
"""

from typing import List

import numpy as np
import torch as tc

from rl2_agents.preprocessing.common import one_hot, Preprocessing

from utils.constants import DEVICE


class MDPPreprocessing(Preprocessing):
    def __init__(self, num_states: int, num_actions: int):
        super().__init__()
        self._num_states = num_states
        self._num_actions = num_actions

    @property
    def output_dim(self):
        return self._num_states + self._num_actions + 2

    def forward(
        self,
        curr_obs: np.ndarray,
        prev_action: tc.LongTensor,
        prev_reward: tc.FloatTensor,
        prev_done: tc.FloatTensor,
    ) -> tc.FloatTensor:
        """
        Creates an input vector for a meta-learning agent.

        Args:
            curr_obs: tc.FloatTensor of shape [B, ..., C, H, W]
            prev_action: tc.LongTensor of shape [B, ...]
            prev_reward: tc.FloatTensor of shape [B, ...]
            prev_done: tc.FloatTensor of shape [B, ...]

        Returns:
            tc.FloatTensor of shape [B, ..., S+A+2]
        """
        curr_obs = tc.LongTensor(curr_obs).to(DEVICE)
        emb_o = one_hot(curr_obs, depth=self._num_states)
        emb_a = one_hot(prev_action, depth=self._num_actions)
        prev_reward = prev_reward.unsqueeze(-1)
        prev_done = prev_done.unsqueeze(-1)
        vec = tc.cat((emb_o, emb_a, prev_reward, prev_done), dim=-1).float()
        return vec


class DGFPreprocessing(Preprocessing):
    def __init__(
        self,
        num_states: List,
        dim_states: List,
        num_actions: int,
        env_height: int,
        leader_cont: bool,
    ):
        super().__init__()
        self._num_states = num_states
        self._dim_states = dim_states
        self._num_actions = num_actions
        self._env_height = env_height
        self._leader_cont = leader_cont

    @property
    def output_dim(self):
        if self._leader_cont:
            return (
                self._dim_states[0]  # follower x, y
                + self._dim_states[1] * self._num_states[2]  # follower local view occupancy
                + self._dim_states[2]  # leader obs
                + self._dim_states[3]  # leader action
                + self._num_actions  # follower previous action
                + 2  # follower previous reward + follower previous done
            )
        else:
            return (
                self._dim_states[0]  # follower x, y
                + self._dim_states[1] * self._num_states[2]  # follower local view occupancy
                + self._dim_states[2]  # leader obs
                + self._dim_states[3] * self._num_states[-1]  # leader action
                + self._num_actions  # follower previous action
                + 2  # follower previous reward + follower previous done
            )

    def forward(
        self,
        curr_obs: np.ndarray,
        prev_action: tc.LongTensor,
        prev_reward: tc.FloatTensor,
        prev_done: tc.FloatTensor,
    ) -> tc.FloatTensor:
        pos = tc.FloatTensor(curr_obs[..., : self._dim_states[0]]).to(DEVICE)
        occps = tc.LongTensor(
            curr_obs[
                ..., self._dim_states[0] : self._dim_states[0] + self._dim_states[1]
            ]
        ).to(DEVICE)

        emb_occps = []
        for k in range(self._dim_states[1]):
            emb_occps.append(
                one_hot(tc.atleast_1d(occps[..., k]), depth=self._num_states[2])
            )

        ol = tc.LongTensor(
            curr_obs[
                ...,
                self._dim_states[0] + self._dim_states[1] : sum(self._dim_states[:3]),
            ]
        ).to(DEVICE)

        al = curr_obs[..., -self._dim_states[-1] :]
        al = (
            tc.FloatTensor(al / self._env_height).to(DEVICE)
            if self._leader_cont
            else one_hot(
                tc.LongTensor(np.atleast_1d(np.squeeze(al))).to(DEVICE),
                depth=self._num_states[-1],
            )
        )

        emb_a = one_hot(prev_action, depth=self._num_actions)
        prev_reward = prev_reward.unsqueeze(-1)
        prev_done = prev_done.unsqueeze(-1)
        vec = tc.cat(
            (pos, *emb_occps, ol, al, emb_a, prev_reward, prev_done), dim=-1
        ).float()

        return vec
