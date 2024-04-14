import torch
import torch.nn.functional as F
import numpy as np
import copy

from typing import List, Dict

import wandb

from model import Actor, Critic
from wrappers.follower import FollowerWrapper 

AGENTS = ["leader", "follower"]
INIT_LEADER_OBS = torch.tensor([[1],[2],[3],[4],[5]], dtype=torch.float)

class PPOLeaderFollower:

    def __init__(self, env: FollowerWrapper):
        self.env = env
        
        leader_obs_dim = 1
        leader_act_dim = env.action_space("leader").n
        follower_obs_dim = self.env.num_queries + 1
        follower_act_dim = env.action_space("follower").n

        self.leader_actor = Actor(leader_obs_dim, leader_act_dim)
        self.leader_critic = Critic(leader_obs_dim, 1) 
        self.follower_actor = Actor(follower_obs_dim, follower_act_dim)
        self.follower_critic = Critic(follower_obs_dim, 1)
        self._init_hyperparameters()
        run = wandb.init(project="stackelberg",
                    config={"pretraining_actor_lr": self.training_actor_lr, 
                    "pretraining_critic_lr": self.training_critic_lr},
                    )

    def _init_hyperparameters(self):
        self.n_eps_per_batch = 3
        self.eps_length = 20
        self.n_iter_per_update = 5
        self.gamma = 0.95
        self.clip = 0.2
        self.pretraining_actor_lr = 0.1
        self.pretraining_critic_lr = 0.1
        self.training_actor_lr = 0.1
        self.training_critic_lr = 0.1

    def sample_leader_policy(self):
        actor_leader = copy.deepcopy(self.leader_actor)
        [param.data.fill_(np.random.randn()) for name, param in actor_leader.named_parameters()]
        return actor_leader
    
    def get_actions(self, actor: Actor, obs):
        logits = actor(obs).detach().cpu()
        action = np.argmax(logits)
        log_prob = np.log(logits[action])

        return action, log_prob
    
    def compute_returns(self, rewards: List):
        returns = []
        discounted_rewards = 0
        for reward in reversed(rewards):
            discounted_rewards += reward + self.gamma * discounted_rewards
            returns.insert(0, discounted_rewards)

        return returns
    
    def evaluate(self, actor: Actor, critic: Critic, batch_obs: torch.Tensor, batch_act: torch.Tensor):
        V = critic(batch_obs).squeeze()
        logits = actor(batch_obs)
        log_probs = []
        for k in range(self.n_eps_per_batch * self.eps_length):
            log_probs.append(logits[k][int(batch_act[k].item())])
        log_probs = torch.stack(log_probs)

        return V, log_probs
    
    # def follower_observation(self, obs_dict: Dict):
    #     obs = obs_dict["follower"]["queries"].tolist()
    #     obs.insert(0, obs_dict["follower"]["original"])
    #     return torch.tensor(obs, dtype=torch.float)
    
    def pretraining(self, iterations):

        actor_optim = torch.optim.AdamW(self.follower_actor.parameters(),lr=self.pretraining_actor_lr)
        critic_optim = torch.optim.AdamW(self.follower_critic.parameters(),lr=self.pretraining_critic_lr)

        for iter in range(iterations):
            batch_obs = {"leader": [], "follower": []}
            batch_act = {"leader": [], "follower": []}
            batch_log_probs = {"leader": [], "follower": []}
            batch_returns = {"leader": [], "follower": []}
            for eps in range(self.n_eps_per_batch):

                random_leader_policy = self.sample_leader_policy()
                random_leader_responses = torch.argmax(random_leader_policy(INIT_LEADER_OBS),dim=-1).to(torch.float)
                self.env.set_leader_response(random_leader_responses)
                    
                rewards_per_eps = {"leader": [], "follower": []}
                obs = self.env.reset()

                for k in range(self.eps_length):
                    actions, log_probs = {}, {}
                    actions["leader"], log_probs["leader"] = self.get_actions(actor=self.leader_actor, obs=obs["leader"])
                    actions["follower"], log_probs["follower"] = self.get_actions(actor=self.follower_actor, obs=obs["follower"])
                    obs, rewards, terminated, truncated, infos = self.env.step(actions=actions)
                    
                    for agent in AGENTS:
                        batch_obs[agent].append(obs[agent])
                        batch_act[agent].append(actions[agent])
                        batch_log_probs[agent].append(log_probs[agent])
                        rewards_per_eps[agent].append(rewards[agent])

                    # why terminated here at 1st step?
                    # if terminated:
                    #     break

                del random_leader_policy
                
                for agent in AGENTS:
                    batch_returns[agent] += self.compute_returns(rewards_per_eps[agent])

            for agent in AGENTS:
                batch_obs[agent] = torch.tensor(batch_obs[agent], dtype=torch.float)
                batch_act[agent] = torch.tensor(batch_act[agent], dtype=torch.float)
                batch_log_probs[agent] = torch.tensor(batch_log_probs[agent], dtype=torch.float)
                batch_returns[agent] = torch.tensor(batch_returns[agent], dtype=torch.float)

            for j in range(self.n_iter_per_update):
            
                V, curr_log_probs = self.evaluate(actor=self.follower_actor, critic=self.follower_critic,
                                                           batch_obs=batch_obs["follower"], batch_act=batch_act["follower"])
                adv = batch_returns["follower"] - V
                
                adv = (adv - adv.mean()) / (adv.std() + 1e-10)

                ratios = torch.exp(curr_log_probs - batch_log_probs["follower"])

                surr1 = ratios * adv
                surr2 = torch.clamp(ratios, 1 - self.clip, 1 + self.clip) * adv

                actor_loss = (-torch.min(surr1, surr2)).mean()
                critic_loss = F.mse_loss(batch_returns["follower"], V)
            
                actor_optim.zero_grad()
                actor_loss.backward(retain_graph=True)
                actor_optim.step()

                critic_optim.zero_grad()
                critic_loss.backward()
                critic_optim.step()

                wandb.log({"actor loss": actor_loss, "critic_loss": critic_loss})