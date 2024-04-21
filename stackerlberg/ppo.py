import torch
import torch.nn.functional as F
import torch.distributions as td
import numpy as np
import copy

from typing import List, Dict

import wandb

from model import Actor, Critic
from wrappers.follower import FollowerWrapper 

AGENTS = ["leader", "follower"]
INIT_LEADER_OBS = torch.tensor([[0],[1],[2],[3],[4]], dtype=torch.float)

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
        self.test_fixed_policy = lambda obs : torch.tensor([1 for s in obs],dtype=torch.int).squeeze()

    def _init_hyperparameters(self):
        self.n_eps_per_batch = 3
        self.n_iter_per_update = 5
        self.n_critic_update_per_actor_update = 100
        self.gamma = 0.95
        self.clip = 0.2
        self.pretraining_actor_lr = 0.1
        self.pretraining_critic_lr = 5e-3
        self.training_actor_lr = 5e-3
        self.training_critic_lr = 5e-3

    def sample_leader_policy(self):
        actor_leader = copy.deepcopy(self.leader_actor)
        [param.data.fill_(np.random.randn()) for name, param in actor_leader.named_parameters()]
        return actor_leader
    
    def get_actions(self, actor: Actor, obs):
        logits = actor(obs)
        dist = td.Categorical(logits=logits)
        action = dist.sample() 
        log_prob = dist.log_prob(action)

        return action, log_prob
    
    def compute_returns(self, rewards: List):
        returns = []
        discounted_rewards = 0
        for reward in reversed(rewards):
            discounted_rewards = reward + self.gamma * discounted_rewards
            returns.insert(0, discounted_rewards)

        return returns
    
    def evaluate(self, actor: Actor, critic: Critic, batch_obs: torch.Tensor, batch_act: torch.Tensor):
        V = critic(batch_obs).squeeze()
        logits = actor(batch_obs)
        dist = td.Categorical(logits=logits)
        log_probs = []
        for k in range(batch_obs.shape[0]):
            log_probs.append(dist.log_prob(batch_act[k]))
        log_probs = torch.stack(log_probs)

        return V, log_probs
    
    def pretraining(self, iterations):

        actor_optim = torch.optim.AdamW(self.follower_actor.parameters(),lr=self.pretraining_actor_lr)  

        for iter in range(iterations):
            print(iter)
            batch_obs = {"leader": [], "follower": []}
            batch_act = {"leader": [], "follower": []}
            batch_log_probs = {"leader": [], "follower": []}
            batch_returns = {"leader": [], "follower": []}
            for eps in range(self.n_eps_per_batch):

                # random_leader_policy = self.sample_leader_policy()
                # random_leader_responses = torch.argmax(random_leader_policy(INIT_LEADER_OBS),dim=-1).to(torch.float)
                random_leader_responses = self.test_fixed_policy(obs=INIT_LEADER_OBS)
                self.env.set_leader_response(random_leader_responses)
                    
                rewards_per_eps = {"leader": [], "follower": []}
                obs = self.env.reset()

                while True:
                    actions, log_probs = {}, {}
                    actions["leader"], log_probs["leader"] = self.test_fixed_policy(obs=torch.tensor([obs["leader"]])), -1
                    actions["follower"], log_probs["follower"] = self.get_actions(actor=self.follower_actor, obs=obs["follower"])
                    obs, rewards, terminated, _, _ = self.env.step(actions=actions)
                    print(actions, rewards)
                    
                    for agent in AGENTS:
                        batch_obs[agent].append(obs[agent])
                        batch_act[agent].append(actions[agent])
                        batch_log_probs[agent].append(log_probs[agent])
                        rewards_per_eps[agent].append(rewards[agent])

                    if terminated["leader"] or terminated["follower"]:
                        break

                # del random_leader_policy
                
                for agent in AGENTS:
                    # batch_returns[agent] += self.compute_returns(rewards_per_eps[agent])
                    return_per_episode = self.compute_returns(rewards_per_eps[agent])[0]
                    batch_returns[agent] += [return_per_episode for _ in range(self.env.episode_length)]

            mean_return_sum_per_episode = {}
            for agent in AGENTS:
                batch_obs[agent] = torch.tensor(batch_obs[agent], dtype=torch.float)
                batch_act[agent] = torch.tensor(batch_act[agent], dtype=torch.float)
                batch_log_probs[agent] = torch.tensor(batch_log_probs[agent], dtype=torch.float)
                batch_returns[agent] = torch.tensor(batch_returns[agent], dtype=torch.float, requires_grad=True)
                mean_return_sum_per_episode[agent] = torch.sum(batch_returns[agent]) / self.n_eps_per_batch
            
            wandb.log({"follower return" : mean_return_sum_per_episode["follower"]})

            # for _ in range(self.n_iter_per_update):
                
            actor_loss = -(batch_log_probs["follower"] * batch_returns["follower"]).mean()
            # batch_returns["follower"] = (batch_returns["follower"] - batch_returns["follower"].mean()) / (batch_returns["follower"].std() + 1e-10)

            # actor_loss = -batch_returns["follower"].mean()
            actor_optim.zero_grad()
            actor_loss.backward(retain_graph=True)
            actor_optim.step()

            wandb.log({"actor loss": actor_loss})