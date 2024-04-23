import os

import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical
from torch.optim import AdamW, Optimizer
import numpy as np

import wandb

from envs.matrix_game import IteratedMatrixGame
from wrappers.follower import FollowerWrapper
from pretrain import create_model, get_policy, get_action, INIT_LEADER_OBS

def train_one_epoch_ppo(
        env: FollowerWrapper,
        actor_leader: nn.Module,
        critic_leader: nn.Module,
        actor_follower: nn.Module,
        actor_optimizer: Optimizer,
        critic_optimizer: Optimizer,
        max_timesteps=5000,
        episode_timesteps=200) -> float:
    
    epoch_total_timesteps = 0
    
    epoch_observations = []
    epoch_actions = []
    epoch_log_probability_actions = []
    epoch_returns: list[float] = []

    while True:

        if epoch_total_timesteps > max_timesteps:
            break
        
        episode_rewards = []

        query_answer = torch.argmax(actor_leader(INIT_LEADER_OBS),dim=-1)
        env.set_leader_response(query_answer)

        epoch_observations += INIT_LEADER_OBS.squeeze().tolist()
        epoch_actions += query_answer.squeeze().tolist()
        # leader_policies = get_policy(actor_leader, INIT_LEADER_OBS)
        leader_logits = actor_leader(INIT_LEADER_OBS)
        for i in range(INIT_LEADER_OBS.shape[0]):
            leader_policy = Categorical(logits=leader_logits[i])
            epoch_log_probability_actions.append(leader_policy.log_prob(epoch_actions[i]).item())
        epoch_returns += np.zeros(5,).tolist()

        observation = env.reset()

        for timestep in range(episode_timesteps):
            epoch_total_timesteps += 1

            action = {}
            leader_policy = get_policy(actor_leader, torch.atleast_1d(torch.as_tensor(observation["leader"],dtype=torch.float)))
            action["leader"], log_probability_action = get_action(leader_policy)
            follower_policy = get_policy(actor_follower, observation["follower"])
            action["follower"], _ = get_action(follower_policy)
            observation, reward, done, _, _ = env.step(action)

            epoch_actions.append(action["leader"])
            epoch_observations.append(observation["leader"])
            episode_rewards.append(reward["leader"])
            epoch_log_probability_actions.append(log_probability_action)

            if done["leader"] is True:

                break

        epoch_returns += compute_return(episode_rewards, 0.95)

    epoch_observations = torch.unsqueeze(torch.as_tensor(epoch_observations, dtype=torch.float),dim=-1)
    epoch_actions = torch.as_tensor(epoch_actions, dtype=torch.float)
    epoch_returns = torch.as_tensor(epoch_returns, dtype=torch.float)
    epoch_log_probability_actions = torch.as_tensor(epoch_log_probability_actions, dtype=torch.float)

    for _ in range(5):

        V = critic_leader(epoch_observations)
        
        adv = epoch_returns - V.detach().squeeze()
        adv = (adv - adv.mean()) / (adv.std() + 1e-10)

        curr_policy = get_policy(model=actor_leader, observation=epoch_observations)
        curr_log_probability_actions = curr_policy.log_prob(epoch_actions)
        ratio = torch.exp(curr_log_probability_actions - epoch_log_probability_actions)

        surr1 = ratio * adv
        surr2 = torch.clamp(ratio, 0.8, 1.2) * adv

        surr_min, _ = torch.min(torch.stack((surr1, surr2)),dim=0)
        actor_loss = -surr_min.mean()
        wandb.log({"actor_loss": actor_loss})

        actor_loss.backward(retain_graph=True)
        actor_optimizer.step()
        actor_optimizer.zero_grad()

        # for _ in range(100):

        critic_loss = nn.MSELoss()(V, epoch_returns)
        wandb.log({"critic loss": critic_loss})
        critic_loss.backward()
        critic_optimizer.step()
        critic_optimizer.zero_grad()
        # V = critic_leader(epoch_observations)

    return float(np.mean(epoch_returns.numpy()))

def compute_return(episode_rewards, gamma):
    
    discounted_reward = 0
    episode_returns = []
    
    for rew in reversed(episode_rewards):
        discounted_reward = rew + gamma * discounted_reward
        episode_returns.insert(0, discounted_reward)

    return episode_returns

def train(epochs=40) -> None:

    env = IteratedMatrixGame(matrix="prisoners_dilemma", episode_length=10, memory=2)
    env = FollowerWrapper(env=env, num_queries=5)

    number_observation_features_follower = env.num_queries + 1
    number_actions_follower = env.action_space("follower").n
    actor_follower = create_model(number_observation_features_follower, number_actions_follower)

    actor_follower.load_state_dict(torch.load(os.path.join(os.path.abspath(os.path.dirname(__file__)), "checkpoints", "pretrained_0.pth")))

    number_observation_features = 1
    number_actions = env.action_space("leader").n
    actor_leader = create_model(number_observation_features, number_actions)
    critic_leader = create_model(number_observation_features, 1)

    actor_optimizer = AdamW(actor_leader.parameters(), 1e-2)
    critic_optimizer = AdamW(critic_leader.parameters(), 1e-1)

    for epoch in range(epochs):
        average_return = train_one_epoch_ppo(env, actor_leader, critic_leader, actor_follower, actor_optimizer, critic_optimizer, max_timesteps=200)
        print('epoch: %3d \t return: %.3f' % (epoch, average_return))
        wandb.log({"epoch": epoch, "return": average_return})

if __name__ == "__main__":
    train(epochs=100)