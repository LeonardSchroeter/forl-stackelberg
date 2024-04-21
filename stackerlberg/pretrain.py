import os

import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical
from torch.optim import Adam, Optimizer
import numpy as np
# import gym  # type: ignore
from envs.matrix_game import IteratedMatrixGame
from wrappers.follower import FollowerWrapper
import wandb
run = wandb.init(project="stackelberg 1")

AGENTS = ["leader", "follower"]
INIT_LEADER_OBS = torch.tensor([[0],[1],[2],[3],[4]], dtype=torch.float)
ALL_POSSIBLE_RESPONSE = torch.as_tensor([[0,0,0,0,0],
                                         [1,0,0,0,0],
                                         [0,1,0,0,0],
                                         [0,0,1,0,0],
                                         [0,0,0,1,0],
                                         [0,0,0,0,1],
                                         [1,1,0,0,0],
                                         [1,0,1,0,0],
                                         [1,0,0,1,0],
                                         [1,0,0,0,1],
                                         [0,1,1,0,0],
                                         [0,1,0,1,0],
                                         [0,1,0,0,1],
                                         [0,0,1,1,0],
                                         [0,0,1,0,1],
                                         [0,0,0,1,1],
                                         [0,0,1,1,1],
                                         [0,1,0,1,1],
                                         [0,1,1,0,1],
                                         [0,1,1,1,0],
                                         [1,0,0,1,1],
                                         [1,0,1,0,1],
                                         [1,0,1,1,0],
                                         [1,1,0,0,1],
                                         [1,1,0,1,0],
                                         [1,1,1,0,0],
                                         [0,1,1,1,1],
                                         [1,0,1,1,1],
                                         [1,1,0,1,1],
                                         [1,1,1,0,1],
                                         [1,1,1,1,0],
                                         [1,1,1,1,1]], dtype=torch.float)

def create_model(number_observation_features: int, number_actions: int) -> nn.Module:

    hidden_layer_features = 32

    return nn.Sequential(
        nn.Linear(in_features=number_observation_features,
                  out_features=hidden_layer_features),
        nn.ReLU(),
        nn.Linear(in_features=hidden_layer_features,
                  out_features=number_actions),
    )


def get_policy(model: nn.Module, observation: np.ndarray) -> Categorical:

    observation_tensor = torch.as_tensor(observation, dtype=torch.float32)
    logits = model(observation_tensor)

    return Categorical(logits=logits)


def get_action(policy: Categorical) -> tuple[int, torch.Tensor]:

    action = policy.sample()  # Unit tensor

    action_int = int(action.item())

    log_probability_action = policy.log_prob(action)

    return action_int, log_probability_action


def calculate_loss(epoch_log_probability_actions: torch.Tensor, epoch_action_rewards: torch.Tensor) -> torch.Tensor:

    return -(epoch_log_probability_actions * epoch_action_rewards).mean()


def pretrain_one_epoch_pg(
        env: FollowerWrapper,
        model: nn.Module,
        optimizer: Optimizer,
        max_timesteps=5000) -> float:
    
    epoch_total_timesteps = 0

    epoch_returns: list[float] = []

    epoch_log_probability_actions = []
    epoch_action_rewards = []

    while True:

        if epoch_total_timesteps > max_timesteps:
            break

        episode_reward: float = 0

        random_leader_responses = torch.as_tensor([env.action_space("leader").sample() for s in INIT_LEADER_OBS], dtype=torch.float)
        # random_leader_responses = torch.as_tensor([0, 0, 0, 1, 1], dtype=torch.float) # tit-for-tat
        print(f"epoch total timesteps: {epoch_total_timesteps}, query answer: {random_leader_responses}")
        env.set_leader_response(random_leader_responses)

        observation = env.reset()

        while True:

            epoch_total_timesteps += 1

            policy = get_policy(model, observation["follower"])
            action = {}
            action["follower"], log_probability_action = get_action(policy)
            # wandb.log({"Follower action" : action["follower"]})
            action["leader"] = int(random_leader_responses[observation["leader"]].item())
            observation, reward, done, _, _ = env.step(action)

            episode_reward += reward["follower"]

            epoch_log_probability_actions.append(log_probability_action)

            if done["follower"] is True:
                for _ in range(env.env.episode_length):
                    epoch_action_rewards.append(episode_reward)

                break

        epoch_returns.append(episode_reward)

    epoch_loss = calculate_loss(torch.stack(
        epoch_log_probability_actions),
        torch.as_tensor(
        epoch_action_rewards, dtype=torch.float32)
    )

    epoch_loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    return float(np.mean(epoch_returns))

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

        query_answer = actor_leader(INIT_LEADER_OBS)
        env.set_leader_response(query_answer)

        epoch_observations += INIT_LEADER_OBS.squeeze().tolist()
        epoch_actions += query_answer.squeeze().tolist()
        

        observation = env.reset()

        for timestep in range(episode_timesteps):
            epoch_total_timesteps += 1

            policy = get_policy(actor_leader, observation["leader"])
            action = {}
            action["leader"], log_probability_action = get_action(policy)
            wandb.log({"Follower action" : action["follower"]})
            action["follower"] = actor_follower(obs=torch.tensor([observation["leader"]]))
            observation, reward, done, _, _ = env.step(action)

            epoch_actions.append(action["leader"])
            epoch_observations.append(observation["leader"])
            episode_rewards.append(reward["leader"])

            epoch_log_probability_actions.append(log_probability_action)

            if done["leader"] is True:

                break

        epoch_returns += compute_return(episode_rewards, 0.95)

    epoch_returns = torch.as_tensor(epoch_returns, dtype=torch.float)
    epoch_log_probability_actions = torch.as_tensor(epoch_log_probability_actions, dtype=torch.float)

    V = critic_leader(epoch_observations)
    
    adv = epoch_returns - V.detach()
    adv = (adv - adv.mean()) / (adv.std() + 1e-10)

    curr_policy = get_policy(model=actor_leader, observation=epoch_observations)
    curr_log_probability_actions = curr_policy.log_prob(epoch_actions)
    ratio = torch.exp(curr_log_probability_actions - epoch_log_probability_actions)

    surr1 = ratio * adv
    surr2 = torch.clamp(ratio, 0.8, 1.2) * adv

    actor_loss = -torch.min((surr1, surr2)).mean()

    actor_loss.backward()
    actor_optimizer.step()
    actor_optimizer.zero_grad()

    for _ in range(100):

        critic_loss = nn.MSELoss()(V, epoch_returns)
        critic_loss.backward()
        critic_optimizer.step()
        critic_optimizer.zero_grad()
        V = critic_leader(epoch_observations)

    return float(np.mean(epoch_returns))

def compute_return(episode_rewards, gamma):
    
    discounted_reward = 0
    episode_returns = []
    
    for rew in reversed(episode_rewards):
        discounted_reward = rew + gamma * discounted_reward

    episode_returns.insert(0, discounted_reward)

    return episode_returns

def pretrain(epochs=40) -> nn.Module:

    env = IteratedMatrixGame(matrix="prisoners_dilemma", episode_length=10, memory=2)
    env = FollowerWrapper(env=env, num_queries=5)

    number_observation_features = env.num_queries + 1
    number_actions = env.action_space("follower").n
    model = create_model(number_observation_features, number_actions)

    optimizer = Adam(model.parameters(), 1e-2)

    for epoch in range(epochs):
        average_return = pretrain_one_epoch_pg(env=env, model=model, optimizer=optimizer)
        print('epoch: %3d \t return: %.3f' % (epoch, average_return))
        wandb.log({"epoch": epoch, "return": average_return})

    torch.save(model.state_dict(), os.path.join(os.path.abspath(os.path.dirname(__file__)),"checkpoints", "pretrained.pth"))
    print("Pretraining done.")

    return model

def train(actor_follower, epochs=40) -> None:

    env = IteratedMatrixGame(matrix="prisoners_dilemma", episode_length=10, memory=2)
    env = FollowerWrapper(env=env, num_queries=5)

    number_observation_features = 1
    number_actions = env.action_space("leader").n
    actor_leader = create_model(number_observation_features, number_actions)
    critic_leader = create_model(number_observation_features, 1)

    actor_optimizer = Adam(actor_leader.parameters(), 1e-2)
    critic_optimizer = Adam(critic_leader.parameters(), 1e-1)

    for epoch in range(epochs):
        average_return = train_one_epoch_ppo(env, actor_leader, critic_leader, actor_follower, actor_optimizer, critic_optimizer)
        print('epoch: %3d \t return: %.3f' % (epoch, average_return))
        wandb.log({"epoch": epoch, "return": average_return})

def test_pretrain(actor: nn.Module) -> None:
    
    # for response in ALL_POSSIBLE_RESPONSE:
    for response in torch.as_tensor([[1,1,1,1,1], # always defected
                                     [0,0,0,0,0], # always cooperate
                                     [0,0,0,1,1]]): # tit-for-tat

        for s in range(5):
            observation = torch.cat((torch.as_tensor([s]), response))
            policy = get_policy(model=actor, observation=observation)
            action, _ = get_action(policy=policy)
            print(f"state: {s}, context: {response.tolist()}, action: {action}, policy: {policy.probs.tolist()}")

if __name__ == '__main__':
    actor_follower = pretrain(epochs=400)
    test_pretrain(actor_follower)
