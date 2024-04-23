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

CKP_PATH = os.path.join(os.path.abspath(os.path.dirname(__file__)),"checkpoints", "pretrained.pth")
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
                                         [1,1,1,1,1]], 
                                         dtype=torch.float)

def create_model(number_observation_features: int, number_actions: int) -> nn.Module:

    hidden_layer_features = 32

    return nn.Sequential(
        nn.Linear(in_features=number_observation_features,
                  out_features=hidden_layer_features),
        # nn.ReLU(),
        # nn.Linear(in_features=hidden_layer_features,
        #           out_features=hidden_layer_features),
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

        # random_leader_responses = torch.as_tensor([env.action_space("leader").sample() for s in INIT_LEADER_OBS], dtype=torch.float)
        # random_leader_responses = torch.as_tensor([0, 0, 0, 1, 1], dtype=torch.float) # tit-for-tat
        random_leader_responses = ALL_POSSIBLE_RESPONSE[np.random.randint(0,ALL_POSSIBLE_RESPONSE.shape[0])] 
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

def pretrain(epochs=40, resume=False) -> nn.Module:

    env = IteratedMatrixGame(matrix="prisoners_dilemma", episode_length=10, memory=2)
    env = FollowerWrapper(env=env, num_queries=5)

    number_observation_features = env.num_queries + 1
    number_actions = env.action_space("follower").n
    model = create_model(number_observation_features, number_actions)

    if resume:
        print(CKP_PATH)
        model.load_state_dict(torch.load(CKP_PATH))

    optimizer = Adam(model.parameters(), 5e-2)

    for epoch in range(epochs):
        average_return = pretrain_one_epoch_pg(env=env, model=model, optimizer=optimizer, max_timesteps=5000)
        print('epoch: %3d \t return: %.3f' % (epoch, average_return))
        wandb.log({"epoch": epoch, "return": average_return})

    torch.save(model.state_dict(), CKP_PATH)
    print("Pretraining done.")

    return model

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
    actor_follower = pretrain(epochs=200, resume=True)
    test_pretrain(actor_follower)
