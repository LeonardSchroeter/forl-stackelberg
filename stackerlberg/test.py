import os
import torch

from envs.matrix_game import IteratedMatrixGame
from wrappers.follower import FollowerWrapper
from ppo import PPOLeaderFollower, INIT_LEADER_OBS

def test_pretraining():
    
    file_dir = os.path.abspath(os.path.dirname(__file__))
    env = IteratedMatrixGame(matrix="prisoners_dilemma", episode_length=10, memory=2)
    env = FollowerWrapper(env=env, num_queries=5)
    learner = PPOLeaderFollower(env=env)
    learner.follower_actor.load_state_dict(torch.load(os.path.join(file_dir, "follower_oracle.pth")))

    random_leader_responses = learner.random_policy(obs=INIT_LEADER_OBS)
    learner.env.set_leader_response(random_leader_responses)
    obs = env.reset()

    for k in range(20):
        actions = {}
        actions["leader"] = learner.random_policy(obs=torch.tensor([obs["leader"]]))
        actions["follower"], _ = learner.get_actions(actor=learner.follower_actor, obs=obs["follower"])
        print(actions["follower"])
        obs, rewards, _, _, _ = env.step(actions)

if __name__ == "__main__":
    test_pretraining()