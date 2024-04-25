import numpy as np
import pandas as pd
from stable_baselines3.td3.policies import TD3Policy
import torch as th
import torch.nn as nn
import time

from stable_baselines3 import TD3, DDPG
from stable_baselines3.td3 import MlpPolicy
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise
from torch.nn.modules import activation

from environment import SimpleSupplyChain


def make_policy(net_architecture=[512, 512]):
    policy = MlpPolicy(observation_space=SimpleSupplyChain.observation_space,
                       action_space=SimpleSupplyChain.action_space,
                       lr_schedule=0.1,
                       net_arch=net_architecture)

    return policy

# Reward Mean: 7041.430871062279 [std. dev: 465.25285975768423] (episodes: 1000)


def train_td3(timesteps=5e5, net_architecture=None):
    print("Created Environment...")
    env = SimpleSupplyChain()

    n_actions = env.action_space.shape[-1]
    action_noise = NormalActionNoise(mean=np.zeros(
        n_actions), sigma=0.1 * np.ones(n_actions))

    if net_architecture:
        policy_kwargs = {
            # "activation_fn": th.nn.ReLU,
            "net_arch": net_architecture
        }
    else:
        policy_kwargs = {}

    agent = TD3(policy="MlpPolicy", env=env,
                 action_noise=action_noise, policy_kwargs=policy_kwargs,
                 verbose=1, tensorboard_log="./tensorboard/TD3")

    print("Starting Model Training...")
    agent.learn(total_timesteps=timesteps, log_interval=10)

    file_name = f"td3_{int(time.time())}"
    agent.save(file_name)

    print(f"Training Finished. Model saved as >>> {file_name}")
    return agent


# Reward Mean: 6705.04089457035 [std. dev: 366.5834462025138] (episodes: 1000)
def train_ddpg(timesteps=5e5):
    print("Created Environment...")
    env = SimpleSupplyChain()

    n_actions = env.action_space.shape[-1]

    action_noise = OrnsteinUhlenbeckActionNoise(mean=np.zeros(
        n_actions), sigma=0.1 * np.ones(n_actions))

    agent = DDPG("MlpPolicy", env, action_noise=action_noise,
                 verbose=1, tensorboard_log="./tensorboard/DDPG")

    print("Starting Model Training...")
    agent.learn(total_timesteps=timesteps, log_interval=10)

    file_name = f"ddpg_{int(time.time())}"
    agent.save(file_name)

    print(f"Training Finished. Model saved as >>> {file_name}")
    return agent


def load_td3(file_name):
    agent = TD3.load(file_name)
    return agent


def load_ddpg(file_name):
    agent = DDPG.load(file_name)
    return agent


def test_agent(agent1, agent2, log=False, num_episodes=10):
    env = SimpleSupplyChain()

    total_rewards = []
    transitions = []

    for episode in range(num_episodes):
        obs = env.reset()
        total_reward = 0
        done = False

        t = 0

        while not done:
            action1, _states1 = agent1.predict(obs)
            action2, _states2 = agent2.predict(obs)
