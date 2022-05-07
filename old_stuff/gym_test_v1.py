import numpy as np
import warnings
warnings.filterwarnings('ignore',category=FutureWarning)
import tensorflow as tf
import gym
import os
import datetime
from gym import wrappers
import matplotlib.pyplot as plt
from envs.pacman import PacmanEnv

def main():
    env = PacmanEnv()

    env._reset()
    episode_reward = 0
    while True:
        action = env.action_space.sample()
        _, reward, done, _ = env._step(action)
        episode_reward += reward
        if done:
            print('Reward: %s' % episode_reward)
            env._render()
            break

if __name__ == '__main__':
    main()