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
    # 1. Load Environment and Q-table structure
    env = PacmanEnv()
    print("action_space : " + str(env.action_space))
    print("observation_space : " + str(env.observation_space))
    Q = np.zeros([env.observation_space.n, env.action_space.n])

    # 2. Parameters of Q-leanring
    eta = .628
    gma = .9
    epis = 31
    rev_list = [] # rewards per episode calculate

    # 3. Q-learning Algorithm
    for i in range(epis):
        # Reset environment
        s = env._reset()
        if i%10 ==0 :
            env._render()
        rAll = 0
        d = False
        j = 0
        #The Q-Table learning algorithm
        while j < 99:
            j+=1
            # Choose action from Q table
            a = np.argmax(Q[s,:] + np.random.randn(1,env.action_space.n)*(1./(i+1)))
            #Get new state & reward from environment
            if i%1 ==0 :
                print("action : " + str(a))
                if a>3:
                    print("erreur : ")
                    print(Q.shape)
            s1,r,d,_ = env._step(a)
            #Update Q-Table with new knowledge
            Q[s,a] = Q[s,a] + eta*(r + gma*np.max(Q[s1,:]) - Q[s,a])
            rAll += r
            s = s1
            if d == True:
                break
        rev_list.append(rAll)
        if i%10 ==0 :
            env._render()

if __name__ == '__main__':
    main()