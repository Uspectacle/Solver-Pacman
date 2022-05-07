import numpy as np
import tensorflow as tf
import gym
import os
import datetime
import time
from gym import wrappers
import matplotlib.pyplot as plt
from envs.pacman import PacmanEnv

class DQN:
    def __init__(self, model, num_actions, max_experiences, min_experiences, batch_size, lr):
        self.num_actions = num_actions
        self.batch_size = batch_size
        self.optimizer = tf.optimizers.Adam(lr)
        self.model = model
        self.experience = {'s': [], 'a': [], 'r': [], 's2': [], 'done': []}
        self.max_experiences = max_experiences
        self.min_experiences = min_experiences
        self.loss = tf.keras.metrics.Mean(name='loss')

    @tf.function
    def train(self, TargetNet, gamma):
        ids = tf.random.uniform(shape=[self.batch_size], minval=0, maxval=len(self.experience['s']), dtype = tf.int64) 
        ids = tf.expand_dims(ids, 1) 

        states = tf.gather_nd(tf.constant(np.array(self.experience['s']), dtype=tf.float32), ids)
        actions = tf.gather_nd(tf.constant(np.array(self.experience['a']), dtype=tf.int32), ids)
        rewards = tf.gather_nd(tf.constant(np.array(self.experience['r']), dtype=tf.float32), ids)
        states_next = tf.gather_nd(tf.constant(np.array(self.experience['s2']), dtype=tf.float32), ids)
        dones = tf.gather_nd(tf.constant(np.array(self.experience['done'])), ids)

        value_next = tf.math.reduce_max(TargetNet.model(states_next), axis=1)
        actual_values = tf.where(dones, rewards, rewards + gamma*value_next)

        with tf.GradientTape() as tape:
            selected_action_values = tf.math.reduce_sum(self.model(states) * tf.one_hot(actions, self.num_actions), axis=1)
            loss = tf.keras.losses.MSE(actual_values, selected_action_values)
        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))

        self.loss(loss)
        return

    def get_actions(self, state, epsilon):
        if np.random.random_sample() < epsilon:
            return tf.one_hot(np.random.choice(self.num_actions), self.num_actions)
        else:
            states = np.expand_dims(state, axis=0)
            return self.model(states.astype('float32'))[0]

    def add_experience(self, exp):
        if len(self.experience['s']) >= self.max_experiences:
            for key in self.experience.keys():
                self.experience[key].pop(0)
        for key, value in exp.items():
            self.experience[key].append(value)

    def copy_weights(self, TrainNet):
        variables1 = self.model.trainable_variables
        variables2 = TrainNet.model.trainable_variables
        for v1, v2 in zip(variables1, variables2):
            v1.assign(v2.numpy())



def play_game(env, TrainNet, TargetNet, epsilon, copy_step, gamma, show):
    sum_reward = 0
    iter = 0
    done = False
    observations = env.reset()
    TrainNet.loss.reset_states()
    if show:
        f = open("last_game.txt","w")
        
    while not done:
        actions = TrainNet.get_actions(observations, epsilon)
        action = np.argmax(actions)
        prev_observations = observations
        observations, reward, done, _ = env.step(action)
        exp = {'s': prev_observations, 'a': action, 'r': reward, 's2': observations, 'done': done}
        TrainNet.add_experience(exp)
        sum_reward += reward

        if len(TrainNet.experience['s']) >= TrainNet.min_experiences:
            TrainNet.train(TargetNet, gamma)

        iter += 1
        if iter % copy_step == 0:
            TargetNet.copy_weights(TrainNet)
        if show:
            f.write('\n' + str(iter) + ' | actions : ' + str(actions) + ' | reward : ' + str(reward) + '\n')
            f.write(env.render())
    
    if show:
        f.close()

    if len(TrainNet.experience['s']) >= TrainNet.min_experiences:
        loss = TrainNet.loss.result().numpy()
    else:
        loss = 0
    return np.array([sum_reward, loss, env.score, env.time])


def main():

    print("Opening gym environement")
    name = 'Pacman_v11'
    env = PacmanEnv(ghosts_number=1)
    print("Successfully open the environement :", name)
    
    N = 1000000
    N_avg = 100
    N_info = 10

    gamma = 0.9
    copy_step = 100

    num_actions = env.action_space.n

    max_experiences = 10000
    min_experiences = 100
    batch_size = 200
    lr = 1e-2

    inputs = tf.keras.layers.Input(shape=env.observation_space.sample().shape)
    conv = tf.keras.layers.Conv2D(50, (2, 2), activation='relu', kernel_initializer='RandomNormal')(inputs)
    gmax = tf.keras.layers.GlobalMaxPool2D()(conv)
    flatten1 = tf.keras.layers.Flatten()(gmax)
    dense1 = tf.keras.layers.Dense(num_actions, activation='tanh', kernel_initializer='RandomNormal')(flatten1)


    flatten2 = tf.keras.layers.Flatten()(inputs)
    dense2 = tf.keras.layers.Dense(100, activation='relu', kernel_initializer='RandomNormal')(flatten2)
    dense3 = tf.keras.layers.Dense(num_actions, activation='tanh', kernel_initializer='RandomNormal')(dense2)


    add = tf.keras.layers.Add()([dense1, dense3])
    model = tf.keras.Model(inputs=inputs, outputs=add, name='on_test')
    model.summary()

    print("Creation of DQN model :")
    TrainNet = DQN(model, num_actions, max_experiences, min_experiences, batch_size, lr)
    TargetNet = DQN(model, num_actions, max_experiences, min_experiences, batch_size, lr)


    total_stat = np.empty((N, 5))

    epsilon = 0.99
    decay = 0.999
    min_epsilon = 0.001

    show = True
    start_time = time.time()
    iter_time = time.time()

    # tf.config.experimental_run_functions_eagerly(True)

    print("Training begin")
    for n in range(N):
        epsilon = max(min_epsilon, epsilon * decay)

        total_stat[n, 0:4] = play_game(env, TrainNet, TargetNet, epsilon, copy_step, gamma, show)

        total_stat[n, 4] = time.time() - iter_time
        iter_time = time.time()

        show = False
        if (n+1) % N_info == 0 or n == 0:
            show = True
            
            if n < N_avg :
                avg_stat = np.mean(total_stat[0 : 1 + n], axis=0, keepdims=1)
            else:
                avg_stat = np.array([np.mean(total_stat[i : 1 + min(n, i + N_avg)], axis=0) for i in range(n - N_avg)])
            

            template = "Successfull episode: {0:7d}/{1:7d} | eps: {2:5.2f}% | reward: {3:4.0f} (avg) | loss: {4:.3f} (avg) | time: {5:.3f} s (avg) | time total: {6:.0f}"
            print(template.format(n + 1, N, epsilon*100, avg_stat[-1, 0], avg_stat[-1, 1], avg_stat[-1, 4], time.time() - start_time))

            TrainNet.model.save_weights(name + "_model.h5")

            fig, axs = plt.subplots(4, figsize=(12, 12))
            fig.suptitle(name + ' model progression')
            for i in range(4):
                axs[i].plot(total_stat[: 1 + n, i], 'whitesmoke')
                axs[i].plot(avg_stat[: 1 + n, i], 'b')
            axs[0].set_title('reward')
            axs[1].set_title('loss')
            axs[2].set_title('score')
            axs[3].set_title('timer')
            plt.savefig(name + '_progress.png')
            plt.close()
    return

main()