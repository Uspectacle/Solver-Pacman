import numpy as np
import tensorflow as tf
import gym
import os
import datetime
import time
from gym import wrappers
import matplotlib.pyplot as plt
from envs.pacman import PacmanEnv



class MyModel(tf.keras.Model):
    def __init__(self, num_actions, shape_states):
        super(MyModel, self).__init__()
        self.conv = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='RandomNormal', input_shape=shape_states)
        self.gmax = tf.keras.layers.GlobalMaxPool2D()
        self.flatten = tf.keras.layers.Flatten()
        self.d = tf.keras.layers.Dense(num_actions, activation='softmax', kernel_initializer='RandomNormal')

    @tf.function
    def call(self, x):
        x = self.conv(x)
        x = self.gmax(x)
        x = self.flatten(x)
        return self.d(x)

class DQN:
    def __init__(self, shape_states, num_actions, gamma, max_experiences, min_experiences, batch_size, lr):
        self.num_actions = num_actions
        self.shape_states = shape_states
        self.batch_size = batch_size
        self.optimizer = tf.optimizers.Adam(lr)
        self.gamma = gamma
        self.model = MyModel(num_actions, shape_states)
        self.experience = {'s': [], 'a': [], 'r': [], 's2': [], 'done': []}
        self.max_experiences = max_experiences
        self.min_experiences = min_experiences
        self.loss = tf.keras.metrics.Mean(name='loss')
    
    @tf.function
    def train(self, TargetNet):
        ids = tf.random.uniform(shape=[self.batch_size], minval=0, maxval=len(self.experience['s']), dtype = tf.int64) 
        ids = tf.expand_dims(ids, 1) 

        states = tf.gather_nd(tf.constant(np.array(self.experience['s']), dtype=tf.float32), ids)
        actions = tf.gather_nd(tf.constant(np.array(self.experience['a']), dtype=tf.int32), ids)
        rewards = tf.gather_nd(tf.constant(np.array(self.experience['r']), dtype=tf.float32), ids)
        states_next = tf.gather_nd(tf.constant(np.array(self.experience['s2']), dtype=tf.float32), ids)
        dones = tf.gather_nd(tf.constant(np.array(self.experience['done'])), ids)

        value_next = tf.math.reduce_max(TargetNet.model(states_next), axis=1)
        actual_values = tf.where(dones, rewards, rewards + self.gamma*value_next)

        with tf.GradientTape() as tape:
            selected_action_values = tf.math.reduce_sum(self.model(states) * tf.one_hot(actions, self.num_actions), axis=1)
            loss = tf.math.reduce_sum(tf.square(actual_values - selected_action_values))
        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))

        self.loss(loss)
        return

    def get_action(self, state, epsilon):
        if np.random.random_sample() < epsilon:
            return np.random.choice(self.num_actions)
        else:
            states = np.expand_dims(state, axis=0)
            return np.argmax(self.model(states.astype('float32'))[0])

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



def play_game(env, TrainNet, TargetNet, epsilon, copy_step, show):
    sum_reward = 0
    iter = 0
    done = False
    observations = env.reset()
    TrainNet.loss.reset_states()
    if show:
        f = open("last_game.txt","w")
    while not done:
        action = TrainNet.get_action(observations, epsilon)
        prev_observations = observations
        observations, reward, done, _ = env.step(action)
        exp = {'s': prev_observations, 'a': action, 'r': reward, 's2': observations, 'done': done}
        TrainNet.add_experience(exp)
        sum_reward += reward

        if len(TrainNet.experience['s']) >= TrainNet.min_experiences:
            TrainNet.train(TargetNet)

        iter += 1
        if iter % copy_step == 0:
            TargetNet.copy_weights(TrainNet)
        if show:
            f.write('\n' + str(iter) + '\n')
            f.write(env.render())
    
    if show:
        f.close()

    if len(TrainNet.experience['s']) >= TrainNet.min_experiences:
        loss = TrainNet.loss.result().numpy()
    else:
        loss = 0
    return [sum_reward, loss]


def main():

    print("Opening gym environement")
    name = 'Pacman_v11'
    env = PacmanEnv(ghosts_number=1)
    print("Successfully open the environement :", name)
    
    N = 1000000
    N_avg = 500
    N_info = 100

    gamma = 0.99
    copy_step = 100

    shape_states = env.observation_space.sample().shape
    num_actions = env.action_space.n

    max_experiences = 10000
    min_experiences = 1000
    batch_size = 32
    lr = 1e-2

    print("Creation of DQN model :")
    TrainNet = DQN(shape_states, num_actions, gamma, max_experiences, min_experiences, batch_size, lr)
    TargetNet = DQN(shape_states, num_actions, gamma, max_experiences, min_experiences, batch_size, lr)

    TargetNet.model.build(tuple([batch_size]) + shape_states)
    TargetNet.model.summary()

    total_rewards = np.empty(N)
    total_loss = np.empty(N)
    total_times = np.empty(N)

    epsilon = 0.99
    decay = 0.99999
    min_epsilon = 0.1

    show = True
    start_time = time.time()
    iter_time = time.time()

    # tf.config.experimental_run_functions_eagerly(True)

    print("Training begin")
    for n in range(N):
        epsilon = max(min_epsilon, epsilon * decay)

        [reward, loss] = play_game(env, TrainNet, TargetNet, epsilon, copy_step, show)

        total_rewards[n] = reward
        total_loss[n] = loss
        total_times[n] = time.time() - iter_time
        iter_time = time.time()

        show = False
        if (n+1) % N_info == 0 or n == 0:
            show = True

            avg_rewards = total_rewards[max(0, n - N_avg) : 1 + n].mean()
            avg_loss = total_loss[max(0, n - N_avg) : 1 + n].mean()
            avg_time = total_times[max(0, n - N_avg) : 1 + n].mean()

            template = "Successfull episode: {0:6d}/{1:6d} | eps: {2:2.2f}% | reward: {3:5.0f} (avg) | loss: {4:6.0f} (avg) | time: {5:.3f} s (avg) | time total: {6:.0f}"
            print(template.format(n+1, N, epsilon*100, avg_rewards, avg_loss, avg_time, time.time()-start_time))

            TrainNet.model.save_weights(name+"_model.h5")

            fig, axs = plt.subplots(2)
            fig.suptitle(name+' model progression')
            reward = [total_rewards[i : min(n, 1 + i + N_avg)].mean() for i in range(max(0, n - N_avg))]
            axs[0].plot(reward)
            axs[0].set_title('reward')
            loss = [total_loss[i : min(n, 1 + i + N_avg)].mean() for i in range(max(0, n - N_avg))]
            axs[1].plot(loss)
            axs[1].set_title('loss')
            plt.savefig(name+'_progress.png')
            plt.close()
    return

main()