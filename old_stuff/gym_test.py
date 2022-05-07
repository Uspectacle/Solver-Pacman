import numpy as np
import tensorflow as tf
import gym
import os
import datetime
from gym import wrappers
import matplotlib.pyplot as plt
from envs.pacman import PacmanEnv

class MyModel(tf.keras.Model):
    def __init__(self, obs_shape, conv_units, dense_units, num_actions):
        super(MyModel, self).__init__()
        self.input_layer = tf.keras.layers.Conv2D(conv_units[0], (3, 3), activation='relu', input_shape = obs_shape, kernel_initializer='RandomNormal')
        self.hidden_layers = []
        for i in conv_units[1:]:
            self.hidden_layers.append(tf.keras.layers.Conv2D(i, (3, 3), activation='relu', kernel_initializer='RandomNormal'))
            self.hidden_layers.append(tf.keras.layers.MaxPooling2D((2, 2)))
        self.hidden_layers.append(tf.keras.layers.Flatten())
        self.hidden_layers.append(tf.keras.layers.Dropout(0.5)) # to avoid any overfitting 
        for i in dense_units:
            self.hidden_layers.append(tf.keras.layers.Dense(i, activation='relu', kernel_initializer='RandomNormal'))
        self.output_layer = tf.keras.layers.Dense(num_actions, activation='sigmoid', kernel_initializer='RandomNormal')

    @tf.function
    def call(self, inputs):
        print('\ninput :\n')
        print(inputs)
        print('\n')
        z = self.input_layer(inputs)
        print('\nz :\n')
        print(z)
        print('\n')
        for layer in self.hidden_layers:
            z = layer(z)
            print('\nz :\n')
            print(z)
            print('\n')
        output = self.output_layer(z)
        print('\noutput :\n')
        print(output)
        print('\n')
        return output

class DQN:
    def __init__(self, obs_shape, num_actions, conv_units, dense_units, gamma, max_experiences, min_experiences, batch_size, lr):
        self.num_actions = num_actions
        self.batch_size = batch_size
        self.optimizer = tf.compat.v2.keras.optimizers.Adam(lr)
        self.gamma = gamma
        self.model = MyModel(obs_shape, conv_units, dense_units, num_actions)
        self.experience = {'s': [], 'a': [], 'r': [], 's2': [], 'done': []}
        self.max_experiences = max_experiences
        self.min_experiences = min_experiences

    def predict(self, inputs):
        return (batchsize, 4)   #self.model(np.atleast_2d(inputs.astype('float32')))
    
    @tf.function
    def train(self, TargetNet):
        if len(self.experience['s']) < self.min_experiences:
            return 0
        ids = np.random.randint(low=0, high=len(self.experience['s']), size=self.batch_size)
        states = np.asarray([self.experience['s'][i] for i in ids])
        actions = np.asarray([self.experience['a'][i] for i in ids])
        rewards = np.asarray([self.experience['r'][i] for i in ids])
        states_next = np.asarray([self.experience['s2'][i] for i in ids])
        dones = np.asarray([self.experience['done'][i] for i in ids])
        value_next = np.max(TargetNet.predict(states_next), axis=1)
        actual_values = np.where(dones, rewards, rewards+self.gamma*value_next)

        with tf.GradientTape() as tape:
            selected_action_values = tf.math.reduce_sum(
                self.predict(states) * tf.one_hot(actions, self.num_actions), axis=1)
            loss = tf.math.reduce_sum(tf.square(actual_values - selected_action_values))
        variables = self.model.trainable_variables
        gradients = tape.gradient(loss, variables)
        self.optimizer.apply_gradients(zip(gradients, variables))

    def get_action(self, states, epsilon):
        if np.random.random() < epsilon:
            return np.random.choice(self.num_actions)
        else:
            return np.argmax(self.predict(np.atleast_2d(states))[0])

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

def play_game(env, TrainNet, TargetNet, epsilon, copy_step):
    rewards = 0
    iter = 0
    done = False
    observations = env._reset()
    while not done:
        action = TrainNet.get_action(observations, epsilon)
        prev_observations = observations
        observations, reward, done, _ = env._step(action)
        rewards += reward
        if done:
            reward = -200
            env._reset()

        exp = {'s': prev_observations, 'a': action, 'r': reward, 's2': observations, 'done': done}
        TrainNet.add_experience(exp)
        TrainNet.train(TargetNet)
        iter += 1
        if iter % copy_step == 0:
            TargetNet.copy_weights(TrainNet)
    return rewards

def main():
    env = PacmanEnv()
    gamma = 0.99
    copy_step = 25
    obs_shape = env.observation_space.shape
    num_actions = env.action_space.n
    conv_units = [64, 128]
    dense_units = [100, 50, 100]
    max_experiences = 10000
    min_experiences = 100
    batch_size = 32
    lr = 1e-2
    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir = 'logs/dqn/' + current_time
    #summary_writer = tf.summary.create_file_writer(log_dir)

    TrainNet = DQN(obs_shape, num_actions, conv_units, dense_units, gamma, max_experiences, min_experiences, batch_size, lr)
    TargetNet = DQN(obs_shape, num_actions, conv_units, dense_units, gamma, max_experiences, min_experiences, batch_size, lr)
    N = 50000
    total_rewards = np.empty(N)
    epsilon = 0.99
    decay = 0.9999
    min_epsilon = 0.1
    for n in range(N):
        epsilon = max(min_epsilon, epsilon * decay)
        total_reward = play_game(env, TrainNet, TargetNet, epsilon, copy_step)
        total_rewards[n] = total_reward
        avg_rewards = total_rewards[max(0, n - 100):(n + 1)].mean()
        #with summary_writer.as_default():
        #    tf.summary.scalar('episode reward', total_reward, step=n)
        #    tf.summary.scalar('running avg reward(100)', avg_rewards, step=n)
        if n % 100 == 0:
            print("episode:", n, "episode reward:", total_reward, "eps:", epsilon, "avg reward (last 100):", avg_rewards)
    print("avg reward for last 100 episodes:", avg_rewards)
    env.close()
    return total_rewards

total_rewards = main()