import numpy as np
import tensorflow as tf
import gym
import os
import datetime
import time
from gym import wrappers
import matplotlib.pyplot as plt

class DQN:
    def __init__(self, model):
        self.model = model
        self.experience = {'s': [], 'a': [], 'r': [], 's2': [], 'done': []}
        self.ids = tf.keras.metrics.MeanTensor(name='ids')
        self.states = tf.keras.metrics.MeanTensor(name='states')
        self.actions = tf.keras.metrics.MeanTensor(name='actions')
        self.rewards = tf.keras.metrics.MeanTensor(name='rewards')
        self.states_next = tf.keras.metrics.MeanTensor(name='states_next')
        self.dones = tf.keras.metrics.MeanTensor(name='dones')
        self.predicted_Q_next = tf.keras.metrics.MeanTensor(name='predicted_Q_next')
        self.actual_Q = tf.keras.metrics.MeanTensor(name='actual_Q')
        self.predicted_Q = tf.keras.metrics.MeanTensor(name='predicted_Q')
        self.loss = tf.keras.metrics.Mean(name='loss')


    @tf.function
    def train(self, TargetNet, num_actions, gamma, batch_size, optimizer):
        ids = tf.random.uniform(shape=[batch_size], minval=0, maxval=len(self.experience['s']), dtype = tf.int64) 
        ids = tf.expand_dims(ids, 1) 
        self.ids(ids)
        states = tf.gather_nd(tf.constant(np.array(self.experience['s']), dtype=tf.float32), ids)
        actions = tf.gather_nd(tf.constant(np.array(self.experience['a']), dtype=tf.int32), ids)
        rewards = tf.gather_nd(tf.constant(np.array(self.experience['r']), dtype=tf.float32), ids)
        states_next = tf.gather_nd(tf.constant(np.array(self.experience['s2']), dtype=tf.float32), ids)
        dones = tf.gather_nd(tf.constant(np.array(self.experience['done'])), ids)
        self.states(states)
        self.actions(actions)
        self.rewards(rewards)
        self.states_next(states_next)
        self.dones(dones)

        predicted_Q_next = tf.math.reduce_max(TargetNet.model(states_next), axis=1)
        actual_Q = tf.where(dones, rewards, rewards + gamma*predicted_Q_next)
        self.predicted_Q_next(predicted_Q_next)
        self.actual_Q(actual_Q)

        with tf.GradientTape() as tape:
            predicted_Q = tf.math.reduce_sum(self.model(states) * tf.one_hot(actions, num_actions), axis=1)
            loss = tf.keras.losses.MSE(actual_Q, predicted_Q)

        self.predicted_Q(predicted_Q)
        gradients = tape.gradient(loss, self.model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))

        self.loss(loss)
        return ids

    def get_Qs(self, state, epsilon, num_actions):
        if np.random.random_sample() < epsilon:
            return tf.one_hot(np.random.choice(num_actions), num_actions)
        else:
            states = np.expand_dims(state, axis=0)
            return self.model(states.astype('float32'))[0]

    def add_experience(self, exp, max_experiences):
        if len(self.experience['s']) >= max_experiences:
            for key in self.experience.keys():
                self.experience[key].pop(0)
        for key, value in exp.items():
            self.experience[key].append(value)

    def copy_weights(self, TrainNet):
        variables1 = self.model.trainable_variables
        variables2 = TrainNet.model.trainable_variables
        for v1, v2 in zip(variables1, variables2):
            v1.assign(v2.numpy())

def play_game(env, TrainNet, TargetNet, epsilon, copy_step, gamma, batch_size, optimizer, min_experiences, max_experiences, show, boolen):
    sum_reward = 0
    timer = 0
    done = False
    observations = env.reset()
    if show:
        f = open("last_game.txt","w")
        
    while not done:
        TrainNet.ids.reset_states()
        TrainNet.states.reset_states()
        TrainNet.actions.reset_states()
        TrainNet.rewards.reset_states()
        TrainNet.states_next.reset_states()
        TrainNet.dones.reset_states()
        TrainNet.predicted_Q_next.reset_states()
        TrainNet.actual_Q.reset_states()
        TrainNet.predicted_Q.reset_states()
        TrainNet.loss.reset_states()

        Qs = TrainNet.get_Qs(observations, epsilon, env.action_space.n)
        action = np.argmax(Qs)
        prev_observations = observations
        observations, reward, done, _ = env.step(action)
        exp = {'s': prev_observations, 'a': action, 'r': reward, 's2': observations, 'done': done}
        TrainNet.add_experience(exp, max_experiences)
        sum_reward += reward

        if len(TrainNet.experience['s']) >= min_experiences:
            # print("ids_test = ", TrainNet.train(TargetNet, env.action_space.n, gamma, batch_size, optimizer))
            if boolen :
                # print("ids_test = ", TrainNet.ids.result().numpy())
                # print("states_test = ", TrainNet.states.result().numpy())
                print("actions_test = ", TrainNet.actions.result().numpy())
                print("rewards_test = ", TrainNet.rewards.result().numpy())
                # print("states_next_test = ", TrainNet.states_next.result().numpy())
                print("dones_test = ", TrainNet.dones.result().numpy())
                print("predicted_Q_next_test = ", TrainNet.predicted_Q_next.result().numpy())
                print("actual_Q_test = ", TrainNet.actual_Q.result().numpy())
                print("predicted_Q_test = ", TrainNet.predicted_Q.result().numpy())
                print("loss_test = ", TrainNet.loss.result().numpy())
                print("len = ", len(TrainNet.experience['s']))
                boolen = False

        timer += 1
        if timer % copy_step == 0:
            TargetNet.copy_weights(TrainNet)
        if show:
            template = "\n {0:3d} | Q values : ^ {1:5.2f}, v {2:5.2f}, < {3:5.2f}, > {4:5.2f} | Reward: {5:2d}\n"
            f.write(template.format(timer, Qs.numpy()[0], Qs.numpy()[1], Qs.numpy()[2], Qs.numpy()[3], reward))
            f.write(env.render())
    
    if show:
        f.close()

    if len(TrainNet.experience['s']) >= min_experiences:
        loss = TrainNet.loss.result().numpy()
    else:
        loss = None
    return np.array([sum_reward, loss, env.score, env.time])

def progress(total_stat, avg_stat, model, name, n):
    fig, axs = plt.subplots(6, figsize = [6.4, 8])
    fig.suptitle(name + " with " + model)

    axs[0].plot(total_stat[: 1 + n, 0], 'lavender')
    axs[0].plot(avg_stat[: 1 + n, 0], 'b', label = 'Rewards')
    axs[0].legend(loc='upper right', handlelength = 0)
    axs[0].get_xaxis().set_visible(False)

    axs[1].semilogy(total_stat[: 1 + n, 1], 'lavender')
    axs[1].semilogy(avg_stat[: 1 + n, 1], 'b', label = 'Loss')
    axs[1].legend(loc='upper right', handlelength = 0)
    axs[1].get_xaxis().set_visible(False)

    axs[2].plot(total_stat[: 1 + n, 2], 'lavender')
    axs[2].plot(avg_stat[: 1 + n, 2], 'b', label = 'Score')
    axs[2].legend(loc='upper right', handlelength = 0)
    axs[2].get_xaxis().set_visible(False)

    axs[3].plot(total_stat[: 1 + n, 3], 'lavender')
    axs[3].plot(avg_stat[: 1 + n, 3], 'b', label = 'Timer')
    axs[3].legend(loc='upper right', handlelength = 0)
    axs[3].get_xaxis().set_visible(False)

    axs[4].plot(total_stat[: 1 + n, 2] / total_stat[: 1 + n, 3], 'lavender')
    axs[4].plot(avg_stat[: 1 + n, 2] / avg_stat[: 1 + n, 3], 'b', label = 'Score / Timer')
    axs[4].legend(loc='upper right', handlelength = 0)
    axs[4].get_xaxis().set_visible(False)

    axs[5].plot(total_stat[: 1 + n, 5], 'lavender')
    axs[5].plot(avg_stat[: 1 + n, 5], 'b', label = 'epsilon')
    axs[5].legend(loc='upper right', handlelength = 0)

    plt.savefig(name + '_' + model + '_progress.png')
    plt.close()
    return

def filters(env, model):
    for layer in model.layers:
        if 'conv' in layer.name :
            filters, biases = layer.get_weights()
            f_min, f_max = filters.min(), filters.max()
            filters = (filters - f_min) / (f_max - f_min)
            n_filter = min(filters.shape[-1], 50)
            fig, axs = plt.subplots(n_filter, env.action_space.n, figsize = [6.4, 2.4 + 0.9*n_filter])
            fig.suptitle(layer.name)
            for i in range(n_filter):
                f = filters[:, :, :, i]
                for j in range(env.action_space.n):
                    axs[i, j].set_xticks([])
                    axs[i, j].set_yticks([])
                    axs[i, j].imshow(f[:, :, j], cmap='gray', label = str(biases[i]), vmin=0, vmax=1)
            plt.savefig(env.name + '_' + model.name + '_filters.png')
            plt.close()
            break

def main(   env, 
            model, 
            optimizer = tf.keras.optimizers.Adam(1e-2),
            lr_min = 1e-2,
            lr_max = 1e-100,
            min_experiences = 100,
            max_experiences = 10000,
            batch_size = 200,
            gamma = 0.9,
            copy_step = 100,
            epsilon = 0.99,
            decay = 0.999,
            min_epsilon = 0.001,
            N = 1000000,
            N_avg = 100,
            N_info = 10,
            N_show = 100,
            N_progress = 100,
            N_filters = 100):

    TrainNet = DQN(model)
    TargetNet = DQN(model)

    total_stat = np.empty((N, 6))

    show = True
    start_time = time.time()
    iter_time = time.time()

    # tf.config.experimental_run_functions_eagerly(True)
    boolen = False
    print("Training begin")
    for n in range(N):
        epsilon = max(min_epsilon, epsilon * decay)
        total_stat[n, 5] = epsilon

        total_stat[n, 0:4] = play_game( env, 
                                        TrainNet, 
                                        TargetNet, 
                                        epsilon, 
                                        copy_step, 
                                        gamma, 
                                        batch_size, 
                                        optimizer, 
                                        min_experiences, 
                                        max_experiences, 
                                        show, boolen)

        total_stat[n, 4] = time.time() - iter_time
        iter_time = time.time()

        show = False
        if (n+1) % N_info == 0:
            if n < N_avg :
                avg_stat = np.mean(total_stat[0 : 1 + n], axis=0, keepdims=1)
            else:
                avg_stat = np.array([np.mean(total_stat[i : 1 + min(n, i + N_avg)], axis=0) for i in range(n - N_avg)])
            template = "Successfull episode: {0:7d}/{1:7d} | eps: {2:5.2f}% | reward: {3:3.0f} (avg) | loss: {4:.2e} (avg) | time: {5:.3f} s (avg) | time total: {6:.0f}"
            print(template.format(n + 1, N, epsilon*100, avg_stat[-1, 0], avg_stat[-1, 1], avg_stat[-1, 4], time.time() - start_time))
            TrainNet.model.save_weights(env.name + "_" + model.name + ".h5")
            optimizer.learning_rate = max(lr_max, min(lr_min, lr_min*avg_stat[-1, 1]*10))
            boolen = True

        if (n+1) % N_show == 0:
            show = True

        if (n+1) % N_progress == 0:
            progress(total_stat, avg_stat, model.name, env.name, n)
            
        if (n+1) % N_filters == 0:
            filters(env, model)
    return