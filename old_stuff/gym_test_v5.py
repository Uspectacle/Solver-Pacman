import numpy as np
import warnings
warnings.filterwarnings('ignore', category=RuntimeWarning)
warnings.filterwarnings('ignore', category=FutureWarning)
import tensorflow as tf
import gym
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import datetime
import time
from gym import wrappers
import matplotlib.pyplot as plt
from envs.pacman import PacmanEnv



class MyModel(tf.keras.Model):
    def __init__(self, shape_states, num_actions):
        super(MyModel, self).__init__()
        self.input_layer = tf.keras.layers.InputLayer(input_shape=shape_states)
        self.hidden_layers = []
        self.hidden_layers.append(tf.keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='RandomNormal'))
        self.hidden_layers.append(tf.keras.layers.Conv2D(64, (5, 5), activation='relu', kernel_initializer='RandomNormal'))
            # self.hidden_layers.append(tf.keras.layers.MaxPooling2D((2, 2)))
        self.hidden_layers.append(tf.keras.layers.Flatten())
        self.hidden_layers.append(tf.keras.layers.Dropout(0.5)) # to avoid any overfitting 
        self.hidden_layers.append(tf.keras.layers.Dense(50, activation='tanh', kernel_initializer='RandomNormal'))
        self.hidden_layers.append(tf.keras.layers.Dense(50, activation='tanh', kernel_initializer='RandomNormal'))
        self.output_layer = tf.keras.layers.Dense(num_actions, activation='sigmoid', kernel_initializer='RandomNormal')


    @tf.function
    def call(self, inputs):
        z = self.input_layer(inputs)
        for layer in self.hidden_layers:
            z = layer(z)
        output = self.output_layer(z)
        return output



class DQN:
    def __init__(self, shape_states, num_actions, conv_units, dense_units, gamma, max_experiences, min_experiences, batch_size, lr):
        self.num_actions = num_actions
        self.shape_states = shape_states
        self.batch_size = batch_size
        self.optimizer = tf.optimizers.Adam(lr)
        self.gamma = gamma
        self.model = MyModel(shape_states, conv_units, dense_units, num_actions)
        self.experience = {'s': [], 'a': [], 'r': [], 's2': [], 'done': []}
        self.max_experiences = max_experiences
        self.min_experiences = min_experiences
        self.loss = 0

    def predict(self, inputs):
        #while len(inputs.shape) <= len(self.shape_states):
        return self.model(inputs.astype('float32'))
        
    
    #@tf.function
    def train(self, TargetNet):
        if len(self.experience['s']) < self.min_experiences:
            return 0
        elif len(self.experience['s']) == self.min_experiences:
            print("Successfully reached minimum experience")
        
        ids = np.random.randint(low=0, high=len(self.experience['s']), size=self.batch_size)
        
        states = np.asarray([self.experience['s'][i] for i in ids])
        actions = np.asarray([self.experience['a'][i] for i in ids])
        rewards = np.asarray([self.experience['r'][i] for i in ids])
        states_next = np.asarray([self.experience['s2'][i] for i in ids])
        dones = np.asarray([self.experience['done'][i] for i in ids])

        value_next = TargetNet.predict(states_next)
        # while tf.math.greater(tf.shape(tf.shape(value_next)), tf.shape(tf.shape(rewards))) :
        # for i in range(len(self.shape_states)-1):
        value_next = tf.math.reduce_max(value_next, axis=1)
        rewards = tf.convert_to_tensor(rewards, dtype=tf.float32)
        actual_values = tf.where(dones, rewards, rewards + self.gamma*value_next)

        with tf.GradientTape() as tape:
            selected_action_values = tf.math.reduce_sum(self.predict(states) * tf.one_hot(actions, self.num_actions), axis=1)
            loss = tf.math.reduce_sum(tf.square(actual_values - selected_action_values))
        variables = self.model.trainable_variables
        gradients = tape.gradient(loss, variables)
        self.optimizer.apply_gradients(zip(gradients, variables))
        self.loss = loss.numpy() #.eval() it does not work, my bigest problem
        return


    def get_action(self, state, epsilon):
        if np.random.random() < epsilon:
            return np.random.choice(self.num_actions)
        else:
            states = np.expand_dims(state, axis=0)
            return np.argmax(self.predict(states)[0])

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
    if show:
        f = open("last_game.txt","w")
    rewards = 0
    losss = 0
    iter = 0
    done = False
    observations = env.reset()
    while not done:
        action = TrainNet.get_action(observations, epsilon)
        prev_observations = observations
        observations, reward, done, _ = env.step(action)
        rewards += reward
        if show:
            f.write(env.render())
        if done:
            if show:
                f.close()
            reward = -200
            env.reset()
        exp = {'s': prev_observations, 'a': action, 'r': reward, 's2': observations, 'done': done}
        TrainNet.add_experience(exp)
        TrainNet.train(TargetNet)
        losss += TrainNet.loss 
        iter += 1
        if iter % copy_step == 0:
            TargetNet.copy_weights(TrainNet)
    return [rewards, losss/iter]



def main(N, atari = False):
    print("Opening gym_environement")
    if atari:
        name = 'MsPacman-v0'
        env = gym.make('MsPacman-v0')
    else :
        name = 'Pacman-v6'
        env = PacmanEnv()
    print("Successfully open the environement :", name)
    gamma = 0.99
    copy_step = 100
    shape_states = env.observation_space.sample().shape
    print("shape_states :", shape_states)
    num_actions = env.action_space.n
    print("num_actions :", num_actions)
    max_experiences = 10000
    min_experiences = 1000
    batch_size = 32
    lr = 1e-2
    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir = 'logs/dqn/' + current_time
    summary_writer = tf.summary.create_file_writer(log_dir)
    print("Creation of DQN : TrainNet & TargetNet")
    TrainNet = DQN(shape_states, num_actions, gamma, max_experiences, min_experiences, batch_size, lr)
    print("Successfully created DQN : TrainNet")
    TargetNet = DQN(shape_states, num_actions, gamma, max_experiences, min_experiences, batch_size, lr)
    print("Successfully created DQN : TargetNet")
    N_avg = 100
    N_info = 10
    total_rewards = np.empty(N)
    total_loss = np.empty(N)
    epsilon = 0.99
    decay = 0.9999
    min_epsilon = 0.1
    start_time = time.time()
    total_times = np.empty(N_avg)
    show_first = False
    print("Training begin")
    # tf.config.experimental_run_functions_eagerly(True)
    for n in range(N):
         
        iter_time = time.time()
        epsilon = max(min_epsilon, epsilon * decay)

        [reward, loss] = play_game(env, TrainNet, TargetNet, epsilon, copy_step, show_first)
        total_rewards[n] = reward
        total_loss[n] = loss
        avg_rewards = total_rewards[max(0, n - N_avg):(n + 1)].mean()
        avg_loss    =    total_loss[max(0, n - N_avg):(n + 1)].mean()
        with summary_writer.as_default():
            tf.summary.scalar('episode reward', reward, step=n)
            tf.summary.scalar('running avg reward('+str(N_avg)+')', avg_rewards, step=n)
        total_times[n%N_avg] = time.time() - iter_time
        if n == 0:
            show_first = False
            print("Successfully finished the first episode in : ", time.time()-start_time, "s")
        if (n+1) % N_info == 0:
            print("episode: {0}/{1}, episode reward: {2:.1f}, eps: {3:.4f}, avg reward (last {4}): {5:.1f}, episode loss: {6:.1f}, avg loss: {7:.1f}, avg time: {8:.1f}, time total: {9:.1f}".format(
                  n+1, N, reward, epsilon, N_avg, avg_rewards, loss, avg_loss, total_times.mean(), time.time()-start_time))
        
    # play_game(env, TrainNet, TargetNet, epsilon, copy_step, show=True)
    env.close()
    print("Successfully trained in : ", time.time()-start_time, "s")
    TrainNet.model.save_weights("model_"+name+".h5")
    if not atari:
        play_game(env, TrainNet, TargetNet, epsilon, copy_step, True)
    print("Successfully saved model to disk")
    reward = [np.mean(total_rewards[min(0, i-N_avg) : i]) for i in range(len(total_rewards))]
    loss   = [np.mean(   total_loss[min(0, i-N_avg) : i]) for i in range(len(   total_loss))]
    print("Ploting reward and loss...")

    fig, axs = plt.subplots(2)
    axs[0].plot(reward)
    axs[0].set_title('reward')
    axs[1].plot(loss)
    axs[1].set_title('loss')
    plt.savefig('pacman_progress.png')
    print("Successfully reached the end !")
    return

main(5000, atari=False)
# main(10, atari=True)

