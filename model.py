from DQN import main
from pacman import PacmanEnv
from new_layer import Cropping2D_custom
import numpy as np
import tensorflow as tf
import h5py

tf.config.experimental_run_functions_eagerly(True)

env = PacmanEnv(ghosts_number=1)

class MyModel(tf.keras.Model):
    def __init__(self, inputs_shape, num_actions):
        super(MyModel, self).__init__()
        self.inputs = tf.keras.layers.InputLayer(input_shape=inputs_shape)
        self.crop = Cropping2D_custom((3, 3), 2)
        self.flatten = tf.keras.layers.Flatten()
        self.outputs = tf.keras.layers.Dense(num_actions, activation = 'linear', kernel_initializer = 'RandomNormal')
        # self.view = tf.keras.metrics.MeanTensor(name='view')
        # self.view2 = tf.keras.metrics.MeanTensor(name='view2')
        

    @tf.function
    def call(self, x):
        x = self.inputs(x)
        self.view2=x[0]
        x = self.crop(x)
        self.view=x[0]
        x = self.flatten(x)
        return self.outputs(x)

model = MyModel(env.observation_space.sample().shape, env.action_space.n)
# model.ids.reset_states()
# model.crop.box.reset_states()

input_test_0 = np.zeros([11, 11, 4])
input_test_1 = np.zeros([11, 11, 4])
input_test_2 = np.zeros([11, 11, 4])
input_test_3 = np.zeros([11, 11, 4])
input_test_4 = np.zeros([11, 11, 4])
input_test_5 = np.zeros([11, 11, 4])
input_test_6 = np.zeros([11, 11, 4])

input_test_0[2,3,3] = 1
input_test_1[1,1,3] = 1
input_test_2[3,3,3] = 1
input_test_3[2,1,3] = 1
input_test_4[8,7,3] = 1
input_test_5[9,3,3] = 1
input_test_6[9,7,3] = 1

input_test = tf.concat([tf.expand_dims(input_test_0.astype("float32"), axis=0), 
                        tf.expand_dims(input_test_1.astype("float32"), axis=0), 
                        tf.expand_dims(input_test_2.astype("float32"), axis=0), 
                        tf.expand_dims(input_test_3.astype("float32"), axis=0), 
                        tf.expand_dims(input_test_4.astype("float32"), axis=0), 
                        tf.expand_dims(input_test_5.astype("float32"), axis=0), 
                        tf.expand_dims(input_test_6.astype("float32"), axis=0)], axis = 0)

model(input_test)

# print("ids_test = \n", np.argwhere(model.ids.result().numpy()==1))
# print("boxes = ", model.crop.box.result().numpy())

# inputs = tf.keras.layers.Input(shape=env.observation_space.sample().shape)
# crop = Cropping2D_custom(env.observation_space.sample().shape[0], env.observation_space.sample().shape[1], 3, 3, 3)(inputs)
# conv1 = tf.keras.layers.Conv2D(64, (2, 2), activation='relu', kernel_initializer='RandomNormal')(inputs)
# conv2 = tf.keras.layers.Conv2D(128, (2, 2), activation='relu', kernel_initializer='RandomNormal')(conv1)

# gmax = tf.keras.layers.GlobalMaxPool2D()(conv2)

# flatten = tf.keras.layers.Flatten()(conv1)

# dense1 = tf.keras.layers.Dense(env.action_space.n, activation='linear', kernel_initializer='RandomNormal')(flatten1)
# conv1 = tf.keras.layers.Conv2D(2048, (3, 3), activation='relu', kernel_initializer='RandomNormal')(inputs)
# gmax = tf.keras.layers.GlobalMaxPool2D()(conv1)
# conv2 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='RandomNormal')(conv1)
# flatten2 = tf.keras.layers.Flatten()(gmax)

# dense1 = tf.keras.layers.Dense(128, activation='relu', kernel_initializer='RandomNormal')(flatten)
# outputs = tf.keras.layers.Dense(env.action_space.n, activation='linear', kernel_initializer='RandomNormal')(flatten)

# add = tf.keras.layers.Add()([dense1, dense3])

# model = tf.keras.Model(inputs=inputs, outputs=outputs, name='crop')

# model.save("C:/Users/Mario/Desktop/gym_pacman/" + env.name + "_" + model.name + ".h5") # 'c22_128_c33_128_3d_128.h5')# env.name + "_" + model.name + ".h5")
# model.load_weights("C:/Users/Mario/Desktop/gym_pacman/" + env.name + "_" + model.name + ".h5") #env.name + "_" + model.name + ".h5")
# model = tf.keras.models.load_model("C:/Users/Mario/Desktop/gym_pacman/" + env.name + "_" + model.name + ".h5") # 'c22_128_c33_128_3d_128.h5')# env.name + "_" + model.name + ".h5")


if False: # True: # 
        model_name = model.name
        json_file = open(env.name + "_" + model_name + ".json", 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        model = tf.keras.models.model_from_json(loaded_model_json)
        model.load_weights(env.name + "_" + model_name + ".h5")

model.summary()

main(   env, 
        model, 
        optimizer = tf.keras.optimizers.Adam(1e-2),
        lr_min = 1e-1,
        lr_max = 1e-16,
        min_experiences = 128,
        max_experiences = 8192,
        batch_size = 128,
        gamma = 0,
        copy_step = 100,
        epsilon = 0.9,
        decay = 0.999,
        min_epsilon = 0.01,
        N = 1000000,
        N_avg = 100,
        N_info = 1,
        N_show = 10,
        N_progress = 10, 
        N_filters = 100)