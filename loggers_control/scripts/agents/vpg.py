from __future__ import absolute_import, division, print_function

import numpy as np
import random
import tensorflow as tf
import rospy

from utils import gen_utils
from utils import solo_utils
from utils.gen_utils import bcolors
from tensorflow.keras.layers import Dense
from tensorflow.keras import Model

import pdb


# class Memory:
#     def __init__(self, memory_cap):
#         self.memory_cap = memory_cap
#         self.memory = []
#     def store(self, experience):
#         # pop a random experience if memory full
#         if len(self.memory) >= self.memory_cap:
#             self.memory.pop(random.randint(0, len(self.memory)-1))
#         self.memory.append(experience)
#
#     def sample_batch(self, batch_size):
#         # Select batch
#         if len(self.memory) < batch_size:
#             batch = random.sample(self.memory, len(self.memory))
#         else:
#             batch = random.sample(self.memory, batch_size)
#
#         return zip(*batch)

class VPGAgent:
    def __init__(self, hyp_params):
        # super(VPGAgent, self).__init__()
        # hyper-parameters
        self.epsilon = hyp_params["epsilon"]
        self.actions = hyp_params["actions"]
        self.gamma = hyp_params["gamma"]
        self.dim_state = hyp_params["dim_state"]
        self.num_episodes = hyp_params["num_episodes"]
        self.num_steps = hyp_params["num_steps"]
        self.batch_size = hyp_params["batch_size"]
        self.update_step = hyp_params["update_step"]
        self.wall_bonus = hyp_params["wall_bonus"]
        self.door_bonus = hyp_params["door_bonus"]
        self.dist_bonus = hyp_params["dist_bonus"]
        self.model_path = hyp_params["model_path"]
        # pi(a|s;theta)
        self.policy_net = tf.keras.models.Sequential([
            tf.keras.layers.Dense(128, input_shape=(self.dim_state, ), activation='relu'),
            # tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(len(self.actions), activation="softmax")
        ])
        adam = tf.keras.optimizers.Adam(lr=1e-3)
        self.policy_net.compile(
            optimizer=adam
            loss="mean_squared_error",
            metrics=["accuracy"]
        )
        self.policy_net.summary()
        self.pnet_callback = tf.keras.callbacks.ModelCheckpoint(
            self.model_path,
            save_weights_only=True,
            verbose=1
        )
        # init replay memory
        # self.replay_memory = Memory(memory_cap=50000)

    def train_one_epoch(self):
        pass

    def train(self, env):
        pass
