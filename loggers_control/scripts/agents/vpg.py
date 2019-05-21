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
    def __init__(self, params):
        # super(VPGAgent, self).__init__()
        # hyper-parameters
        self.dim_state = params["dim_state"]
        self.actions = params["actions"]
        self.layer_size = params["layer_size"]
        self.gamma = params["gamma"]
        self.learning_rate = params["learning_rate"]
        self.batch_size = params["batch_size"]
        self.update_step = params["update_step"]
        # init Memory

        # pi(a|s;theta)
        assert len(self.layer_size) >= 1
        self.policy_net = tf.keras.models.Sequential([
            tf.keras.layers.Dense(self.layer_size[0], input_shape=(self.dim_state, ), activation='relu')
        ])
        for i in range(1, len(self.layer_size)):
            self.policy_net.add(Dense(self.layer_size[i], activation='relu'))
        self.policy_net.add(Dense(len(self.actions), activation='softmax'))
        self.optimizer = tf.keras.optimizers.Adam(lr=self.learning_rate)
        adam = tf.keras.optimizers.Adam(lr=1e-3)

        self.policy_net.summary()

    def train_one_epoch(self):
        pass

    def loss(self, minibatch):
        batch_states = np.array(minibatch[0])
        batch_rtaus = np.array(minibatch[1])
        act_probs = self.policy_net(batch_states)
        multinomial = np.zeros(act_probs.shape)
        for i in range(act_probs.shape[0]):
            multinomial[i] = np.random.multinomial(1,act_probs[i])
        return loss_object(y_true=target_q, y_pred=q_values)

    def grad(self, minibatch):
        with tf.GradientTape() as tape:
            loss_value = self.loss(minibatch)

        return loss_value, tape.gradient(loss_value, self.qnet_active.trainable_variables)

    def train(self, env):
        loss_value, grads = self.grad(states_memory)
        self.optimizer.apply_gradients(zip(grads, self.policy_net.trainable_variables))
        print("loss: {}".format(loss_value))
