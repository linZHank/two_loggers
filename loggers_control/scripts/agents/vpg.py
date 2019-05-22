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
            Dense(self.layer_size[0], input_shape=(self.dim_state, ), activation='relu')
        ])
        for i in range(1, len(self.layer_size)):
            self.policy_net.add(Dense(self.layer_size[i], activation='relu'))
        self.policy_net.add(Dense(len(self.actions), activation='softmax'))
        self.optimizer = tf.keras.optimizers.Adam(lr=self.learning_rate)
        adam = tf.keras.optimizers.Adam(lr=1e-3)

        self.policy_net.summary()

    def loss(self, batch_states, batch_rtaus):
        acts_prob = self.policy_net(batch_states)
        acts_onehot = np.zeros(acts_prob.shape)
        for i in range(act_probs.shape[0]):
            acts_onehot[i] = np.random.multinomial(1,acts_prob[i])
        log_probs = tf.reduce_sum(acts_onehot * tf.math.log(acts_prob), axis=1)
        return -tf.reduce_mean(batch_rtaus * log_probs)#loss_object(y_true=target_q, y_pred=q_values)

    def grad(self, batch_states, batch_rtaus):
        with tf.GradientTape() as tape:
            loss_value = self.loss(batch_states, batch_rtaus)

        return loss_value, tape.gradient(loss_value, self.policy_net.trainable_variables)

    def train(self, env):
        loss_value, grads = self.grad(batch_states, batch_rtaus)
        self.optimizer.apply_gradients(zip(grads, self.policy_net.trainable_variables))
        print("loss: {}".format(loss_value))
