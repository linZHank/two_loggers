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

    def sample_action(self, state):
        return np.argmax(self.policy_net.predict(state.reshape(1,-1)))

    def loss(self, batch_states, batch_acts, batch_rtaus):
        acts_prob = self.policy_net(batch_states)
        acts_onehot = tf.one_hot(batch_acts)
        log_probs = tf.reduce_sum(acts_onehot * tf.math.log(acts_prob), axis=1)
        return -tf.reduce_mean(batch_rtaus * log_probs)

    def grad(self, batch_states, batch_acts, batch_rtaus):
        with tf.GradientTape() as tape:
            loss_value = self.loss(batch_states, batch_rtaus)

        return loss_value, tape.gradient(loss_value, self.policy_net.trainable_variables)

    def train(self, batch_states, batch_acts, batch_rtaus):
        loss_value, grads = self.grad(batch_states, batch_acts, batch_rtaus)
        self.optimizer.apply_gradients(zip(grads, self.policy_net.trainable_variables))
        print("loss: {}".format(loss_value))

    def save_model(self):
        self.policy_net.summary()
        # create model saving directory if not exist
        model_dir = os.path.dirname(self.model_path)
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        self.policy_net.save(self.model_path)
        print("policy_net model save at {}".format(self.model_path))
