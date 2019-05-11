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


class Memory:
    def __init__(self, memory_cap):
        self.memory_cap = memory_cap
        self.memory = []
    def store(self, experience):
        # pop a random experience if memory full
        if len(self.memory) >= self.memory_cap:
            self.memory.pop(random.randint(0, len(self.memory)-1))
        self.memory.append(experience)

    def sample_batch(self, batch_size):
        # Select batch
        if len(self.memory) < batch_size:
            batch = random.sample(self.memory, len(self.memory))
        else:
            batch = random.sample(self.memory, batch_size)

        return zip(*batch)

class DQNAgent:
    def __init__(self, params):
        # super(DQNAgent, self).__init__()
        # hyper-parameters
        self.dim_state = params["dim_state"]
        self.actions = params["actions"]
        self.layer_size = params["layer_size"]
        self.gamma = params["gamma"]
        self.learning_rate = params["learning_rate"]
        self.batch_size = params["batch_size"]
        self.memory_cap = params["memory_cap"]
        self.update_step = params["update_step"]
        self.model_path = params["model_path"]
        self.delta_dist = 0
        self.epsilon = 1
        # Q(s,a;theta)
        self.qnet_active = tf.keras.models.Sequential()
        for i in range(len(self.layer_size)):
            self.qnet_active.add(Dense(self.layer_size[i], activation="relu"))
        self.qnet_active.add(Dense(len(self.actions)))
        # self.qnet_active = tf.keras.models.Sequential([
        #     Dense(64, input_shape=(self.dim_state, ), activation='relu'),
        #     Dense(64, activation='relu'),
        #     Dense(len(self.actions))
        # ])
        # Q^(s,a;theta_)
        self.qnet_stable = tf.keras.models.clone_model(self.qnet_active)
        # self.qnet_stable = tf.keras.models.Sequential([
        #     Dense(64, input_shape=(self.dim_state, ), activation='relu'),
        #     Dense(64, activation='relu'),
        #     Dense(len(self.actions))
        # ])
        # optimizer
        self.optimizer = tf.keras.optimizers.Adam(lr=self.learning_rate)
        # init replay memory
        self.replay_memory = Memory(memory_cap=self.memory_cap)

    def epsilon_greedy(self, state):
        """
        If a random number(0,1) beats epsilon, return index of largest Q-value.
        Else, return a random index
        """
        if np.random.rand() > self.epsilon:
            return np.argmax(self.qnet_active.predict(state.reshape(1,-1)))
        else:
            print(bcolors.WARNING, "Take a random action!", bcolors.ENDC)
            return np.random.randint(len(self.actions))

    def epsilon_decay(self, episode_index, num_episodes):
        self.epsilon = np.clip(1-2*(episode_index/num_episodes), 5e-2, 1)

        return self.epsilon

    def loss(self, minibatch):
        (batch_states, batch_actions, batch_rewards, batch_done_flags, batch_next_states) = [np.array(minibatch[i]) for i in range(len(minibatch))]
        loss_object = tf.keras.losses.MeanSquaredError()
        q_values = tf.math.reduce_sum(tf.cast(self.qnet_active(batch_states), tf.float32) * tf.one_hot(batch_actions, len(self.actions)), axis=-1)
        target_q = batch_rewards + (1. - batch_done_flags) * self.gamma * tf.math.reduce_max(self.qnet_stable(batch_next_states), axis=-1)

        return loss_object(y_true=target_q, y_pred=q_values)

    def grad(self, minibatch):
        with tf.GradientTape() as tape:
            loss_value = self.loss(minibatch)

        return loss_value, tape.gradient(loss_value, self.qnet_active.trainable_variables)

    def save_model(self):
        self.qnet_active.save_weights(self.model_path)

    def train(self):
        # sample a minibatch
        minibatch = self.replay_memory.sample_batch(self.batch_size)
        # compute gradient for one epoch
        loss_value, grads = self.grad(minibatch)
        self.optimizer.apply_gradients(zip(grads, self.qnet_active.trainable_variables))
        loss_value = self.loss(minibatch)
        print("loss: {}".format(loss_value))
