from __future__ import absolute_import, division, print_function

import sys
import os
import numpy as np
import random
import pickle
import tensorflow as tf
import rospy

from utils import data_utils
from utils.data_utils import bcolors
from tensorflow.keras.layers import Dense
from tensorflow.keras import Model


class Memory:
    """
    This class defines replay buffer
    """
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
        # hyper-parameters
        self.dim_state = params["dim_state"]
        self.actions = params["actions"]
        self.layer_sizes = params["layer_sizes"]
        if type(params["layer_sizes"]) == int:
            self.layer_sizes = [params["layer_sizes"]]
        self.gamma = params["gamma"]
        self.learning_rate = params["learning_rate"]
        self.batch_size = params["batch_size"]
        self.memory_cap = params["memory_cap"]
        self.update_step = params["update_step"]
        self.delta_dist = 0
        self.epsilon = 1
        self.loss_value = np.inf
        # Q(s,a;theta)
        assert len(self.layer_sizes) >= 1
        self.qnet_active = tf.keras.models.Sequential([
            Dense(self.layer_sizes[0], activation='relu', input_shape=(self.dim_state,))
        ])
        for i in range(1,len(self.layer_sizes)):
            self.qnet_active.add(Dense(self.layer_sizes[i], activation="relu"))
        self.qnet_active.add(Dense(len(self.actions)))
        # Q^(s,a;theta_)
        self.qnet_stable = tf.keras.models.clone_model(self.qnet_active)
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

    def epsilon_decay(self, num, den, lower=0.05, upper=1):
        """
        Construct an epsilon decay function
            Args:
                num: numerator
                den: denominator
                upper: upper bound of epsilon
                lower: lower bound of epsilon
            Returns:
                self.epsilon: epsilon at current episode
        """
        self.epsilon = np.clip(1-num/den, lower, upper)

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

    def train(self):
        # sample a minibatch
        minibatch = self.replay_memory.sample_batch(self.batch_size)
        # compute gradient for one epoch
        self.loss_value, grads = self.grad(minibatch)
        self.optimizer.apply_gradients(zip(grads, self.qnet_active.trainable_variables))
        # loss_value = self.loss(minibatch)
        print("loss: {}".format(self.loss_value))

    def save_model(self, model_path):
        self.qnet_active.summary()
        # create model saving directory if not exist
        model_dir = os.path.dirname(model_path)
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        # save model
        self.qnet_active.save(model_path)
        print("policy_net model saved at {}".format(model_path))

    def load_model(self, model_path):
        self.qnet_active = tf.keras.models.load_model(model_path)
        mem_path = os.path.join(os.path.dirname(model_path),'memory.pkl')
        with open(mem_path, 'rb') as f:
            self.replay_memory = pickle.load(f)
            print("Replay Buffer Loaded")
        self.qnet_active.summary()

    def save_memory(self, model_path):
        model_dir = os.path.dirname(model_path)
        # save transition buffer memory
        data_utils.save_pkl(content=self.replay_memory, fdir=model_dir, fname='memory.pkl')
        print("transitions memory saved at {}".format(model_dir))
