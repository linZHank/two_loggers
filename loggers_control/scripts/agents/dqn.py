from __future__ import absolute_import, division, print_function

import sys
import os
import numpy as np
import random
import pickle
import tensorflow as tf
import rospy
import logging

from utils import data_utils
from utils.data_utils import bcolors
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers import Dense
from tensorflow.keras import Model


def create_agent_params(dim_state, actions, layer_sizes, gamma, learning_rate, batch_size, memory_cap, update_step, decay_period, final_eps):
    """
    Create agent parameters dict based on args
    """
    agent_params = {}
    agent_params["dim_state"] = dim_state
    agent_params["actions"] = actions
    agent_params["layer_sizes"] = layer_sizes
    agent_params["gamma"] = gamma
    agent_params["learning_rate"] = learning_rate
    agent_params["batch_size"] = batch_size
    agent_params["memory_cap"] = memory_cap
    agent_params["update_step"] = update_step
    agent_params["decay_period"] = decay_period
    agent_params['final_eps'] = final_eps

    return agent_params

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
        logging.debug("experience: {} stored to memory".format(experience))

    def sample_batch(self, batch_size):
        # Select batch
        if len(self.memory) < batch_size:
            batch = random.sample(self.memory, len(self.memory))
        else:
            batch = random.sample(self.memory, batch_size)
        logging.debug("A batch of memories are sampled with size: {}".format(batch_size))

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
        self.update_counter = 0
        self.epsilon = 1
        self.loss_value = np.inf
        # Q(s,a;theta)
        assert len(self.layer_sizes) >= 1
        inputs = tf.keras.Input(shape=(self.dim_state,), name='state')
        x = layers.Dense(self.layer_sizes[0], activation='relu')(inputs)
        for i in range(1,len(self.layer_sizes)):
            x = layers.Dense(self.layer_sizes[i], activation='relu')(x)
        outputs = layers.Dense(len(self.actions))(x)
        self.qnet_active = Model(inputs=inputs, outputs=outputs, name='qnet_model')
        # clone active Q-net to create stable Q-net
        self.qnet_stable = tf.keras.models.clone_model(self.qnet_active)
        # optimizer
        self.optimizer = tf.keras.optimizers.Adam(lr=self.learning_rate)
        # loss function
        self.loss_fn = tf.keras.losses.MeanSquaredError()
        # metrics
        self.acc_metric = keras.metrics.Accuracy()
        self.mse_metric = keras.metrics.MeanSquaredError()
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

    def linearly_decaying_epsilon(self, decay_period, episode, warmup_episodes=100, final_eps=0.005):
        """
        Returns the current epsilon for the agent's epsilon-greedy policy. This follows the Nature DQN schedule of a linearly decaying epsilon (Mnih et al., 2015). The schedule is as follows:
            Begin at 1. until warmup_steps steps have been taken; then Linearly decay epsilon from 1. to final_eps in decay_period steps; and then Use epsilon from there on.
        Args:
            decay_period: float, the period over which epsilon is decayed.
            episode: int, the number of training steps completed so far.
            warmup_episodes: int, the number of steps taken before epsilon is decayed.
            final_eps: float, the final value to which to decay the epsilon parameter.
        Returns:
            A float, the current epsilon value computed according to the schedule.
        """
        episodes_left = decay_period + warmup_episodes - episode
        bonus = (1.0 - final_eps) * episodes_left / decay_period
        bonus = np.clip(bonus, 0., 1.-final_eps)
        self.epsilon = final_eps + bonus

        return self.epsilon

    def train(self):
        # sample a minibatch from
        minibatch = self.replay_memory.sample_batch(self.batch_size)
        (batch_states, batch_actions, batch_rewards, batch_done_flags, batch_next_states) = [np.array(minibatch[i]) for i in range(len(minibatch))]
        # open a GradientTape to record the operations run during the forward pass
        with tf.GradientTape() as tape:
            # run forward pass
            pred_q = tf.math.reduce_sum(tf.cast(self.qnet_active(batch_states), tf.float32) * tf.one_hot(batch_actions, len(self.actions)), axis=-1)
            target_q = batch_rewards + (1. - batch_done_flags) * self.gamma * tf.math.reduce_max(self.qnet_stable(batch_next_states), axis=-1)
            # compute loss value
            self.loss_value = self.loss_fn(y_true=target_q, y_pred=pred_q)
        # use the gradient tape to automatically retrieve the gradients of the trainable variables with respect to the loss.
        grads = tape.gradient(self.loss_value, self.qnet_active.trainable_weights)
        # run one step of gradient descent
        self.optimizer.apply_gradients(zip(grads, self.qnet_active.trainable_weights))
        # update metrics
        self.acc_metric(target_q, pred_q)
        self.mse_metric(target_q, pred_q)
        # display metrics
        train_acc = self.acc_metric.result()
        train_mse = self.mse_metric.result()
        print("Training  accuracy: {}, \tTraining mse: {}".format(train_acc, train_mse))
        # reset training metrics
        self.acc_metric.reset_states()
        self.mse_metric.reset_states()

    def save_model(self, model_path):
        self.qnet_active.summary()
        # create model saving directory if not exist
        model_dir = os.path.dirname(model_path)
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        # save model
        self.qnet_active.save(model_path)
        logging.info("policy_net model saved at {}".format(model_path))

    def load_model(self, model_path):
        self.qnet_active = tf.keras.models.load_model(model_path)
        mem_path = os.path.join(os.path.dirname(model_path),'memory.pkl')
        with open(mem_path, 'rb') as f:
            self.replay_memory = pickle.load(f)
            logging.debug("Replay Buffer Loaded")
        self.qnet_active.summary()

    def save_memory(self, model_path):
        model_dir = os.path.dirname(model_path)
        # save transition buffer memory
        data_utils.save_pkl(content=self.replay_memory, fdir=model_dir, fname='memory.pkl')
        logging.info("transitions memory saved at {}".format(model_dir))
