#! /usr/bin/env python

"""
Model based control for turtlebot with vanilla policy gradient in crib environment
Navigate towards preset goal
Author: LinZHanK (linzhank@gmail.com)
Inspired by: https://github.com/openai/spinningup/blob/master/spinup/examples/pg_math/1_simple_pg.py
"""
from __future__ import absolute_import, division, print_function

import numpy as np
import tensorflow as tf
import gym
import rospy
import random
import os
import time
import datetime
import matplotlib.pyplot as plt

from utils import bcolors, obs_to_state


def mlp(x, sizes, activation=tf.tanh, output_activation=None):
  # Build a feedforward neural network.
  for size in sizes[:-1]:
    x = tf.layers.dense(x, units=size, activation=activation)
  return tf.layers.dense(x, units=sizes[-1], activation=output_activation)

def train_one_episode(batch_size):
  # make some empty lists for one episode.
  batch_states = [] # for states
  batch_actions = [] # for actions
  batch_rtaus = [] # for weights R(tau)
  batch_returns = [] # for measuring episode returns
  batch_lengths = [] # for measuring episode lenghts
  # reset episode-specific variables
  obs = env_reset() # first observation comes from starting distribution (NEED WRITE IN: solo_escape_utils)
  state = obs_to_state(obs) # convert observation into state (NEED WRITE IN: solo_escape_utils)
  done = False # signal from environment that episode is over
  episode_rewards = [] # list for rewards accrued throughout the episode
  # collect experience by acting in the environment with current policy
  while True:
    batch_states.append(state)
    # act in the environment
    action_id = sess.run(action_id, feed_dict={state_ph: state.reshape(1,-1)})[0]
    obs, rew, done, info = env_step(action_id) # (NEED WRITE IN: solo_escape_utils)
    state = obs_to_state(obs)
    # save action, reward
    batch_acts.append(action_id)
    episode_rewards.append(rew)
    if done:
      # if episode is over, record info about episode
      episode_return, episode_length = sum(episode_rewards), len(episode_rewards)
      batch_returns.append(episode_return)
      batch_lengths.append(episode_length)
      # the weight for each logprob(a|s) is R(tau)
      batch_rtaus += [episode_return] * ep_length
      # reset episode-specific variables
      obs, done, episode_rewards = env.reset(), False, []
      # end experience loop if we have enough of it
      if len(batch_states) > batch_size:
        break
  # take a single policy gradient update step
  batch_loss, _ = sess.run([loss, train_op],
                           feed_dict={
                            state_ph: np.array(batch_states),
                            action_ph: np.array(batch_actions),
                            rtaus_ph: np.array(batch_rtaus)
                           })
  return batch_loss, batch_returns, batch_lengths
  
if __name__ == "__main__":
  rospy.init_node("crib_nav_vpg", anonymous=True, log_level=rospy.WARN)
  # make environment, check spaces
  statespace_dim = 7 # x, y, x_dot, y_dot, cos_theta, sin_theta, theta_dot
  actionspace_dim = 2
  hidden_sizes = [32]
  num_epochs = 50
  lr = 1e-2
  batch_size = 5000
  # make core of policy network
  state_ph = tf.placeholder(shape=(None, statespace_dim), dtype=tf.float32)
  logits = mlp(state_ph, sizes=hidden_sizes+[actionspace_dim])
  # make action selection op (outputs int actions, sampled from policy)
  action_id = tf.squeeze(tf.multinomial(logits=logits,num_samples=1), axis=1)
  # make loss function whose gradient, for the right data, is policy gradient
  rtaus_ph = tf.placeholder(shape=(None,), dtype=tf.float32)
  action_ph = tf.placeholder(shape=(None,), dtype=tf.int32)
  action_masks = tf.one_hot(action_ph, actionspace_dim)
  log_probs = tf.reduce_sum(action_masks * tf.nn.log_softmax(logits), axis=1)
  loss = -tf.reduce_mean(return_ph * log_probs)
  # make train op
  train_op = tf.train.AdamOptimizer(learning_rate=lr).minimize(loss)
  # start a tf session
  sess = tf.InteractiveSession()
  sess.run(tf.global_variables_initializer())
  # training loop
  for i in range(num_epochs):
    batch_loss, batch_returns, batch_lengths = train_one_epoch(batch_size)
    print(
      "epoch: {:3d}\t loss: {:.3f}\t return: {:.3f}\t ep_len: {:.3f}".format(
        i, batch_loss, np.mean(batch_returns), np.mean(batch_lengths)
      )
    )
  
