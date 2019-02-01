#! /usr/bin/env python

"""
Model based control for turtlebot with vanilla policy gradient in crib environment
Navigate towards preset goal
Author: LinZHanK (linzhank@gmail.com)
Inspired by: https://github.com/openai/spinningup/blob/master/spinup/examples/pg_math/1_simple_pg.py
"""
from __future__ import absolute_import, division, print_function

import sys
sys.path.insert(0, "/home/linzhank/ros_ws/src/two_loggers/loggers_control/scripts/envs")

import numpy as np
import tensorflow as tf
import gym
import rospy
import random
import os
import time
import datetime
import matplotlib.pyplot as plt

from solo_escape_task_env import SoloEscapeEnv
from utils import bcolors


def mlp(x, sizes, activation=tf.tanh, output_activation=None):
  # Build a feedforward neural network.
  for size in sizes[:-1]:
    x = tf.layers.dense(x, units=size, activation=activation)
  return tf.layers.dense(x, units=sizes[-1], activation=output_activation)

def train(agent, dim_state=7, num_actions=3, hidden_sizes=[32], learning_rate=1e-3, num_episodes=50, num_steps=64, batch_size=10000):
  # make core of policy network
  states_ph = tf.placeholder(shape=(None, dim_state), dtype=tf.float32)
  logits = mlp(states_ph, sizes=hidden_sizes+[num_actions])
  # make action selection op (outputs int actions, sampled from policy)
  actions_id = tf.squeeze(tf.multinomial(logits=logits,num_samples=1), axis=1)
  # make loss function whose gradient, for the right data, is policy gradient
  rtaus_ph = tf.placeholder(shape=(None,), dtype=tf.float32)
  actid_ph = tf.placeholder(shape=(None,), dtype=tf.int32)
  action_masks = tf.one_hot(actid_ph, num_actions)
  log_probs = tf.reduce_sum(action_masks * tf.nn.log_softmax(logits), axis=1)
  loss = -tf.reduce_mean(rtaus_ph * log_probs)
  # make train op
  train_op = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)
  # start a session
  sess = tf.Session()
  sess.run(tf.global_variables_initializer())
  saver = tf.train.Saver()
  
  # for training policy
  def train_one_episode():
    # make some empty lists for logging.
    batch_states = [] # for observations
    batch_actions = [] # for actions
    batch_rtaus = [] # for R(tau) weighting in policy gradient
    batch_returns = [] # for measuring episode returns
    batch_lengths = [] # for measuring episode lengths
    # reset episode-specific variables
    state, _ = agent.env_reset()       # first obs comes from starting distribution
    done = False            # signal from environment that episode is over
    ep_rewards = []            # list for rewards accrued throughout ep
    dist_0 = np.linalg.norm(state[:2]-np.array([0,-6.02]))
    for st in range(num_steps):
      # save obs
      batch_states.append(state.copy())
      # act in the environment
      action_id = sess.run(actions_id, {states_ph: state.reshape(1,-1)})[0]
      if action_id == 0: # go straight
        action = np.array([.5, 0])
      elif action_id == 1: # turn left
        action = np.array([0, 1.0])
      else: # turn right
        action = np.array([0, -1.0])
      state, rew, done, info = agent.env_step(action)
      # add small reward if bot getting closer to exit
      dist = np.linalg.norm(state[:2]-np.array([0,-6.02]))
      # adjust reward based on relative distance to the exit
      if info["status"] == "escaped":
        rew *= num_steps
      elif info["status"] == "door":
        rew += 0.01/(state[1]+6.02)*num_steps
      else:
        rew += (dist_0-dist)
      # save action, reward
      batch_actions.append(action_id)
      ep_rewards.append(rew)
      # update bot's distance to exit
      dist_0 = dist
      rospy.loginfo("Episode: {}, Step: {} \naction: {}, state: {}, reward: {}, done: {}".format(
        ep,
        st,
        action,
        state,
        rew,
        done
      ))
      if done or len(batch_states) > batch_size:
        break
    # if episode is over, record info about episode
    ep_return, ep_length = sum(ep_rewards), len(ep_rewards)
    batch_returns.append(ep_return)
    batch_lengths.append(ep_length)
    # the weight for each logprob(a|s) is R(tau)
    batch_rtaus += [ep_return] * ep_length
    # if len(batch_states) > batch_size:
    #   break
    # take a single policy gradient update step
    batch_loss, _ = sess.run([loss, train_op],
                             feed_dict={
                               states_ph: np.array(batch_states),
                               actid_ph: np.array(batch_actions),
                               rtaus_ph: np.array(batch_rtaus)
                             })
    return batch_loss, batch_returns, batch_lengths
  
  # training loop
  sedimentary_returns = []
  for ep in range(num_episodes):
    batch_loss, batch_returns, batch_lengths = train_one_episode()
    print('epoch: %3d \t loss: %.3f \t return: %.3f \t ep_len: %.3f'%
                (ep, batch_loss, np.mean(batch_returns), np.mean(batch_lengths)))
    sedimentary_returns.append(batch_returns)
    save_path = saver.save(sess, "/home/linzhank/ros_ws/src/two_loggers/loggers_control/vpg_model/model.ckpt")
    rospy.loginfo("Model saved in path : {}".format(save_path))
    rospy.logerr("Success Count: {}".format(agent.success_count))
  plt.plot(sedimentary_returns)
  plt.show()
  
  

if __name__ == "__main__":
  rospy.init_node("solo_escape_vpg", anonymous=True, log_level=rospy.INFO)
  # make an instance from env class
  escaper = SoloEscapeEnv()
  # make hyper-parameters
  statespace_dim = 7 # x, y, x_dot, y_dot, cos_theta, sin_theta, theta_dot
  actionspace_dim = 3
  hidden_sizes = [64]
  num_episodes = 128
  num_steps = 1024
  learning_rate = 1e-3
  batch_size = 5000
  # make core of policy network
  train(agent=escaper, dim_state = statespace_dim, num_actions=actionspace_dim,
        learning_rate=learning_rate, num_episodes=num_episodes,
        num_steps=num_steps, batch_size=batch_size)
