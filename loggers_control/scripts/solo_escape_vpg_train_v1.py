#! /usr/bin/env python

"""
Vanilla Policy Gradient control single logger escape the walled cell.
Adjust reward based on different situations
Author: LinZHanK (linzhank@gmail.com)
Inspired by: https://github.com/openai/spinningup/blob/master/spinup/examples/pg_math/1_simple_pg.py
"""
from __future__ import absolute_import, division, print_function

import sys
sys.path.insert(0, "/home/linzhank/ros_ws/src/two_loggers/loggers_control/scripts/envs")
import argparse
import numpy as np
import tensorflow as tf
import rospy
import random
import os
import time
from datetime import datetime
import pickle
import csv
import matplotlib.pyplot as plt

from solo_escape_task_env import SoloEscapeEnv
from utils import bcolors


def mlp(x, sizes, activation=tf.tanh, output_activation=None):
  # Build a feedforward neural network.
  for size in sizes[:-1]:
    x = tf.layers.dense(x, units=size, activation=activation)
  return tf.layers.dense(x, units=sizes[-1], activation=output_activation)

# bonus functions
def bonusWallDividedNumsteps(bw,ns): return bw/ns # bonus_time
def weightedD0(w,d0): return w*d0 # bonus_distance
def d0MinusD(d0,d): return d0-d # bonus approach
def zero(x,y): return 0

def train(agent, model_path,
          dim_state=7, num_actions=3,
          hidden_sizes=[64], learning_rate=1e-3,
          num_episodes=400, num_steps=1000,
          bonus_wall=-.01, bonus_door=0.1,
          bonus_time_func=zero, bonus_distance_func=zero, bonus_approach_func=zero):
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
    dist_0 = np.linalg.norm(state[:2]-np.array([0,-6.2]))
    bonus_distance = dist_0/11
    for st in range(num_steps):
      # save obs
      batch_states.append(state.copy())
      # act in the environment
      action_id = sess.run(actions_id, {states_ph: state.reshape(1,-1)})[0]
      if action_id == 0: # forward left
        action = np.array([.5, 1.])
      elif action_id == 1: # forward right
        action = np.array([.5, -1.])
      else: # forward
        action = np.array([.5, 0.])
      state, rew, done, info = agent.env_step(action)
      # compute current distance to exit
      dist = np.linalg.norm(state[:2]-np.array([0,-6.2]))
      # consider bonus terms
      bonus_time = bonus_time_func(bonus_wall, num_steps)
      bonus_distance = bonus_distance_func(11, dist_0)
      bonus_approach = bonus_approach_func(dist_0, dist)
      # adjust reward based on relative distance to the exit
      if info["status"] == "escaped":
        bonus = bonus_distance
      elif info["status"] == "door":
        bonus = bonus_time+bonus_door
      elif info["status"] == "trapped":
        bonus = bonus_time+bonus_approach
      else: # hit wall
        bonus = bonus_time+bonus_wall
      rew += bonus
      # save action, reward
      batch_actions.append(action_id)
      ep_rewards.append(rew)
      # update robot's distance to exit
      dist_0 = dist
      # log this step
      rospy.loginfo("Episode: {}, Step: {} \naction: {}, state: {}, reward: {}, done: {}".format(
        ep,
        st,
        action,
        state,
        rew,
        done
      ))
      if done:
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
  deposit_returns = []
  for ep in range(num_episodes):
    batch_loss, batch_returns, batch_lengths = train_one_episode()
    print('epoch: %3d \t loss: %.3f \t return: %.3f \t ep_len: %.3f'%
                (ep, batch_loss, np.mean(batch_returns), np.mean(batch_lengths)))
    deposit_returns.append(batch_returns)
    save_path = saver.save(sess, model_path)
    rospy.loginfo("Model saved in path : {}".format(save_path))
    rospy.logerr("Success Count: {}".format(agent.success_count))
  # plot returns
  fig, ax = plt.subplots()
  ax.plot(np.arange(len(deposit_returns)), deposit_returns)
  ax.set(xlabel="Episode", ylabel='Episodic Return')
  ax.grid()
  figure_fname = os.path.join(os.path.dirname(model_path),"returns.png")
  plt.savefig(figure_fname)
  plt.close(fig)
  

if __name__ == "__main__":
  # make arg parser
  parser = argparse.ArgumentParser()
  parser.add_argument("--model_path", type=str,
                      default="/home/linzhank/ros_ws/src/two_loggers/loggers_control/vpg_model-"+datetime.now().strftime("%Y-%m-%d-%H-%M")+"/model.ckpt")
  parser.add_argument("--hidden_sizes", type=list, default = [64])
  parser.add_argument("--learning_rate", type=float, default = 1e-3)
  parser.add_argument("--num_episodes", type=int, default = 800)
  parser.add_argument("--num_steps", type=int, default = 1000)
  parser.add_argument("--bonus_wall", type=float, default = -.01)
  parser.add_argument("--bonus_door", type=float, default = .1)
  parser.add_argument("--bonus_time", type=bool, default = True)
  parser.add_argument("--bonus_distance", type=bool, default = True)
  parser.add_argument("--bonus_approach", type=bool, default = True)
  args = parser.parse_args()
  # time
  start_time = time.time()
  rospy.init_node("solo_escape_vpg", anonymous=True, log_level=rospy.INFO)
  # make an instance from env class
  escaper = SoloEscapeEnv()
  statespace_dim = 7 # x, y, x_dot, y_dot, cos_theta, sin_theta, theta_dot
  actionspace_dim = 3
  # store hyper-parameters
  bonus_time_func = zero
  bonus_distance_func = zero
  bonus_approach_func = zero
  if args.bonus_time:
    bonus_time_func = bonusWallDividedNumsteps
  if args.bonus_distance:
    bonus_distance_func = weightedD0
  if args.bonus_approach:
    bonus_approach_func = d0MinusD
  # make core of policy network
  train(agent=escaper, model_path=args.model_path,
        dim_state = statespace_dim, num_actions=actionspace_dim,
        hidden_sizes=args.hidden_sizes, learning_rate=args.learning_rate,
        num_episodes=args.num_episodes, num_steps=args.num_steps,
        bonus_wall=args.bonus_wall, bonus_door=args.bonus_door,
        bonus_time_func=bonus_time_func, bonus_distance_func=bonus_distance_func, bonus_approach_func=bonus_approach_func)
  rospy.logdebug("success: {}".format(escaper.success_count))

  hyp_params = {
    "statespace_dim": statespace_dim,
    "actionspace_dim": actionspace_dim,
    "hidden_sizes": args.hidden_sizes,
    "learning_rate": args.learning_rate,
    "num_episodes": args.num_episodes,
    "num_steps": args.num_steps,
  }               
  # time
  end_time = time.time()
  training_time = end_time - start_time
  # store results
  train_info = hyp_params
  train_info["success_count"] = escaper.success_count
  train_info["training_time"] = training_time
  train_info["bonus_wall"] = args.bonus_wall
  train_info["bonus_door"] = args.bonus_door
  train_info["bonus_time"] = bonus_time_func.func_name
  train_info["bonus_distance"] = bonus_distance_func.func_name
  train_info["bonus_approach"] = bonus_approach_func.func_name

  # save hyper-parameters
  file_name = "hyper_parameters.pkl"
  file_dir = os.path.dirname(args.model_path)
  file_path = os.path.join(file_dir,file_name)
  with open(file_path, "wb") as hfile:
    pickle.dump(hyp_params, hfile, pickle.HIGHEST_PROTOCOL)
  # save results
  file_name = "train_information.csv"
  file_dir = os.path.dirname(args.model_path)
  file_path = os.path.join(file_dir,file_name)
  with open(file_path, "w") as rfile:
    for key in train_info.keys():
      rfile.write("{},{}\n".format(key,train_info[key]))
  
