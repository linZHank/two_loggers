"""
Evaluate DQN models for double_escape_task with specified initial config
Author: LinZHanK (linzhank@gmail.com)
"""
from __future__ import absolute_import, division, print_function

import sys
import os
import pickle
from datetime import datetime
import numpy as np
from numpy import pi
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras.layers import Dense
import rospy

from envs.double_escape_task_env import DoubleEscapeEnv
from utils import data_utils, double_utils
from utils.data_utils import bcolors
from agents.dqn import DQNAgent

if __name__ == '__main__':
    # specify init config
    init_config = [3.25,-4,-pi,-pi/2,-pi/2]
    # instantiate env
    env = DoubleEscapeEnv()
    obs, info = env.reset(init_config)
    state_0 = double_utils.obs_to_state(obs, "all")
    state_1 = double_utils.obs_to_state(obs, "all")
    # load agent models
    model_dir = os.path.dirname(sys.path[0])+"/saved_models/double_escape/dqn/2019-12-20-12-10/"
    with open(os.path.join(model_dir,"agent_0/agent_parameters.pkl"), "rb") as f:
        agent_params_0 = pickle.load(f)
    with open(os.path.join(model_dir,"agent_1/agent_parameters.pkl"), "rb") as f:
        agent_params_1 = pickle.load(f)
    # instantiate agents
    agent_0 = DQNAgent(agent_params_0)
    agent_0.load_model(os.path.join(model_dir, "agent_0/model.h5"))
    agent_1 = DQNAgent(agent_params_1)
    agent_1.load_model(os.path.join(model_dir, "agent_1/model.h5"))
    # init eval parameters
    num_steps = 200
    eval_params = {'wall_bonus': False,'door_bonus':False,'time_bonus':False,'success_bonus':False,'num_steps':num_steps}
    traj_0, traj_1 = [], []
    # start evaluation
    for st in range(num_steps):
        traj_0.append((state_0[7:9]))
        traj_1.append((state_1[14:16]))
        action_index_0 = np.argmax(agent_0.qnet_active.predict(state_0.reshape(1,-1)))
        action_0 = agent_params_0["actions"][action_index_0]
        action_index_1 = np.argmax(agent_1.qnet_active.predict(state_1.reshape(1,-1)))
        action_1 = agent_params_1["actions"][action_index_1]
        obs, rew, done, info = env.step(action_0, action_1)
        rew, done = double_utils.adjustplt.xlim(-5,5)
_reward(eval_params, env)
        next_state_0 = double_utils.obs_to_state(obs, "all")
        next_state_1 = double_utils.obs_to_state(obs, "all")
        state_0 = next_state_0
        state_1 = next_state_1
        # logging
        rospy.loginfo(
            "Step: {} \naction0: {}->{}, action0: {}->{}, agent_0 state: {}, agent_1 state: {}, reward: {}, status: {} \nsuccess_count: {}".format(
                st,
                action_index_0,
                action_0,
                action_index_1,
                action_1,
                next_state_0,
                next_state_1,
                rew,
                info["status"],
                env.success_count
            ),
        )
        if done:
            break
    env.reset([0,0,0,0,0])

    # save and plot trajectories
    traj_0 = -np.asarray(traj_0)
    traj_1 = -np.asarray(traj_1)
    date_time = datetime.now().strftime("%Y-%m-%d-%H-%M")
    traj_dir = os.path.join(model_dir, 'validate_cases', date_time)
    if not os.path.exists(traj_dir):
        os.makedirs(traj_dir)
    np.save(os.path.join(traj_dir, 'trajectory_0.npy'), traj_0)
    np.save(os.path.join(traj_dir, 'trajectory_1.npy'), traj_1)
    # compute travel distance
    diff_0 = np.diff(traj_0, axis=0)
    d_0 = np.hypot(diff_0[:,0], diff_0[:,1])
    diff_1 = np.diff(traj_1, axis=0)
    d_1 = np.hypot(diff_1[:,0], diff_1[:,1])
    total_dist = sum(d_0) + sum(d_1)
    with open(os.path.join(traj_dir,'total_distance.txt'), 'w') as f:
        f.write('{}'.format(total_dist))
    # plot
    left_wall = plt.Rectangle((-5,5),4,1,facecolor="grey")
    right_wall = plt.Rectangle((1,5),4,1,facecolor="grey")
    fig, ax = plt.subplots()
    ax.add_patch(left_wall)
    ax.add_patch(right_wall)
    ax.plot(traj_0[:,0], traj_0[:,1], 'r.')
    ax.plot(traj_1[:,0], traj_1[:,1], 'b^')
    ax.set_xlim(-5,5)
    ax.set_ylim(0,9)
    ax.set(xlabel='X (m)', ylabel='Y (m)')
    plt.savefig(os.path.join(traj_dir,'traj.png'))
    # plt.show()
