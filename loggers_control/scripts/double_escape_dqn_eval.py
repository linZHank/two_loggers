"""
An implementation of Deep Q-network (DQN) for solo_escape_task
DQN is a Model free, off policy, reinforcement learning algorithm (https://deepmind.com/research/dqn/)
Author: LinZHanK (linzhank@gmail.com)
"""
from __future__ import absolute_import, division, print_function

import sys
import os
from datetime import datetime
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense
import rospy

from envs.double_escape_task_env import DoubleEscapeEnv
from utils import gen_utils, double_utils, tf_utils
from utils.gen_utils import bcolors
from agents.dqn import DQNAgent

if __name__ == "__main__":
    # Main really starts here
    # start_time = time.time()
    rospy.init_node("double_escape_dqn_test", anonymous=True, log_level=rospy.INFO)
    # make an instance from env class
    env = DoubleEscapeEnv()
    env.reset()
    # hyper-parameters
    agent0_params = {}
    agent1_params = {}
    # agent_0 parameters
    agent0_params["dim_state"] = len(double_utils.obs_to_state(env.observation, "all"))
    agent0_params["actions"] = np.array([np.array([.5, -1]), np.array([.5, 1]), np.array([-.5, -1]), np.array([-.5, 1]), np.array([0, 0])])
    agent0_params["layer_size"] = [256,128]
    agent0_params["gamma"] = 0.99
    agent0_params["learning_rate"] = 3e-4
    agent0_params["batch_size"] = 2000
    agent0_params["memory_cap"] = 500000
    agent0_params["update_step"] = 10000
    agent0_params["model_path"] = os.path.dirname(sys.path[0])+"/saved_models/double_escape/dqn_model/2019-05-17-15-45/agent0/model.h5"
    # agent_1 parameters
    agent1_params["dim_state"] = len(double_utils.obs_to_state(env.observation, "all"))
    agent1_params["actions"] = np.array([np.array([.5, -1]), np.array([.5, 1]), np.array([-.5, -1]), np.array([-.5, 1]), np.array([0, 0])])
    agent1_params["layer_size"] = [256,128]
    agent1_params["epsilon"] = 1
    agent1_params["gamma"] = 0.99
    agent1_params["learning_rate"] = 3e-4
    agent1_params["batch_size"] = 2000
    agent1_params["memory_cap"] = 500000
    agent1_params["update_step"] = 10000
    agent1_params["model_path"] = os.path.dirname(sys.path[0])+"/saved_models/double_escape/dqn_model/2019-05-17-15-45/agent1/model.h5"
    # evaluation params
    num_episodes = 10
    num_steps = 200
    # load model for robot_0
    model_0 = tf.keras.models.Sequential([
        Dense(agent0_params["layer_size"][0], activation='relu', input_shape=(agent0_params["dim_state"],))
    ])
    if len(agent0_params["layer_size"]) >= 2:
        for i in range(1, len(agent0_params["layer_size"])):
            model_0.add(Dense(agent0_params["layer_size"][i], activation="relu"))
    model_0.add(Dense(len(agent0_params["actions"])))
    model_0 = tf.keras.models.load_model(agent0_params["model_path"])
    # load model for robot_1
    model_1 = tf.keras.models.Sequential([
        Dense(agent1_params["layer_size"][0], activation='relu', input_shape=(agent1_params["dim_state"],))
        ])
    if len(agent1_params["layer_size"]) >= 2:
        for i in range(1, len(agent1_params["layer_size"])):
            model_1.add(Dense(agent1_params["layer_size"][i], activation="relu"))
    model_1.add(Dense(len(agent1_params["actions"])))
    model_1 = tf.keras.models.load_model(agent1_params["model_path"])
    # start evaluating
    for ep in range(num_episodes):
        obs, _ = env.reset()
        done = False
        state_agt0 = double_utils.obs_to_state(obs, "all")
        state_agt1 = double_utils.obs_to_state(obs, "all")
        for st in range(num_steps):
            agent0_acti = np.argmax(model_0.predict(state_agt0.reshape(1,-1)))
            agent0_action = agent0_params["actions"][agent0_acti]
            agent1_acti = np.argmax(model_1.predict(state_agt1.reshape(1,-1)))
            agent1_action = agent1_params["actions"][agent1_acti]
            obs, rew, done, info = env.step(agent0_action, agent1_action)
            next_state_agt0 = double_utils.obs_to_state(obs, "all")
            next_state_agt1 = double_utils.obs_to_state(obs, "all")
            state_agt0 = next_state_agt0
            state_agt1 = next_state_agt1
            # logging
            print(
                bcolors.OKGREEN,
                "Episode: {}, Step: {} \naction0: {}->{}, action0: {}->{}, agent_0 state: {}, agent_1 state: {}, reward: {}, status: {}".format(
                    ep,
                    st,
                    agent0_acti,
                    agent0_action,
                    agent1_acti,
                    agent1_action,
                    next_state_agt0,
                    next_state_agt1,
                    rew,
                    info
                ),
                bcolors.ENDC
            )
            if done:
                break
