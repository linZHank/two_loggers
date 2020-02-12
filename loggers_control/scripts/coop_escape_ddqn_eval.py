"""
Evaluation of learned model for coop_escape_task
Author: LinZHanK (linzhank@gmail.com)
"""
from __future__ import absolute_import, division, print_function

import sys
import os
import pickle
from datetime import datetime
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense
import rospy

from envs.coop_escape_task_env import CoopEscapeEnv
from utils import data_utils, coop_utils
from utils.data_utils import bcolors
from agents.ddqn import DoubleDQNAgent

if __name__ == "__main__":
    # instantiate env
    env = CoopEscapeEnv()
    env.reset()
    # load agent models
    model_dir = os.path.dirname(sys.path[0])+"/saved_models/coop_escape/ddqn/2020-01-19-13-12/"
    with open(os.path.join(model_dir,"agent_0/agent_parameters.pkl"), "rb") as f:
        agent_params_0 = pickle.load(f)
    with open(os.path.join(model_dir,"agent_1/agent_parameters.pkl"), "rb") as f:
        agent_params_1 = pickle.load(f)
    # # load train parameters
    # with open(model_dir+"/train_parameters.pkl", 'rb') as f:
    #     train_params = pickle.load(f)
    # instantiate agents
    agent_0 = DoubleDQNAgent(agent_params_0)
    agent_0.load_model(os.path.join(model_dir, "agent_0"))
    agent_1 = DoubleDQNAgent(agent_params_1)
    agent_1.load_model(os.path.join(model_dir, "agent_1"))

    # evaluation params
    num_episodes = 100
    num_steps = 160
    ep = 0
    eval_params = {'wall_bonus': False,'door_bonus':False,'time_bonus':False,'success_bonus':False,'num_steps':num_steps}
    # start evaluating
    while ep <= num_episodes:
        obs, info = env.reset()
        done = False
        if info["status"][0] == "blew" or info["status"][1] == "blew":
            rospy.logerr("Model blew up, skip this episode")
            obs, info = env.reset()
            continue
        state_0 = coop_utils.obs_to_state(obs, "logger_0")
        state_1 = coop_utils.obs_to_state(obs, "logger_1")
        for st in range(num_steps):
            action_index_0 = np.argmax(agent_0.qnet_active.predict(state_0.reshape(1,-1)))
            action_0 = agent_params_0["actions"][action_index_0]
            action_index_1 = np.argmax(agent_1.qnet_active.predict(state_1.reshape(1,-1)))
            action_1 = agent_params_1["actions"][action_index_1]
            obs, rew, done, info = env.step(action_0, action_1)
            rew, done = coop_utils.adjust_reward(eval_params, env)

            next_state_0 = coop_utils.obs_to_state(obs, "logger_0")
            next_state_1 = coop_utils.obs_to_state(obs, "logger_1")
            state_0 = next_state_0
            state_1 = next_state_1
            # logging
            rospy.loginfo(
                "Episode: {}, Step: {} \naction0: {}->{}, action0: {}->{}, reward: {}, status: {} \nsuccess_count: {}".format(
                    ep,
                    st,
                    action_index_0,
                    action_0,
                    action_index_1,
                    action_1,
                    rew,
                    info["status"],
                    env.success_count
                ),
            )
            if done:
                ep += 1
                break

    print("Loggers succeeded escaping {} out of {}".format(env.success_count, num_episodes))
    env.reset()
