#! /usr/bin/env python
"""
Training two logger robots escaping a cell with Deep Q-network (DQN)
DQN is a model free, off policy, reinforcement learning algorithm (https://deepmind.com/research/dqn/)
Author: LinZHanK (linzhank@gmail.com)

Train new models example:
    python double_escape_dqn_train.py --num_episodes 20000 --num_steps 400 --normalize --update_step 10000 --time_bonus -0.0025 --wall_bonus -0.025 --door_bonus 0 --success_bonus 1
Continue training models example:
    python double_escape_dqn_train.py --source '2019-07-17-17-57' --num_episodes 100
"""
from __future__ import absolute_import, division, print_function

import sys
import os
import time
from datetime import datetime
import numpy as np
import tensorflow as tf
import rospy
import pickle

from envs.double_escape_task_env import DoubleEscapeEnv
from utils import data_utils, double_utils
from agents.dqn import DQNAgent

import pdb


if __name__ == "__main__":
    # create argument parser
    args = double_utils.get_args()
    # make an instance from env class
    env = DoubleEscapeEnv()
    env.reset()
    # model path
    date_time = datetime.now().strftime("%Y-%m-%d-%H-%M"),
    model_path_0 = os.path.dirname(sys.path[0])+"/saved_models/solo_escape/dqn/"+date_time+"/agent_0/model.h5"
    model_path_1 = os.path.dirname(sys.path[0])+"/saved_models/solo_escape/dqn/"+date_time+"/agent_1/model.h5"
    # create training parameters
    if not args.source: # source is empty, create new params
        complete_episodes = 0
        train_params = double_utils.create_train_params(date_time complete_episodes, args.source, args.normalize, args.num_episodes, args.num_steps, args.time_bonus, args.wall_bonus, args.door_bonus, args.success_bonus)
        # init agent parameters
        dim_state = len(double_utils.obs_to_state(env.observation, "all"))
        actions = np.array([np.array([1, -1]), np.array([1, 1])])
        layer_sizes = [256, 256]
        gamma = 0.99
        learning_rate = 1e-4
        batch_size = 2048
        memory_cap = 1000000
        update_step = 10000
        decay_period = args.num_episodes/4
        final_eps = 1e-2
        agent_params_0 = double_utils.create_agent_params(dim_state, actions, layer_sizes, gamma, learning_rate, batch_size, memory_cap, update_step, decay_period, final_eps)
        agent_params_0['update_counter'] = 0
        agent_params_1 = agent_params_0
        # instantiate new agents
        agent_0 = DQNAgent(agent_params_0)
        model_path_0 = os.path.dirname(sys.path[0])+"/saved_models/double_escape/dqn/"+date_time+"/agent_0/model.h5"
        agent_1 = DQNAgent(agent_params_1)
        model_path_1 = os.path.dirname(sys.path[0])+"/saved_models/double_escape/dqn/"+date_time+"/agent_1/model.h5"
        assert os.path.dirname(os.path.dirname(model_path_0)) == os.path.dirname(os.path.dirname(model_path_1))
        # init returns and losses storage
        train_params['ep_returns'] = []
        agent_params_0['ep_losses'] = []
        agent_params_1['ep_losses'] = []
        # init random starting poses
        train_params['pose_buffer'] = double_utils.create_pose_buffer(train_params['num_episodes']+1)
        # init first episode and step
        obs, _ = env.reset(train_params['pose_buffer'][0])
        state_0 = double_utils.obs_to_state(obs, "all")
        state_1 = double_utils.obs_to_state(obs, "all")
        train_params['success_count'] = 0
        # new means and stds
        mean_0 = state_0 # states average
        std_0 = np.zeros(agent_params_0["dim_state"])+1e-8 # n*Var
        mean_1 = state_1 # states average
        std_1 = np.zeros(agent_params_1["dim_state"])+1e-8 # n*Var
    else: # source is not empty, load params
        model_load_dir = os.path.dirname(sys.path[0])+"/saved_models/double_escape/dqn/"+args.source
        # load train parameters
        train_params_path = os.path.join(model_load_dir, "train_params.pkl")
        with open(train_params_path, 'rb') as f:
            train_params = pickle.load(f)
        train_params['source'] = args.source
        train_params["date_time"] = date_time
        ep_returns = train_params['ep_returns']
        # load agents parameters
        agent_params_path_0 = os.path.join(model_load_dir,"agent_0/agent0_parameters.pkl")
        with open(agent_params_path_0, 'rb') as f:
            agent_params_0 = pickle.load(f) # load agent_0 model
        agent_params_path_1 = os.path.join(model_load_dir,"agent_1/agent1_parameters.pkl")
        with open(agent_params_path_1, 'rb') as f:
            agent_params_1 = pickle.load(f) # load agent_1 model
        # load dqn models & memory buffers
        agent_0 = DQNAgent(agent_params_0)
        model_path_0 = os.path.dirname(sys.path[0])+"/saved_models/double_escape/dqn/"+date_time+"/agent_0/model.h5"
        agent_0.load_model(os.path.join(model_load_dir, "agent_0/model.h5"))
        agent_1 = DQNAgent(agent_params_1)
        model_path_1 = os.path.dirname(sys.path[0])+"/saved_models/double_escape/dqn/"+date_time+"/agent_1/model.h5"
        agent_1.load_model(os.path.join(model_load_dir, "agent_1/model.h5"))
        # init robots from loaded pose buffer
        obs, _ = env.reset(train_params['pose_buffer'][train_params['complete_episodes']])
        state_0 = double_utils.obs_to_state(obs, "all")
        state_1 = double_utils.obs_to_state(obs, "all")
        env.success_count = train_params['success_count']
        # load means and stds
        mean_0 = agent_params_0['mean']
        std_0 = agent_params_0['std']
        mean_1 = agent_params_1['mean']
        std_1 = agent_params_1['std']

    # learning
    start_time = time.time()
    ep = train_params['num_episodes'])
    while ep <= train_params['complete_episodes']:
        # check simulation crash
        if sum(np.isnan(state_0)) >= 1 or sum(np.isnan(state_1)) >= 1:
            print(bcolors.FAIL, "Simulation Crashed", bcolors.ENDC)
            train_params['complete_episodes'] = ep
            break # terminate main loop if simulation crashed
        epsilon_0 = agent_0.linearly_decaying_epsilon(decay_period=agent_params_0['decay_period'], episode=ep, init_eps=agent_params_0['init_eps'], final_eps=agent_params_0['final_eps'])
        epsilon_1 = agent_1.linearly_decaying_epsilon(decay_period=agent_params_1['decay_period'], episode=ep, init_eps=agent_params_1['init_eps'], final_eps=agent_params_1['final_eps'])
        logging.info("epsilon_0: {}, epsilon_1: {}".format(epsilon_0, epsilon_1))
        rospy.logdebug("epsilon_0: {}, epsilon_1: {}".format(epsilon_0, epsilon_1))
        done, ep_rewards, loss_vals_0, loss_vals_1 = False, [], [], []
        for st in range(train_params["num_steps"]):
            # check simulation crash
            if sum(np.isnan(state_0)) >= 1 or sum(np.isnan(state_1)) >= 1:
                logging.critical("Simulation Crashed")
                break # tbreakout loop if gazebo crashed
            # normalize states
            if train_params['normalize']:
                norm_state_0 = data_utils.normalize(state_0, mean_0, std_0)
                norm_state_1 = data_utils.normalize(state_1, mean_1, std_1)
                rospy.logdebug("\nagent_0 states normalize: {}\nagent_1 states normalize: {}".format(norm_state_0, norm_state_1))
            else:
                norm_state_0 = state_0
                norm_state_1 = state_1
            action_index_0 = agent_0.epsilon_greedy(norm_state_0)
            action_0 = agent_0.actions[action_index_0]
            action_index_1 = agent_1.epsilon_greedy(norm_state_1)
            action_1 = agent_1.actions[action_index_1]
            obs, rew, done, info = env.step(action_0, action_1)
            next_state_0 = double_utils.obs_to_state(obs, "all")
            next_state_1 = double_utils.obs_to_state(obs, "all")
            # compute incremental mean and std
            inc_mean_0 = data_utils.increment_mean(mean_0, next_state_0, (ep+1)*(st+1)+1)
            inc_std_0 = data_utils.increment_std(std_0, mean_0, inc_mean_0, next_state_0, (ep+1)*(st+1)+1)
            inc_mean_1 = data_utils.increment_mean(mean_1, next_state_1, (ep+1)*(st+1)+1)
            inc_std_1 = data_utils.increment_std(std_1, mean_1, inc_mean_1, next_state_1, (ep+1)*(st+1)+1)
            # update mean and std
            mean_0, std_0, mean_1, std_1 = inc_mean_0, inc_std_0, inc_mean_1, inc_std_1
            agent_params_0['mean'] = mean_0
            agent_params_0['std'] = std_0
            agent_params_0['mean'] = mean_1
            agent_params_1['std'] = std_1
            # normalize next states
            if train_params['normalize']:
                norm_next_state_0 = data_utils.normalize(next_state_0, mean_0, std_0)
                norm_next_state_1 = data_utils.normalize(next_state_1, mean_1, std_1)
                rospy.logdebug("\nagent_0 next states normalized: {}\nagent_1 next states normalized: {}".format(norm_next_state_0, norm_next_state_1))
            else:
                norm_next_state_0 = next_state_0
                norm_next_state_1 = next_state_1
            # adjust reward based on bonus options
            rew, done = double_utils.adjust_reward(train_params, env)
            ep_rewards.append(rew)
            train_params['success_count'] = env.success_count
            # store transitions
            if not info["status"][0] == "blew" or info["status"][1] == "blew":
                agent_0.replay_memory.store((norm_state_0, action_index_0, rew, done, norm_next_state_0))
                agent_1.replay_memory.store((norm_state_1, action_index_1, rew, done, norm_next_state_1))
                print(bcolors.OKBLUE, "transition saved to memory", bcolors.ENDC)
            else:
                print(bcolors.FAIL, "model blew up, transition not saved", bcolors.ENDC)
            # log step
            rospy.loginfo(
                "Episode: {}, Step: {}, epsilon_0: {}, epsilon_0: {} \nstate_0: {}, state_1: {}, \naction_0: {}, action_1: {}, \nnext_state_0: {}, next_state_1: {} \nreward/episodic_return: {}/{}, \nstatus: {}, \nnumber of success: {}".format(
                    ep+1,
                    st+1,
                    agent_0.epsilon,
                    agent_1.epsilon,
                    state_0,
                    state_1,
                    action_0,
                    action_1,
                    next_state_0,
                    next_state_1,
                    rew,
                    sum(ep_rewards),
                    info["status"],
                    env.success_count
                )
            )
            # train one epoch
            agent_0.train()
            loss_vals_0.append(agent_0.loss_value)
            agent_1.train()
            loss_vals_1.append(agent_1.loss_value)
            state_0 = next_state_0
            state_1 = next_state_1
            # update q-statble net
            agent_params_0['update_counter'] += 1
            agent_params_1['update_counter'] += 1
            if not agent_params_0['update_counter'] % agent_params_0['update_step']:
                agent_0.qnet_stable.set_weights(agent_0.qnet_active.get_weights())
                rospy.logerr("agent_0 Q-net weights updated!")
            if not agent_params_1['update_counter'] % agent_params_1['update_step']:
                agent_1.qnet_stable.set_weights(agent_1.qnet_active.get_weights())
                rospy.logerr("agent_1 Q-net weights updated!")
            if done:
                train_params['complete_episodes'] += 1
                break
        ep_returns.append(sum(ep_rewards))
        ep_losses_0.append(sum(loss_vals_0)/len(loss_vals_0))
        ep_losses_1.append(sum(loss_vals_1)/len(loss_vals_1))
        agent_0.save_model(model_path_0)
        agent_1.save_model(model_path_1)
        ep += 1
        # reset env
        obs, _ = env.reset(train_params['pose_buffer'][ep+1])
        state_0 = double_utils.obs_to_state(obs, "all")
        state_1 = double_utils.obs_to_state(obs, "all")

    # time training
    end_time = time.time()
    train_dur = end_time - start_time
    env.reset()

    # save replay buffer memories
    agent_0.save_memory(model_path_0)
    agent_1.save_memory(model_path_1)
    # save agent parameters
    data_utils.save_pkl(content=agent_params_0, fdir=os.path.dirname(model_path_0), fname="agent_parameters.pkl")
    data_utils.save_pkl(content=agent_params_1, fdir=os.path.dirname(model_path_1), fname="agent_parameters.pkl")
    # create train info
    train_info = train_params
    train_info["train_dur"] = train_dur
    train_info['success_count'] = env.success_count
    train_info["agent0_learning_rate"] = agent_params_0["learning_rate"]
    train_info["agent0_state_dimension"] = agent_params_0["dim_state"]
    train_info["agent0_action_options"] = agent_params_0["actions"]
    train_info["agent0_layer_sizes"] = agent_params_0["layer_sizes"]
    train_info["agent1_learning_rate"] = agent_params_1["learning_rate"]
    train_info["agent1_state_dimension"] = agent_params_1["dim_state"]
    train_info["agent1_action_options"] = agent_params_1["actions"]
    train_info["agent1_layer_sizes"] = agent_params_1["layer_sizes"]
    # save train info
    data_utils.save_csv(content=train_info, fdir=os.path.dirname(os.path.dirname(model_path_0)), fname="train_information.csv")
    data_utils.save_pkl(content=train_params, fdir=os.path.dirname(os.path.dirname(model_path_0)), fname="train_parameters.pkl")
    # save results
    np.save(os.path.join(os.path.dirname(model_path_0), 'ep_returns.npy'), ep_returns)
    np.save(os.path.join(os.path.dirname(model_path_1), 'ep_returns.npy'), ep_returns)
    np.save(os.path.join(os.path.dirname(model_path_0), 'ep_losses.npy'), ep_losses_0)
    np.save(os.path.join(os.path.dirname(model_path_1), 'ep_losses.npy'), ep_losses_1)

    # plot episodic returns
    data_utils.plot_returns(returns=ep_returns, mode=0, save_flag=True, fdir=os.path.dirname(os.path.dirname(model_path_0)))
    # plot accumulated returns
    data_utils.plot_returns(returns=ep_returns, mode=1, save_flag=True, fdir=os.path.dirname(os.path.dirname(model_path_0)))
    # plot averaged return
    data_utils.plot_returns(returns=ep_returns, mode=2, save_flag=True,
    fdir=os.path.dirname(os.path.dirname(model_path_0)))
