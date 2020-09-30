#! /usr/bin/env python
"""
Train identical loggers in double_escape env using heterogeneous DQN with shared reward
"""

from __future__ import absolute_import, division, print_function

import sys
import os
import numpy as np
import time
from datetime import datetime
import matplotlib.pyplot as plt
import rospy
import tensorflow as tf

from envs.de import DoubleEscape
from agents.dqn import ReplayBuffer, DeepQNet


if __name__=='__main__':
    env = DoubleEscape()
    dim_obs = env.observation_space_shape[1]*2
    num_act = env.action_reservoir.shape[0]
    agent_0 = DeepQNet(
        dim_obs=dim_obs,
        num_act=num_act,
        lr=5e-5,
    )
    agent_1 = DeepQNet(
        dim_obs=dim_obs,
        num_act=num_act,
        lr=5e-5,
    )
    replay_buffer_0 = ReplayBuffer(dim_obs=dim_obs, size=int(2e6))
    replay_buffer_1 = ReplayBuffer(dim_obs=dim_obs, size=int(2e6))
    model_dir = os.path.join(sys.path[0], 'saved_models', env.name, agent_0.name, datetime.now().strftime("%Y-%m-%d-%H-%M"))
    # tensorboard
    summary_writer = tf.summary.create_file_writer(model_dir)
    summary_writer.set_as_default()
    # params
    batch_size = 8192
    switch_flag = False # this is hete unique
    switch_freq = 10 # this is hete unique
    train_freq = 100
    train_after = 20000
    warmup_episodes = 500
    decay_period = 1500
    total_steps = int(4e6)
    episodic_returns = []
    sedimentary_returns = []
    episodic_steps = []
    step_counter = 0
    episode_counter = 0
    success_counter = 0
    save_freq = 1000
    obs, done, ep_ret, ep_len = env.reset(), False, np.zeros(2), 0
    start_time = time.time()
    # start training
    while step_counter <= total_steps:
        while 'blown' in env.status: 
            obs, ep_ret, ep_len = env.reset(), np.zeros(2), 0
        s0 = obs[[0,-1]].flatten()
        s1 = obs[[1,-1]].flatten()
        a0 = np.squeeze(agent_0.act(np.expand_dims(s0, axis=0)))
        a1 = np.squeeze(agent_1.act(np.expand_dims(s1, axis=0)))
        n_obs, rew, done, info = env.step(np.array([int(a0), int(a1)]))
        n_s0 = n_obs[[0,-1]].flatten()
        n_s1 = n_obs[[1,-1]].flatten()
        rospy.logdebug("\nstate: {} \naction: {} \nreward: {} \ndone: {} \nn_state: {}".format(obs, (a0, a1), rew, done, n_obs))
        ep_ret += rew
        ep_len += 1
        replay_buffer_0.store(s0, a0, rew[0], done, n_s0)
        replay_buffer_1.store(s1, a1, rew[1], done, n_s1)
        obs = n_obs.copy() # SUPER CRITICAL
        step_counter += 1
        # train one batch
        if not step_counter%train_freq and step_counter>train_after:
            if not step_counter%(train_freq*switch_freq):
                switch_flag = not switch_flag
            if switch_flag:
                for _ in range(train_freq):
                    minibatch_0 = replay_buffer_0.sample_batch(batch_size=batch_size)
                    loss_q_0 = agent_0.train_one_batch(data=minibatch_0)
                    print("\nloss_q0: {}".format(loss_q_0))
            else:
                for _ in range(train_freq):
                    minibatch_1 = replay_buffer_1.sample_batch(batch_size=batch_size)
                    loss_q_1 = agent_1.train_one_batch(data=minibatch_1)
                    print("\nloss_q1: {}".format(loss_q_1))
        # handle episode termination
        if done or (ep_len==env.max_episode_steps):
            episode_counter += 1
            episodic_returns.append(ep_ret)
            sedimentary_returns.append(np.sum(np.array(episodic_returns), axis=0)/episode_counter)
            episodic_steps.append(step_counter)
            if info.count('escaped')==2:
                success_counter += 1
            rospy.loginfo("\n====\nEpisode: {} \nEpisodeLength: {} \nTotalSteps: {} \nEpisodeReturn: {} \nSucceeded: {} \nSedimentaryReturn: {} \nTimeElapsed: {} \n====\n".format(episode_counter, ep_len, step_counter, ep_ret, success_counter, sedimentary_returns[-1], time.time()-start_time))
            tf.summary.scalar("episode return 0", ep_ret[0], step=episode_counter)
            tf.summary.scalar("episode return 1", ep_ret[1], step=episode_counter)
            # save model
            if not episode_counter%save_freq or step_counter==total_steps:
                model_path_0 = os.path.join(model_dir, 'agent_0', str(episode_counter))
                if not os.path.exists(os.path.dirname(model_path_0)):
                    os.makedirs(os.path.dirname(model_path_0))
                agent_0.q.q_net.save(model_path_0)
                model_path_1 = os.path.join(model_dir, 'agent_1', str(episode_counter))
                if not os.path.exists(os.path.dirname(model_path_1)):
                    os.makedirs(os.path.dirname(model_path_1))
                agent_1.q.q_net.save(model_path_1)
                # Save returns 
                np.save(os.path.join(model_dir, 'episodic_returns.npy'), episodic_returns)
                np.save(os.path.join(model_dir, 'sedimentary_returns.npy'), sedimentary_returns)
                np.save(os.path.join(model_dir, 'episodic_steps.npy'), episodic_steps)
                with open(os.path.join(model_dir, 'training_time.txt'), 'w') as f:
                    f.write("{}".format(time.time()-start_time))
            if all(sedimentary_returns[-1])>90:
                print("\nSolved at episode {}, total step {}: average reward: {:.2f}!".format(episode_counter, step_counter, sedimentary_returns[-1]))
                break
            # reset env
            obs, done, ep_ret, ep_len = env.reset(), False, np.zeros(2), 0 
            agent_0.linear_epsilon_decay(episode_counter, decay_period, warmup_episodes)
            agent_1.linear_epsilon_decay(episode_counter, decay_period, warmup_episodes)

    # save final results
    model_path_0 = os.path.join(model_dir, 'agent_0', str(episode_counter))
    agent_0.q.q_net.save(model_path_0)
    model_path_1 = os.path.join(model_dir, 'agent_1', str(episode_counter))
    agent_1.q.q_net.save(model_path_1)
    np.save(os.path.join(model_dir, 'episodic_returns.npy'), np.array(episodic_returns))
    np.save(os.path.join(model_dir, 'sedimentary_returns.npy'), np.array(sedimentary_returns))
    np.save(os.path.join(model_dir, 'episodic_steps.npy'), episodic_steps)
    with open(os.path.join(model_dir, 'training_time.txt'), 'w') as f:
        f.write("{}".format(time.time()-start_time))

    # plot returns
    fig, ax = plt.subplots(figsize=(8, 6))
    fig.suptitle('Averaged Returns')
    ax.plot(sedimentary_returns)
    plt.show()


