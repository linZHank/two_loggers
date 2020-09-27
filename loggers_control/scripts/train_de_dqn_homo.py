#! /usr/bin/env python

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
    agent = DeepQNet(
        dim_obs=dim_obs,
        num_act=num_act,
        lr=1e-4,
        polyak=-1,
    )
    replay_buffer = ReplayBuffer(dim_obs=dim_obs, size=int(2e6))
    model_dir = os.path.join(sys.path[0], 'saved_models', env.name, agent.name, datetime.now().strftime("%Y-%m-%d-%H-%M"))
    # tensorboard
    summary_writer = tf.summary.create_file_writer(model_dir)
    summary_writer.set_as_default()
    # params
    batch_size = 1024
    train_freq = 100
    train_after = 20000
    warmup_episodes = 500
    decay_period = 1500
    total_steps = int(3e6)
    episodic_returns = []
    sedimentary_returns = []
    episodic_steps = []
    step_counter = 0
    episode_counter = 0
    success_counter = 0
    save_freq = 1000
    obs, done, ep_ret, ep_len = env.reset(), False, 0, 0
    start_time = time.time()
    # start training
    while step_counter <= total_steps:
        while 'blown' in env.status:
            obs, ep_ret, ep_len = env.reset(), 0, 0
        s0 = obs[[0,-1]].flatten()
        s1 = obs[[1,-1]].flatten()
        a0 = np.squeeze(agent.act(np.expand_dims(s0, axis=0)))
        a1 = np.squeeze(agent.act(np.expand_dims(s1, axis=0)))
        n_obs, rew, done, info = env.step(np.array([int(a0), int(a1)]))
        n_s0 = n_obs[[0,-1]].flatten()
        n_s1 = n_obs[[1,-1]].flatten()
        rospy.logdebug("\nstate: {} \naction: {} \nreward: {} \ndone: {} \nn_state: {}".format(obs, (a0, a1), rew, done, n_obs))
        ep_ret += np.sum(rew)
        ep_len += 1
        replay_buffer.store(s0, a0, np.sum(rew), done, n_s0)
        replay_buffer.store(s1, a1, np.sum(rew), done, n_s1)
        obs = n_obs.copy() # SUPER CRITICAL
        step_counter += 1
        # train one batch
        if not step_counter%train_freq and step_counter>train_after:
            for _ in range(train_freq):
                minibatch = replay_buffer.sample_batch(batch_size=batch_size)
                loss_q = agent.train_one_batch(data=minibatch)
                print("\nloss_q: {}".format(loss_q))
        # handle episode termination
        if done or (ep_len==env.max_episode_steps):
            episode_counter +=1
            episodic_returns.append(ep_ret)
            sedimentary_returns.append(sum(episodic_returns)/episode_counter)
            episodic_steps.append(step_counter)
            if info.count('escaped')==2:
                success_counter += 1
            rospy.loginfo("\n====\nEpisode: {} \nEpisodeLength: {} \nTotalSteps: {} \nEpisodeReturn: {} \nSucceeded: {} \nSedimentaryReturn: {} \nTimeElapsed: {} \n====\n".format(episode_counter, ep_len, step_counter, ep_ret, success_counter, sedimentary_returns[-1], time.time()-start_time))
            tf.summary.scalar("episode return", ep_ret, step=episode_counter)
            # save model
            if not episode_counter%save_freq or step_counter==total_steps:
                model_path = os.path.join(model_dir, str(episode_counter))
                if not os.path.exists(os.path.dirname(model_path)):
                    os.makedirs(os.path.dirname(model_path))
                agent.q.q_net.save(model_path)
                # Save returns
                np.save(os.path.join(model_dir, 'episodic_returns.npy'), episodic_returns)
                np.save(os.path.join(model_dir, 'sedimentary_returns.npy'), sedimentary_returns)
                np.save(os.path.join(model_dir, 'episodic_steps.npy'), episodic_steps)
                with open(os.path.join(model_dir, 'training_time.txt'), 'w') as f:
                    f.write("{}".format(time.time()-start_time))
            # reset env
            obs, done, ep_ret, ep_len = env.reset(), False, 0, 0
            agent.linear_epsilon_decay(episode_counter, decay_period, warmup_episodes)

    # plot returns
    fig, ax = plt.subplots(figsize=(8, 6))
    fig.suptitle('Averaged Returns')
    ax.plot(sedimentary_returns)
    plt.show()
