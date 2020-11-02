#! /usr/bin/env python

from __future__ import absolute_import, division, print_function

import sys
import os
import numpy as np
import time
from datetime import datetime
import matplotlib.pyplot as plt
import tensorflow as tf

from envs.de import DoubleEscape
from agents.ppo import OnPolicyBuffer, PPOAgent


if __name__=='__main__':
    env = DoubleEscape()
    dim_obs = env.observation_space_shape[1]*2
    num_act = env.action_reservoir.shape[0]
    agent = PPOAgent(
        env_type='discrete',
        dim_obs=dim_obs,
        dim_act=num_act,
    )
    replay_buffer0 = OnPolicyBuffer(dim_obs=dim_obs, dim_act=1, size=2000, gamma=.99, lam=.97)
    replay_buffer1 = OnPolicyBuffer(dim_obs=dim_obs, dim_act=1, size=2000, gamma=.99, lam=.97)
    assert replay_buffer0.max_size==replay_buffer1.max_size
    model_dir = os.path.join(sys.path[0], 'saved_models', env.name, agent.name, 'homo', datetime.now().strftime("%Y-%m-%d-%H-%M"))
    # tensorboard
    summary_writer = tf.summary.create_file_writer(model_dir)
    summary_writer.set_as_default()
    # paramas
    steps_per_epoch = replay_buffer0.max_size
    num_epochs = 2
    actor_iters = 100
    critic_iters = 100
    save_freq=10
    # get ready
    obs, done, ep_ret, ep_len = env.reset(), False, 0, 0
    while 'blown' in env.status:
        obs, done, ep_ret, ep_len = env.reset(), False, 0, 0
    episode_counter, step_counter, success_counter = 0, 0, 0
    stepwise_rewards, episodic_returns, sedimentary_returns, episodic_steps = [], [], [], []
    start_time = time.time()
    onestep_ep = 0 # debug
    # main loop
    for ep in range(num_epochs):
        for st in range(steps_per_epoch):
            o0 = obs[[0,1]].flatten()
            o1 = obs[[1,0]].flatten()
            a0, v0, logp0 = agent.pi_of_a_given_s(np.expand_dims(o0, axis=0))
            a1, v1, logp1 = agent.pi_of_a_given_s(np.expand_dims(o1, axis=0))
            n_obs, rew, done, info = env.step(np.array([int(a0), int(a1)]))
            n_o0 = n_obs[[0,1]].flatten()
            n_o1 = n_obs[[1,0]].flatten()
            print("\nepisode: {}, step: {} \nstate: {} \naction: {} \nreward: {} \ndone: {} \nn_state: {} \ninfo: {}".format(episode_counter+1, ep_len+1, obs, (a0, a1), rew, done, n_obs, info))
            ep_ret += np.mean(rew)
            ep_len += 1
            stepwise_rewards.append(rew)
            step_counter += 1
            replay_buffer0.store(o0, a0, np.mean(rew), v0, logp0)
            replay_buffer1.store(o1, a1, np.mean(rew), v1, logp1)
            obs = n_obs # SUPER CRITICAL!!!
            # handle terminations
            timeout = (ep_len==env.max_episode_steps)
            terminal = done or timeout
            epoch_ended = (st==steps_per_epoch-1)
            if terminal or epoch_ended:
                if epoch_ended and not(terminal):
                    print("Warning: trajectory cut off by epoch at {} steps.".format(ep_len))
                if timeout or epoch_ended:
                    _, v0, _ = agent.pi_of_a_given_s(np.expand_dims(obs[[0,1]].flatten(), axis=0))
                    _, v1, _ = agent.pi_of_a_given_s(np.expand_dims(obs[[1,0]].flatten(), axis=0))
                else:
                    v0, v1 = 0, 0
                replay_buffer0.finish_path(v0)
                replay_buffer1.finish_path(v1)
                if terminal:
                    if ep_len==1:
                        onestep_ep+=1 # debug
                    episode_counter += 1
                    episodic_returns.append(ep_ret)
                    sedimentary_returns.append(sum(episodic_returns)/episode_counter)
                    episodic_steps.append(step_counter)
                    if info=='escaped':
                        success_counter += 1
                    print("\n----\nEpisode: {} TotalSteps: {}, EpReturn: {}, EpLength: {}, Succeeded: {}\n----\n".format(episode_counter, step_counter, ep_ret, ep_len, success_counter))
                    tf.summary.scalar("episode total reward", ep_ret, step=episode_counter)
                obs, done, ep_ret, ep_len = env.reset(), False, 0, 0
                while 'blown' in env.status:
                    obs, done, ep_ret, ep_len = env.reset(), False, 0, 0
        # update actor-critic
        loss_pi, loss_v, loss_info = agent.train(replay_buffer0.get(), actor_iters, critic_iters)
        loss_pi, loss_v, loss_info = agent.train(replay_buffer1.get(), actor_iters, critic_iters)
        print("\n====\nEpoch: {} \nStep: {} \nAveReturn: {} \nSucceeded: {} \nLossPi: {} \nLossV: {} \nKLDivergence: {} \nEntropy: {} \nTimeElapsed: {}\n====\n".format(episode_counter, step_counter, sedimentary_returns[-1], success_counter, loss_pi, loss_v, loss_info['kl'], loss_info['ent'], time.time()-start_time))
        tf.summary.scalar('loss_pi', loss_pi, step=ep)
        tf.summary.scalar('loss_v', loss_v, step=ep) 
        # Save model
        if not ep%save_freq or (ep==num_epochs-1):
            # save logits_net
            logits_net_path = os.path.join(model_dir, 'logits_net', str(ep))
            if not os.path.exists(os.path.dirname(logits_net_path)):
                os.makedirs(os.path.dirname(logits_net_path))
            agent.actor.logits_net.save(logits_net_path)
            # save val_net
            val_net_path = os.path.join(model_dir, 'val_net', str(ep))
            if not os.path.exists(os.path.dirname(val_net_path)):
                os.makedirs(os.path.dirname(val_net_path))
            agent.critic.val_net.save(val_net_path)
            # Save returns
            np.save(os.path.join(model_dir, 'episodic_returns.npy'), episodic_returns)
            np.save(os.path.join(model_dir, 'sedimentary_returns.npy'), sedimentary_returns)
            np.save(os.path.join(model_dir, 'episodic_steps.npy'), episodic_steps)
            with open(os.path.join(model_dir, 'training_time.txt'), 'w') as f:
                f.write("{}".format(time.time()-start_time))    

    print(onestep_ep) # debug
