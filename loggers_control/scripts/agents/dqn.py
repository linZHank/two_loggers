from __future__ import absolute_import, division, print_function

import numpy as np
import random
import tensorflow as tf
import rospy

from utils import gen_utils
from utils import solo_utils
from utils.gen_utils import bcolors
from tensorflow.keras.layers import Dense
from tensorflow.keras import Model

import pdb


class Memory:
    def __init__(self, memory_cap):
        self.memory_cap = memory_cap
        self.memory = []
    def store(self, experience):
        # pop a random experience if memory full
        if len(self.memory) >= self.memory_cap:
            self.memory.pop(random.randint(0, len(self.memory)-1))
        self.memory.append(experience)

    def sample_batch(self, batch_size):
        # Select batch
        if len(self.memory) < batch_size:
            batch = random.sample(self.memory, len(self.memory))
        else:
            batch = random.sample(self.memory, batch_size)

        return zip(*batch)

class DQNAgent:
    def __init__(self, hyp_params):
        # super(DQNAgent, self).__init__()
        # hyper-parameters
        self.epsilon = hyp_params["epsilon"]
        self.actions = hyp_params["actions"]
        self.gamma = hyp_params["gamma"]
        self.learning_rate = hyp_params["learning_rate"]
        self.dim_state = hyp_params["dim_state"]
        self.num_episodes = hyp_params["num_episodes"]
        self.num_steps = hyp_params["num_steps"]
        self.batch_size = hyp_params["batch_size"]
        self.memory_cap = hyp_params["memory_cap"]
        self.update_step = hyp_params["update_step"]
        self.wall_bonus = hyp_params["wall_bonus"]
        self.door_bonus = hyp_params["door_bonus"]
        self.dist_bonus = hyp_params["dist_bonus"]
        self.model_path = hyp_params["model_path"]
        # Q(s,a;theta)
        self.qnet_active = tf.keras.models.Sequential([
            Dense(64, input_shape=(self.dim_state, ), activation='relu'),
            Dense(64, activation='relu'),
            Dense(len(self.actions))
        ])
        # Q^(s,a;theta_)
        self.qnet_stable = tf.keras.models.Sequential([
            Dense(64, input_shape=(self.dim_state, ), activation='relu'),
            Dense(64, activation='relu'),
            Dense(len(self.actions))
        ])
        # optimizer
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
        # init replay memory
        self.replay_memory = Memory(memory_cap=self.memory_cap)

    def epsilon_greedy(self, state):
        """
        If a random number(0,1) beats epsilon, return index of largest Q-value.
        Else, return a random index
        """
        if np.random.rand() > self.epsilon:
            return np.argmax(self.qnet_active.predict(state.reshape(1,-1)))
        else:
            print(bcolors.WARNING, "Take a random action!", bcolors.ENDC)
            return np.random.randint(len(self.actions))

    def epsilon_decay(self, i_episode):
        if 1 - 2*(i_episode/self.num_episodes) >= 1e-3:
            return 1 - 2*(i_episode/self.num_episodes)
        else:
            return 1e-3

    def loss(self, batch_states, batch_actions, batch_rewards, batch_done_flags, batch_next_states):
        loss_object = tf.keras.losses.MeanSquaredError()
        q_values = tf.math.reduce_sum(self.qnet_active(batch_states) * tf.one_hot(batch_actions, len(self.actions)), axis=-1)
        target_q = batch_rewards + (1. - batch_done_flags) * 0.99 * tf.math.reduce_max(self.qnet_stable(batch_next_states),axis=-1)

        return loss_object(y_true=target_q, y_pred=q_values)

    def grad(self, batch_states, batch_actions, batch_rewards, batch_done_flags, batch_next_states):
        with tf.GradientTape() as tape:
            loss_value = self.loss(batch_states, batch_actions, batch_rewards, batch_done_flags, batch_next_states)

        return loss_value, tape.gradient(loss_value, self.qnet_active.trainable_variables)

    def save_model(self):
        self.qnet_active.save_weights(self.model_path)

    def train(self, env):
        update_counter = 0
        ep_returns = []
        for ep in range(self.num_episodes):
            self.epsilon = self.epsilon_decay(ep)
            obs, _ = env.reset()
            state_0 = solo_utils.obs_to_state(obs)
            dist_0 = np.linalg.norm(state_0[:2]-np.array([0,-6.0001]))
            done, ep_rewards = False, []
            for st in range(self.num_steps):
                act_id = self.epsilon_greedy(state_0)
                action = self.actions[act_id]
                obs, rew, done, info = env.step(action)
                state_1 = solo_utils.obs_to_state(obs)
                dist_1 = np.linalg.norm(state_1[:2]-np.array([0,-6.0001]))
                delta_dist = dist_0 - dist_1
                # adjust reward based on relative distance to the exit
                rew, done = solo_utils.adjust_reward(rew, info, delta_dist, done, self.wall_bonus, self.door_bonus, self.dist_bonus)
                # log the progress
                print(
                    bcolors.OKGREEN,
                    "Episode: {}, Step: {} \naction: {}->{}, state: {}, reward: {}, status: {}".format(
                        ep,
                        st,
                        act_id,
                        action,
                        state_1,
                        rew,
                        info
                    ),
                    bcolors.ENDC
                )
                # train an epoch
                self.replay_memory.store((state_0, act_id, rew, done, state_1))
                minibatch = self.replay_memory.sample_batch(self.batch_size)
                (batch_states, batch_actions, batch_rewards, batch_done_flags, batch_next_states) = [np.array(minibatch[i]) for i in range(len(minibatch))]
                # compute gradient for one epoch
                loss_value, grads = self.grad(batch_states, batch_actions, batch_rewards, batch_done_flags, batch_next_states)
                self.optimizer.apply_gradients(zip(grads, self.qnet_active.trainable_variables))
                loss_value = self.loss(batch_states, batch_actions, batch_rewards, batch_done_flags, batch_next_states)
                print("loss: {}".format(loss_value))
                state_0 = state_1
                dist_0 = dist_1
                ep_rewards.append(rew)
                update_counter += 1
                if not update_counter % self.update_step:
                    self.qnet_stable.set_weights(self.qnet_active.get_weights())
                    print(bcolors.BOLD, "Q-net weights updated!", bcolors.ENDC)
                if done:
                    ep_returns.append(sum(ep_rewards))
                    print(bcolors.OKBLUE, "Episode: {}, Success Count: {}".format(ep, env.success_count),bcolors.ENDC)
                    self.save_model()
                    print("model saved at {}".format(self.model_path))
                    break
        gen_utils.plot_returns(returns=ep_returns, mode=2, save_flag=True, path=self.model_path)
