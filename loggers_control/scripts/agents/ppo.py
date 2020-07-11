""" 
A PPO type agent class for pe env 
"""
import tensorflow as tf
import numpy as np
import tensorflow_probability as tfp
tfd = tfp.distributions
import rospy

################################################################
"""
Can safely ignore this block
"""
# restrict GPU and memory growth
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        # Visible devices must be set before GPUs have been initialized
        print(e)
    # Restrict TensorFlow to only use the first GPU
    try:
        tf.config.experimental.set_visible_devices(gpus[0], 'GPU')
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPU")
    except RuntimeError as e:
        # Visible devices must be set before GPUs have been initialized
        print(e)
################################################################


def mlp(dim_inputs, dim_outputs, activation, output_activation=None):
    """
    Take inputs of obs, output according to output_size
    Input should include two dimensions
    """
    # inputs
    inputs = tf.keras.Input(shape=(dim_inputs,), name='input')
    # features
    features = tf.keras.layers.Dense(64, activation=activation)(inputs)
    features = tf.keras.layers.Dense(64, activation=activation)(features)
    # outputs
    outputs = tf.keras.layers.Dense(dim_outputs, activation=output_activation)(features)

    return tf.keras.Model(inputs=inputs, outputs=outputs)

class Actor(tf.Module):
    def __init__(self, dim_obs, dim_act):
        super(Actor, self).__init__()
        self.log_std = tf.Variable(initial_value=-0.5*np.ones(dim_act, dtype=np.float32))
        self.mu_net = mlp(dim_inputs=dim_obs, dim_outputs=dim_act, activation='relu')

    def _distribution(self, obs):
        mu = tf.squeeze(self.mu_net(obs))
        std = tf.math.exp(self.log_std)

        return tfd.Normal(loc=mu, scale=std)

    def _log_prob_from_distribution(self, pi, act):
        return tf.math.reduce_sum(pi.log_prob(act), axis=-1)

    def __call__(self, obs, act=None):
        pi = self._distribution(obs)
        logp_a = None
        if act is not None:
            logp_a = self._log_prob_from_distribution(pi, act)

        return pi, logp_a
class Critic(tf.Module):
    def __init__(self, dim_obs):
        super(Critic, self).__init__()
        self.val_net = mlp(dim_inputs=dim_obs, dim_outputs=1, activation='relu')

    def __call__(self, obs):
        return tf.squeeze(self.val_net(obs), axis=-1)

class PPOAgent:
    def __init__(self, name='ppo_agent', dim_obs=6, dim_act=2, clip_ratio=0.2, lr_actor=3e-4,
                 lr_critic=1e-3, target_kl=0.01, batch_size = 128):
        self.name = name
        self.clip_ratio = clip_ratio
        self.actor = Actor(dim_obs, dim_act)
        self.critic = Critic(dim_obs)
        self.actor_optimizer = tf.keras.optimizers.Adam(learning_rate=lr_actor)
        self.critic_optimizer = tf.keras.optimizers.Adam(learning_rate=lr_critic)
        self.actor_loss_metric = tf.keras.metrics.Mean()
        self.critic_loss_metric = tf.keras.metrics.Mean()
        self.target_kl = target_kl
        self.batch_size = batch_size

    def pi_of_a_given_s(self, obs):
        with tf.GradientTape() as t:
            with t.stop_recording():
                pi = self.actor._distribution(obs) # policy distribution (Gaussian)
                act = pi.sample()
                logp_a = self.actor._log_prob_from_distribution(pi, act)
                val = tf.squeeze(self.critic(obs), axis=-1)

        return act.numpy(), val.numpy(), logp_a.numpy()

    def train(self, actor_dataset, critic_dataset, num_epochs):
        # update actor
        batched_actor_dataset = actor_dataset.shuffle(1024).batch(self.batch_size)
        for epch in range(num_epochs):
            rospy.logdebug("Staring actor epoch: {}".format(epch))
            ep_kl = tf.convert_to_tensor([]) # kl-divergence storage
            ep_ent = tf.convert_to_tensor([]) # entropy storage
            for step, batch in enumerate(batched_actor_dataset):
                with tf.GradientTape() as tape:
                    tape.watch(self.actor.trainable_variables)
                    pi, logp = self.actor(batch['obs'], batch['act']) 
                    ratio = tf.math.exp(logp - batch['logp']) # pi/old_pi
                    clip_adv = tf.math.multiply(tf.clip_by_value(ratio, 1-self.clip_ratio, 1+self.clip_ratio),
                                                batch['adv'])
                    obj = tf.math.minimum(tf.math.multiply(ratio, batch['adv']), clip_adv) # -.01*ent
                    loss_pi = -tf.math.reduce_mean(obj)
                    approx_kl = batch['logp'] - logp
                    ent = tf.math.reduce_sum(pi.entropy(), axis=-1)
                # gradient descent actor weights
                grads_actor = tape.gradient(loss_pi, self.actor.trainable_variables)
                self.actor_optimizer.apply_gradients(zip(grads_actor, self.actor.trainable_variables))
                self.actor_loss_metric(loss_pi)
                # record kl-divergence and entropy
                ep_kl = tf.concat([ep_kl, approx_kl], axis=0)
                ep_ent = tf.concat([ep_ent, ent], axis=0)
                # log loss_pi
                if not step%100:
                    rospy.logdebug("pi update step {}: mean_loss = {}".format(step, self.actor_loss_metric.result()))
            # log epoch
            kl = tf.math.reduce_mean(ep_kl)
            entropy = tf.math.reduce_mean(ep_ent)
            rospy.logdebug("Epoch :{} \nLoss: {} \nEntropy: {} \nKLDivergence: {}".format(
                epch+1,
                self.actor_loss_metric.result(),
                entropy,
                kl
            ))
            # early cutoff due to large kl-divergence
            if kl > 1.5*self.target_kl:
                rospy.logwarn("Early stopping at epoch {} due to reaching max kl-divergence.".format(epch+1))
                break
        # update critic
        batched_critic_dataset = critic_dataset.shuffle(1024).batch(self.batch_size)
        for epch in range(num_epochs):
            rospy.logdebug("Starting critic epoch: {}".format(epch))
            for step, batch in enumerate(batched_critic_dataset):
                with tf.GradientTape() as tape:
                    tape.watch(self.critic.trainable_variables)
                    loss_v = tf.keras.losses.MSE(batch['ret'], self.critic(batch['obs']))
                # gradient descent critic weights
                grads_critic = tape.gradient(loss_v, self.critic.trainable_variables)
                self.critic_optimizer.apply_gradients(zip(grads_critic, self.critic.trainable_variables))
                self.critic_loss_metric(loss_v)
                # log loss_v
                if not step%100:
                    rospy.logdebug("v update step {}: mean_loss = {}".format(step, self.critic_loss_metric.result()))
        
