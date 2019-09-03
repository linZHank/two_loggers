#! /usr/bin/env python
"""
Tensorflow related tools
"""
import numpy as np
import tensorflow as tf

def mlp(x, sizes, activation=tf.tanh, output_activation=None):
    # Build a feedforward neural network.
    for size in sizes[:-1]:
        x = tf.layers.dense(x, units=size, activation=activation)
        return tf.layers.dense(x, units=sizes[-1], activation=output_activation)

def increment_mean(pre_mean, new_data, sample_size):
    """
    Compute incremental mean
    """
    inc_mean = pre_mean + (new_data-pre_mean) / sample_size

    return inc_mean

def increment_std(pre_std, pre_mean, inc_mean, new_data, sample_size):
    """
    Compute incremental standard deviation
    """
    pre_nVar = np.power(pre_std,2)*(sample_size-1)
    inc_std = np.sqrt((pre_nVar+(new_data-pre_mean)*(new_data-inc_mean)) / sample_size)

    return inc_std

def normalize(data, mean, std):
    """
    z-standardize
    """
    normed_data = (data - mean) / np.clip(std, 1e-8, 1e16)

    return normed_data
