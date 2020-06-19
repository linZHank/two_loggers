"""
Plot accumulated returns
Author: LinZHanK (linzhank@gmail.com)
"""
from __future__ import absolute_import, division, print_function

import sys
import os
import pickle

import numpy as np
from numpy import pi
import matplotlib.pyplot as plt

ret_path_0 = sys.path[0]+"/saved_models/double_escape_discrete/dqn/2020-05-29-17-33/double_logger/ep_returns.npy"
ret_path_1 = sys.path[0]+"/saved_models/double_escape_discrete/dqn/2020-06-07-18-26/logger0/ep_returns.npy"

ret_0 = np.load(ret_path_0)
ret_1 = np.load(ret_path_1)

acc_ret_0 = []
ave_ret_0 = []
acc_0 = ret_0[0]
ave_0 = acc_0
for i,r in enumerate(ret_0):
    acc_0 += r
    ave_0 = acc_0/(i+1)
    acc_ret_0.append(acc_0)
    ave_ret_0.append(ave_0)
acc_ret_1 = []
ave_ret_1 = []
acc_1 = ret_1[0]
ave_1 = acc_1
for i,r in enumerate(ret_1):
    acc_1 += r
    ave_1 = acc_1/(i+1)
    acc_ret_1.append(acc_1)
    ave_ret_1.append(ave_1)

fig, ax = plt.subplots()
x = np.arange(len(ret_0))+1
ax.plot(x, ave_ret_0, 'r', label='Integrated Training')
ax.plot(x, ave_ret_1, 'b--', label='Distributed Training')
ax.set_xlim(0,30000)
ax.set_ylim(-200,400)
ax.set(xlabel='Episode', ylabel='Averaged Returns')
ax.grid()
ax.legend(loc='upper left')

plt.show()
