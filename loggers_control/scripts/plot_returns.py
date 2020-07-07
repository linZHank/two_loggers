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

homo_path = sys.path[0]+"/saved_models/double_escape_discrete/dqn/2020-05-29-17-33/double_logger/ep_returns.npy"
hete_path = sys.path[0]+"/saved_models/double_escape_discrete/dqn/2020-06-07-18-26/logger0/ep_returns.npy"
cent_path = sys.path[0]+"/saved_models/double_escape_discrete/cent_dqn_full/2020-06-25-21-25/cent_dqn_full/ep_returns.npy"

ret_homo = np.load(homo_path)
ret_hete = np.load(hete_path)
ret_cent = np.load(cent_path)

acc_ret_homo = []
ave_ret_homo = []
acc_homo = ret_homo[0]
ave_homo = acc_homo
acc_ret_hete = ret_hete[0]
ave_ret_hete = acc_ret_homo
acc_ret_cent = ret_homo[0]
ave_ret_cent = acc_ret_homo
for i,r in enumerate(ret_homo):
    acc_homo += r
    ave_homo = acc_homo/(i+1)
    acc_ret_homo.append(acc_homo)
    ave_ret_homo.append(ave_homo)
acc_ret_hete = []
ave_ret_hete = []
acc_hete = ret_hete[0]
ave_hete = acc_hete
for i,r in enumerate(ret_hete):
    acc_hete += r
    ave_hete = acc_hete/(i+1)
    acc_ret_hete.append(acc_hete)
    ave_ret_hete.append(ave_hete)
acc_ret_cent = []
ave_ret_cent = []
acc_cent = ret_cent[0]
ave_cent = acc_cent
for i,r in enumerate(ret_cent):
    acc_cent += r
    ave_cent = acc_cent/(i+1)
    acc_ret_cent.append(acc_cent)
    ave_ret_cent.append(ave_cent)

fig, ax = plt.subplots()
x = np.arange(len(ret_homo))+1
ax.plot(x, ave_ret_homo, 'lightcoral', label='Homogeneous')
ax.plot(x, ave_ret_hete, 'dodgerblue', label='Heterogeneous')
ax.plot(x, ave_ret_cent, 'lightgreen', label='Centralized')
ax.set_xlim(0,30000)
ax.set_ylim(-200,400)
ax.set(xlabel='Episode', ylabel='Averaged Total Reward')
ax.grid()
ax.legend(loc='upper left')

plt.show()
