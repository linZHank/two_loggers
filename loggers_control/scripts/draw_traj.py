"""
Read in robot trajectories then plot 'em
Author: LinZHanK (linzhank@gmail.com)
"""
from __future__ import absolute_import, division, print_function

import sys
import os
import numpy as np
from numpy import pi
import matplotlib.pyplot as plt


if __name__ == '__main__':
    # specify init config
    traj_dir = os.path.dirname(sys.path[0])+"/saved_models/double_escape/dqn/2019-12-31-17-43/validate_cases/5/"

    # save and plot trajectories
    # traj_0 = -np.asarray(traj_0)
    # traj_1 = -np.asarray(traj_1)
    traj_0 = np.load(os.path.join(traj_dir,'trajectory_0.npy'))
    traj_1 = np.load(os.path.join(traj_dir,'trajectory_1.npy'))
    # plot
    # left_wall = plt.Rectangle((-5,5),4,1,facecolor="grey")
    # right_wall = plt.Rectangle((1,5),4,1,facecolor="grey")
    left_wall = plt.Rectangle((-5,-6),4,1,facecolor="grey")
    right_wall = plt.Rectangle((1,-6),4,1,facecolor="grey")
    fig, ax = plt.subplots()
    ax.add_patch(left_wall)
    ax.add_patch(right_wall)
    ax.plot(traj_0[:,0], traj_0[:,1], 'r.', label='robot 1')
    ax.plot(traj_1[:,0], traj_1[:,1], 'b^', label='robot 2')
    ax.set_xlim(-5,5)
    # ax.set_ylim(0,9)
    ax.set_ylim(-9,3)
    ax.set(xlabel='X (m)', ylabel='Y (m)')
    ax.legend()
    ax.grid()
    plt.savefig(os.path.join(traj_dir,'traj.png'))
    # plt.show()
