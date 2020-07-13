import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
import matplotlib.patches as mpatches
import tensorflow as tf

# load trajectory
traj_path = [
    os.path.join(sys.path[0], 'saved_trajectories', 'homo_8', 'traj.npy'),
    os.path.join(sys.path[0], 'saved_trajectories', 'hete_8', 'traj.npy'),
    os.path.join(sys.path[0], 'saved_trajectories', 'cent_8', 'traj.npy')
]

# traj = np.load(os.path.join(traj_dir, 'traj.npy'))
# acts = np.load(os.path.join(traj_dir, 'acts.npy'))
# load models
# create figure
fig, ax = plt.subplots(1,3, figsize=(36,10))

# draw trajectories
for i in range(3):
    traj = np.load(traj_path[i])
    # create wall patches
    patches = []
    n_wall = mpatches.Rectangle(xy=(-6,5), width=12, height=1, fc='gray')
    patches.append(n_wall)
    w_wall = mpatches.Rectangle(xy=(-6,-5), width=1, height=10, fc='gray')
    patches.append(w_wall)
    e_wall = mpatches.Rectangle(xy=(5,-5), width=1, height=10, fc='gray')
    patches.append(e_wall)
    sw_wall = mpatches.Rectangle(xy=(-6,-6), width=5, height=1, fc='gray')
    patches.append(sw_wall)
    se_wall = mpatches.Rectangle(xy=(1,-6), width=5, height=1, fc='gray')
    patches.append(se_wall)
    ax[i].plot(traj[:,0], traj[:,1], linestyle='--', color='grey')
    ax[i].plot(traj[:,-6], traj[:,-5], linestyle='--', color='black')
    # plot instances of robots, rod and robots' orientation
    ax[i].plot((traj[0,0], traj[0,-6]), (traj[0,1], traj[0,-5]), linewidth=4, color='orangered', alpha=0.7)
    patches.append(mpatches.Circle(xy=traj[0,:2], radius=.25, fill=False, ec='k', alpha=0.7))
    patches.append(mpatches.Circle(xy=traj[0,-6:-4], radius=.25, fc='k', alpha=0.7))
    ax[i].plot((traj[0,0], traj[0,0]+0.4*np.cos(traj[0,4])), (traj[0,1], traj[0,1]+0.4*np.sin(traj[0,4])), color='red')
    ax[i].plot((traj[0,-6], traj[0,-6]+0.4*np.cos(traj[0,-2])), (traj[0,-5], traj[0,-5]+0.4*np.sin(traj[0,-2])), color='red')
    for j in range(1, 11):
        ax[i].plot((traj[len(traj)*j/10-1,0], traj[len(traj)*j/10-1,-6]), (traj[len(traj)*j/10-1,1],traj[len(traj)*j/10-1,-5]), linewidth=4, color='orangered', alpha=0.7)
        patches.append(mpatches.Circle(xy=traj[len(traj)*j/10-1,:2], radius=.25, fill=False, ec='k', alpha=0.7))
        patches.append(mpatches.Circle(xy=traj[len(traj)*j/10-1,-6:-4], radius=.25, fc='k', alpha=0.7))
        ax[i].plot((traj[len(traj)*j/10-1,0], traj[len(traj)*j/10-1,0]+0.4*np.cos(traj[len(traj)*j/10-1,4])), (traj[len(traj)*j/10-1,1], traj[len(traj)*j/10-1,1]+0.4*np.sin(traj[len(traj)*j/10-1,4])), color='red')
        ax[i].plot((traj[len(traj)*j/10-1,-6], traj[len(traj)*j/10-1,-6]+0.4*np.cos(traj[len(traj)*j/10-1,-2])), (traj[len(traj)*j/10-1,-5], traj[len(traj)*j/10-1,-5]+0.4*np.sin(traj[len(traj)*j/10-1,-2])), color='red')
    # set traj axis
    patches_collection = PatchCollection(patches, match_original=True)
    ax[i].add_collection(patches_collection)
    ax[i].axis([-5,5,-9,5])
    ax[i].set_xticks(np.arange(-5,6))
    ax[i].set_yticks(np.arange(-9,6))
    ax[i].grid(color='grey', linestyle=':')
plt.tight_layout()
plt.show()

# # plot qvals
# qvals_0 = np.zeros(traj.shape[0]-1)
# qvals_1 = np.zeros_like(qvals_0)
# qvals_diff = np.zeros_like(qvals_0)
# traj_0 = traj.copy()
# traj_1 = traj.copy()
# traj_1[:,:6] = traj_1[:,-6:]
# for i in range(3):
#     dqn0 = tf.keras.models.load_model(model_paths_0[i])
#     dqn1 = tf.keras.models.load_model(model_paths_1[i])
#     for j in range(qvals_0.shape[1]):
#         qvals_0[i,j] = np.squeeze(dqn0(np.expand_dims(traj_0[j], axis=0)))[acts[j,0]]
#         qvals_1[i,j] = np.squeeze(dqn1(np.expand_dims(traj_1[j], axis=0)))[acts[j,1]]
#         qvals_diff[i,j] = np.absolute(qvals_0[i,j] - qvals_1[i,j])
#         # qvals_1[i,j] = np.max(dqn(np.expand_dims(traj_1[j], axis=0)))
#     ax_q[i].plot(np.arange(j+1), qvals_0[i], color=[.7,.7,.7], label='robot 1')
#     ax_q[i].plot(np.arange(j+1), qvals_1[i], color='k', label='robot 2')
# qvals_mae = np.mean(qvals_diff, axis=-1)
# 
