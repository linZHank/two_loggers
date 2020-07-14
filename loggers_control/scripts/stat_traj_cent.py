import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
import matplotlib.patches as mpatches
import tensorflow as tf

# Physical Params
M_LOGGER = 3.6
R_LOGGER = 0.25
M_ROD = 0.04
R_ROD = 0.02
L_ROD = 2
J_LOGGER = 1./2*M_LOGGER*R_LOGGER**2
J_ROD = 1/12*M_ROD*(3*R_ROD**2+L_ROD**2)
# load trajectory
traj_path = os.path.join(sys.path[0], 'saved_trajectories', 'homo_0', 'traj.npy')
traj = np.load(traj_path)
traj_diff = traj[1:] - traj[:-1]
dist = 0
ke = 0

for td in traj_diff:
    dist += np.linalg.norm(td[:2]) + np.linalg.norm(td[-6:-4])
for t in traj:
    ke += .5*M_LOGGER*np.linalg.norm(t[2:4])**2 \
        + .5*M_LOGGER*np.linalg.norm(t[-4:-2])**2 \
        + .5*M_ROD*np.linalg.norm(t[8:10])**2 \
        + .5*J_LOGGER*t[5]**2 \
        + .5*J_LOGGER*t[-1]**2 \
        + .5*J_ROD*t[11]**2
time = len(traj_diff)
# load model
model_path = os.path.join(sys.path[0], 'saved_models/double_escape_discrete/dqn/2020-05-29-17-33/double_logger/models/5093500.h5')
qvals_0 = np.zeros(traj.shape[0]-1)
qvals_1 = np.zeros_like(qvals_0)
qvals_diff = np.zeros_like(qvals_0)
dqn = tf.keras.models.load_model(model_path)
acts = np.load(os.path.join(os.path.dirname(traj_path), 'acts.npy'))
for i in range(qvals_0.shape[0]):
    qvals_0[i] = np.squeeze(dqn(np.expand_dims(traj[i], axis=0)))[acts[i,0]]
    qvals_1[i] = np.squeeze(dqn(np.expand_dims(traj[i], axis=0)))[acts[i,1]]
    qvals_diff[i] = np.absolute(qvals_0[i] - qvals_1[i])
qvals_mae = np.mean(qvals_diff, axis=-1)
# save stats
with open(os.path.join(os.path.dirname(traj_path), 'travel_distance.txt'), 'w') as f:
    f.write("{}".format(dist))
with open(os.path.join(os.path.dirname(traj_path), 'time_elapsed.txt'), 'w') as f:
    f.write("{}".format(time))
with open(os.path.join(os.path.dirname(traj_path), 'kinetic_energy.txt'), 'w') as f:
    f.write("{}".format(ke))
with open(os.path.join(os.path.dirname(traj_path), 'delta_q.txt'), 'w') as f:
    f.write("{}".format(qvals_mae))

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
