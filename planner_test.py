import numpy as np
from planner import *

traj_init = np.zeros((9, 3))
for i in range(9):
    traj_init[i, :] = np.array([i + 1, i + 1, 1])

q0 = [np.array([0, 0, 0]), np.array([0, 0, 0]), np.array([0, 0, 0])]
TO = trajectory_planner(traj_init, q0)
TO.get_TSDF(np.array([5,6,1]))
TO.get_TSDF_gradient(np.array([5,6,1]))
a = TO.get_human_trajectory()
TO.collision_cost_map(np.array([5,6,1]))
TO.smooth_gradient(TO.current_traj).shape
TO.safety_gradient(TO.current_traj).shape
TO.shot_quality_gradient(TO.current_traj).shape
TO.safety_cost(TO.current_traj)
TO.smooth_cost(TO.current_traj)
TO.shot_quality_cost(TO.current_traj)
TO.value(TO.current_traj)
TO.optimize()
import scipy.io
scipy.io.savemat('trajectories.mat', mdict={'trajs': TO.history})