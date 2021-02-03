# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import airsim
import os
import pprint
import numpy as np
import matplotlib.pyplot as plt
import pickle
import time
import math


def generate_cubic_poly(coeffis, times):
    '''
    Parameters
    ----------
    coeffis : [a, b, c, d]
    f(t) = a*t^3 + b*t^2 + c*t + d
    f'(t) = 3a*t^2 + 2b*t + c
        DESCRIPTION.
    times : a numpy array [t0, t1, t2, ...]
        DESCRIPTION.

    Returns
    -------
    pos = [f(t0), f(t1), ...]
    vel = [f'(t0), f'(t1), ...]
    '''
    assert len(coeffis) == 4, 'this is a cubic line, only four parameters'
    a = coeffis[0]
    b = coeffis[1]
    c = coeffis[2]
    d = coeffis[3]
    pos = []
    vel = []
    for t in times:
        pos.append(a * t ** 3 + b * t ** 2 + c * t + d)
        vel.append(3 * a * t ** 2 + 2 * b * t + c)
    return (pos, vel)


def quaternion2yaw(q):
    # https://robotics.stackexchange.com/questions/16471/get-yaw-from-quaternion
    # https://stackoverflow.com/questions/5782658/extracting-yaw-from-a-quaternion/5783030
    w = q.w_val
    x = q.x_val
    y = q.y_val
    z = q.z_val
    yaw = math.atan2(2.0 * (z * w + x * y), - 1.0 + 2.0 * (w * w + x * x))
    return yaw


times = np.linspace(0, 20, 200)
coeffis_x = [0.005, 0.003, 0.09, 0]
p_x, v_x = generate_cubic_poly(coeffis_x, times)

coeffis_y = [-0.007, -0.004, 0.08, 0]
p_y, v_y = generate_cubic_poly(coeffis_y, times)

coeffis_yaw = [0.0002, 0.004, 0.004, 0]
yaw, yaw_d = generate_cubic_poly(coeffis_yaw, times)

client = airsim.MultirotorClient()
client.confirmConnection()
client.enableApiControl(True)
client.armDisarm(True)

state = client.getMultirotorState()
s = pprint.pformat(state)
print("state: %s" % s)

airsim.wait_key('Press any key to takeoff')
client.takeoffAsync().join()

state = client.getMultirotorState()
print("state: %s" % pprint.pformat(state))

airsim.wait_key('Press any key to move vehicle to (0, 0, -30) at 5 m/s')
client.moveToPositionAsync(0, 0, -30, 5).join()

client.hoverAsync().join()
time.sleep(2)
z = -30
# plt.plot(p_x, p_y)
states = []
# update sampling time
Ts = 0.1
# command
command = {'vx': [], 'vy': [], 'yaw_rate': []}
for i in range(len(p_x) - 1):
    # one step ahead estimation
    x = client.getMultirotorState().kinematics_estimated.position.x_val
    y = client.getMultirotorState().kinematics_estimated.position.y_val
    q_now = client.getMultirotorState().kinematics_estimated.orientation
    yaw_now = quaternion2yaw(q_now)
    print('now yaw is {}'.format(yaw_now))
    vx = (p_x[i + 1] - x) / Ts
    vy = (p_y[i + 1] - y) / Ts
    yaw_rate = (yaw[i + 1] - yaw_now) / np.pi * 180 / Ts
    client.moveByVelocityAsync(vx, vy, 0, Ts,
                               airsim.DrivetrainType.MaxDegreeOfFreedom,
                               airsim.YawMode(True, yaw_rate)).join()
    states.append(client.getMultirotorState())
    command['vx'].append(vx)
    command['vy'].append(vy)
    command['yaw_rate'].append(yaw_rate)

info = {'states': states, 'p_x': p_x, 'v_x': v_x, 'p_y': p_y, 'v_y': v_y, \
        'yaw': yaw, 'yaw_d': yaw_d, 'command': command}
with open("states.txt", "wb") as fp:  # Pickling
    pickle.dump(info, fp)

client.hoverAsync().join()
state = client.getMultirotorState()
print("state: %s" % pprint.pformat(state))

airsim.wait_key('Press any key to reset to original state')

client.armDisarm(False)
client.reset()

# with open("states.txt", "rb") as fp:   # Unpickling
#     d = pickle.load(fp)
