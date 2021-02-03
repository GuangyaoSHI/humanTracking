#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 29 21:48:48 2021

@author: hello_robot
"""

import airsim
import pprint
import time
import numpy as np

client = airsim.MultirotorClient()
client.confirmConnection()
client.enableApiControl(True)
client.armDisarm(True)

state = client.getMultirotorState()
s = pprint.pformat(state)
print("state: %s" % s)

airsim.wait_key('Press any key to takeoff')
client.takeoffAsync().join()

orientation = airsim.to_quaternion(-np.pi / 6, 0, 0)
client.simSetCameraOrientation('0', orientation)
state = client.getMultirotorState()
print("state: %s" % pprint.pformat(state))
cameraState = client.simGetCameraInfo('0')
print("camera state: %s" % pprint.pformat(cameraState))

airsim.wait_key('Press any key to move vehicle to (0, 0, -30) at 5 m/s')
client.moveToPositionAsync(0, 0, -30, 5).join()
robotOrientation = client.getMultirotorState().kinematics_estimated.orientation
eulerAngles = airsim.to_eularian_angles(robotOrientation)
print("orientation is : %s" % pprint.pformat(eulerAngles))
print('next we will increase yaw angle')
yaw_rate = 30
client.moveByVelocityAsync(1, 1, 0, 1,
                           airsim.DrivetrainType.MaxDegreeOfFreedom,
                           airsim.YawMode(True, yaw_rate)).join()

robotOrientation = client.getMultirotorState().kinematics_estimated.orientation
eulerAngles = airsim.to_eularian_angles(robotOrientation)
print("after rotation the robot orientation is : %s" % pprint.pformat(eulerAngles))
client.hoverAsync().join()

# obtain human id
# -- Method 1
obj_list = client.simListSceneObjects('^(cart|Cart)[\w]*')
assert len(obj_list) == 1  # making sure there is only one human
HUMAN_ID = obj_list[0]

for i in range(1):
    humanState = client.simGetObjectPose(HUMAN_ID)
    print("human state: %s" % pprint.pformat(humanState))
    time.sleep(5)

airsim.wait_key('Press any key to reset to original state')

client.armDisarm(False)
client.reset()
