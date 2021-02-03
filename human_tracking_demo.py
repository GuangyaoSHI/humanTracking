import airsim
import pprint
import numpy as np
import math

client = airsim.MultirotorClient()
client.confirmConnection()
client.enableApiControl(True)
client.armDisarm(True)

client.takeoffAsync().join()
orientation = airsim.to_quaternion(-np.pi / 6, 0, 0)
client.simSetCameraOrientation('0', orientation)

client.moveToPositionAsync(0, 0, -30, 10).join()
client.hoverAsync().join()

# obtain human id
# -- Method 1
obj_list = client.simListSceneObjects('^(cart|Cart)[\w]*')
assert len(obj_list) == 1  # making sure there is only one human
HUMAN_ID = obj_list[0]


for i in range(1000):
    humanPosition = client.simGetObjectPose(HUMAN_ID).position
    humanPositionVec = np.array([humanPosition.x_val, humanPosition.y_val, humanPosition.z_val])
    robotPosition = client.getMultirotorState().kinematics_estimated.position
    robotOrientation = client.getMultirotorState().kinematics_estimated.orientation
    # (pitch, roll, yaw)
    eulerAngles = airsim.to_eularian_angles(robotOrientation)

    robotPositionVec = np.array([robotPosition.x_val, robotPosition.y_val, robotPosition.z_val])
    relativePosition = robotPositionVec - humanPositionVec
    assert np.linalg.norm(relativePosition) != 0, 'relative position is a zero vector'
    unitVec = relativePosition/np.linalg.norm(relativePosition)
    desiredPosition = humanPositionVec + 3*unitVec
    if desiredPosition[2] > -1.5:
        desiredPosition[2] = -1.5
    distanceVec = desiredPosition - robotPositionVec
    vx = distanceVec[0]
    vy = distanceVec[1]
    vz = distanceVec[2]
    yaw_rate = (math.atan2(-unitVec[1], -unitVec[0]) - eulerAngles[2])/np.pi * 180
    Ts = 0.5
    client.moveByVelocityAsync(vx, vy, vz, Ts,
                               airsim.DrivetrainType.MaxDegreeOfFreedom,
                               airsim.YawMode(True, yaw_rate)).join()
client.hoverAsync().join()
state = client.getMultirotorState()
print("state: %s" % pprint.pformat(state))

airsim.wait_key('Press any key to reset to original state')

client.armDisarm(False)
client.reset()