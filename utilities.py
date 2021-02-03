import airsim


class Human():
    def __init__(self, humanID):
        self.humanID = humanID

    def get_position(self, client):
        # position.x_val
        position = client.simGetObjectPose(self.HUMAN_ID).position
        return position

    def get_velocity(self, client):
        return

    def get_orientation(self, client):
        orientation = client.simGetObjectPose(self.HUMAN_ID).orientation
        return orientation

class Drone():
    def __init__(self):
        self.client = airsim.MultirotorClient()
        self.client.confirmConnection()
        self.client.enableApiControl(True)
        self.client.armDisarm(True)
        orientation = airsim.to_quaternion(-np.pi / 6, 0, 0)
        self.client.simSetCameraOrientation('0', orientation)

    
