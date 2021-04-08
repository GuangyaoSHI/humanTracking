import numpy as np
import math
import copy

def normalize_vector(X):
    return X / np.linalg.norm(X)


class trajectory_planner():
    def __init__(self, traj_init, q0):
        # initialize a trajectory
        # each point in traj_init is a row vector
        self.current_traj = traj_init
        self.last_traj = np.ones(traj_init.shape)*1000
        # q0 = [q, q_dot, q_ddot] each of them is a column vector
        self.q0 = q0
        # horizon
        self.n = 10
        # for A_smooth
        self.delta_t = 0.5
        self.K = np.diag([1] * (self.n - 1))
        self.K[1:, 0:-1] += np.diag([-1] * (self.n - 2))
        # parameters for smoothness
        self.A_smooth = []
        self.b_smooth = []
        self.c_smooth = []
        self.update_smooth_parameter()
        # parameters for shot quality
        self.theta_rel = np.pi / 2
        self.psi_rel = np.pi / 3
        self.rho = 3
        self.A_shot = []
        self.b_shot = []
        self.c_shot = []
        self.update_shot_parameter()
        # parameters for collision avoidance
        # self.TSDF = self.get_TSDF()
        # self.TSDF_gradient = self.get_TSDF_gradient()
        # threshold for collision avoidance
        self.epsilon = 0.2
        # optimization parameters
        self.eta = 10
        self.lambda1 = 2  # obstacle avoidance
        self.lambda2 = 0.5  # shot quality
        self.AA = np.linalg.inv(self.A_smooth + self.lambda1 * self.A_shot)
        self.maximum_iter = 100
        self.epsilon0 = 0.1
        self.epsilon1 = 0.1
        self.history = [traj_init]

    def get_TSDF(self, X):
        # assume two hemi-sphere obstacle
        assert X[-1] >=0, 'should be above the plane'
        distance1 = np.linalg.norm(np.array([5, 5, 0])-X)
        distance2 = np.linalg.norm(np.array([15, 5, 0]) - X)
        if distance1 < distance2: # closer to obstacle 1
            # distance to the surface
            return distance1-3
        else:
            return distance2-3

    def get_TSDF_gradient(self, X):
        # assume two hemi-sphere obstacle
        assert X[-1] >= 0, 'should be above the plane'
        distance1 = np.linalg.norm(np.array([5, 5, 0]) - X)
        distance2 = np.linalg.norm(np.array([15, 5, 0]) - X)
        if distance1 < distance2:  # closer to obstacle 1
            return (X-np.array([5, 5, 0]))/np.linalg.norm(X-np.array([5, 5, 0]))
        else:
            return (X-np.array([15, 5, 0]))/np.linalg.norm(X-np.array([15, 5, 0]))

    def get_human_trajectory(self):
        # return the next n-1 step information
        predictions = np.ones((self.n-1, 3))
        for i in range(self.n - 1):
            predictions[i, 2] = 1
            predictions[i, 0] = i+1
        human_trajectory = {'position': predictions, 'yaw': np.zeros(self.n-1)}
        return human_trajectory

    def update_smooth_parameter(self):
        e = np.zeros((self.n - 1, 3))
        e[0, :] += -self.q0[0]
        e_dot = np.zeros((self.n - 1, 3))
        e_dot[0, :] += -self.q0[1]
        e_ddot = np.zeros((self.n - 1, 3))
        e_ddot[0, :] += -self.q0[2]

        K0 = self.K / self.delta_t
        K1 = self.K @ self.K / self.delta_t ** 2
        K2 = self.K @ self.K @ self.K
        e0 = e / self.delta_t
        e1 = (self.K @ e + e_dot) / self.delta_t ** 2
        e2 = (self.K @ self.K @ e + self.K @ e_dot + e_ddot) / self.delta_t ** 3

        A0 = np.transpose(K0) @ K0
        A1 = np.transpose(K1) @ K1
        A2 = np.transpose(K2) @ K1

        b0 = np.transpose(K0) @ e0
        b1 = np.transpose(K1) @ e1
        b2 = np.transpose(K2) @ e2

        c0 = np.transpose(e0) @ e0
        c1 = np.transpose(e1) @ e1
        c2 = np.transpose(e2) @ e2

        self.A_smooth = A0 + A1 + A2
        self.b_smooth = b0 + b1 + b2
        self.c_smooth = c0 + c1 + c2

    def update_shot_parameter(self):
        # human_trajectory = {'positions':np.array([[x,y,z],[]]), 'yaw':np.array()}
        human_trajctory = self.get_human_trajectory()
        # just use future positions and don't use current position
        human_position = human_trajctory['position']
        human_yaw = human_trajctory['yaw']
        shot_traj = human_position
        for i in range(self.n - 1):
            shot_traj[i] = human_position[i] + self.rho * np.array(
                [math.cos(human_yaw[i] + self.psi_rel) * math.sin(self.theta_rel),
                 math.sin(human_yaw[i] + self.psi_rel) * math.cos(self.theta_rel),
                 math.cos(self.theta_rel)
                 ])
        K_shot = -np.eye(self.n - 1)
        self.A_shot = np.transpose(K_shot) @ K_shot
        self.b_shot = np.transpose(self.K) @ shot_traj
        self.c_shot = np.transpose(shot_traj) @ shot_traj

    def collision_cost_map(self, X):
        if self.get_TSDF(X) < 0:
            return -self.get_TSDF(X) + 1 / 2 * self.epsilon
        elif 0 <= self.get_TSDF(X) < self.epsilon:
            return 1 / (2 * self.epsilon) * (self.get_TSDF(X) - self.epsilon) ** 2
        else:
            return 0

    def gradient(self, trajectory):
        return self.smooth_gradient(trajectory) \
               + self.lambda1 * self.safety_gradient(trajectory) \
               + self.lambda2 * self.shot_quality_gradient(trajectory)

    def value(self, trajectory):  # value of the cost at point
        return self.smooth_cost(trajectory) \
               + self.lambda1 * self.safety_cost(trajectory) \
               + self.lambda2 * self.shot_quality_cost(trajectory)

    def smooth_cost(self, trajectory):
        return 1 / (2 * (self.n - 1)) * np.trace(np.transpose(trajectory) @ self.A_smooth @ trajectory
                                                 + 2 * np.transpose(trajectory) @ self.b_smooth
                                                 + self.c_smooth)

    def smooth_gradient(self, trajectory):
        return 1 / (self.n - 1) * (np.matmul(self.A_smooth, trajectory) + self.b_smooth)

    def shot_quality_cost(self, trajectory):
        return 1 / (2 * (self.n - 1)) * np.trace(np.transpose(trajectory) @ self.A_shot @ trajectory
                                                 + 2 * np.transpose(trajectory) @ self.b_shot
                                                 + self.c_shot)

    def shot_quality_gradient(self, trajectory):
        return 1 / (self.n - 1) * (self.A_shot @ trajectory + self.b_shot)

    def safety_cost(self, trajectory):
        velocity_profile = self.estimate_velocity(trajectory)
        J_safe = 0
        for i in range(len(trajectory)):
            J_safe += (self.collision_cost_map(trajectory[i,:]) * np.linalg.norm(velocity_profile[i, :])) * self.delta_t
        return J_safe

    def safety_gradient(self, trajectory):
        velocity_profile = self.estimate_velocity(trajectory)
        acceleration_profile = self.estimate_acceleration(trajectory)
        gradients = []
        for i in range(trajectory.shape[0]):
            p_dot_hat = normalize_vector(velocity_profile[i, :])
            k = 1 / np.linalg.norm(velocity_profile[i, :]) ** 2 * (np.eye(3) - np.outer(p_dot_hat, p_dot_hat)) @ acceleration_profile[i, :]
            gradient = np.linalg.norm(velocity_profile[i,:]) * (
                    (np.eye(3) - np.outer(p_dot_hat, p_dot_hat)) @ self.get_TSDF_gradient(trajectory[i, :]) -
                    self.get_TSDF(trajectory[i, :]) * k)
            gradients.append(gradient)
        return np.vstack(tuple(gradients))

    def estimate_velocity(self, trajectory):
        velocity = np.zeros(trajectory.shape)
        for i in range(1, trajectory.shape[0]):
            velocity[i, :] = (trajectory[i, :] - trajectory[i-1, :])/self.delta_t
        velocity[0, :] = velocity[1, :]
        return velocity

    def estimate_acceleration(self, trajectory):
        velocity = self.estimate_velocity(trajectory)
        acceleration = np.ones(trajectory.shape)
        for i in range(1, trajectory.shape[0]):
            acceleration[i, :] = (velocity[i, :] - velocity[i-1, :])/self.delta_t
        acceleration[0, :] = self.q0[2]
        return acceleration

    def optimize(self):
        for i in range(self.maximum_iter):
            gradient = self.gradient(self.current_traj)
            # np.dot(gradient, self.AA @ gradient) / 2 < self.epsilon0 or
            if abs(self.value(self.current_traj) - self.value(self.last_traj)) < self.epsilon1:
                return
            self.last_traj = copy.deepcopy(self.current_traj)
            self.current_traj = self.current_traj - 1 / self.eta * self.AA @ gradient
            self.history.append(copy.deepcopy(self.current_traj))

