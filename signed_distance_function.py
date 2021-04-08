# https://www.reddit.com/r/computervision/comments/6f0oln/need_help_understanding_tsdf_algorithm_from/
# https://oru.diva-portal.org/smash/get/diva2:1136113/FULLTEXT01.pdf
# https://github.com/Microsoft/AirSim/issues/1800
# https://www.groundai.com/project/vision-based-autonomous-landing-in-catastrophe-struck-environments/1#bib.bib23
# https://arxiv.org/pdf/1611.03631.pdf

import binvox_rw
import numpy as np
import multiprocessing

class TSDF():
    def __init__(self, filename='map2.binvox'):
        with open(filename, 'rb') as f:
            self.model = binvox_rw.read_as_3d_array(f)
        # the point in airsim coordinate that generate voxel maps
        self.center = np.array([0, 0, 0])
        # resolution of each voxel
        self.res = 0.5
        # origin of voxel in airsim coordinate
        voxel_origin_x = -int(self.model.dims[0] / 2) * self.res + self.center[0]
        voxel_origin_y = -int(self.model.dims[1] / 2) * self.res + self.center[1]
        voxel_origin_z = -int(self.model.dims[2] / 2) * self.res + self.center[2]
        self.voxel_origin = (voxel_origin_x, voxel_origin_y, voxel_origin_z)
        # coordinates of all voxels in airsim coordinate
        self.coordinates = np.zeros(self.model.dims + [3])
        self.tsdf = np.zeros(self.model.dims + [4])
        for i in range(self.model.dims[0]):
            for j in range(self.model.dims[1]):
                for k in range(self.model.dims[2]):
                    x = (i - int(self.model.dims[0] / 2)) * self.res + self.center[0]
                    y = (j - int(self.model.dims[1] / 2)) * self.res + self.center[1]
                    z = (k - int(self.model.dims[2] / 2)) * self.res + self.center[2]
                    self.coordinates[i, j, k, :] = np.array([x, y, z])
        p = multiprocessing.Pool(multiprocessing.cpu_count())

        indices = []
        for i in range(self.model.dims[0]):
            for j in range(self.model.dims[1]):
                for k in range(self.model.dims[2]):
                    indices.append((i, j, k))
        p.map(self.calculate_tsdf, indices)
        p.close()
        p.join()

    def nearest_obstacle(self, point):
        obstacle_n = (0, 0, 0)
        distance = np.linalg.norm(np.array(point) - np.array(obstacle_n))
        for i in range(self.model.dims[0]):
            for j in range(self.model.dims[1]):
                for k in range(self.model.dims[2]):
                    if self.model.data[i, j, k] and point != (i, j, k):
                        distance_new = np.linalg.norm(np.array(point) - np.array([i, j, k]))
                        if distance_new < distance:
                            obstacle_n = (i, j, k)
                            distance = distance_new
        return obstacle_n, distance

    def nearest_free(self, point):
        free = (0, 0, 0)
        distance = np.linalg.norm(np.array(point) - np.array(free))
        for i in range(self.model.dims[0]):
            for j in range(self.model.dims[1]):
                for k in range(self.model.dims[2]):
                    if (not self.model.data[i, j, k]) and point != (i, j, k):
                        distance_new = np.linalg.norm(np.array(point) - np.array([i, j, k]))
                        if distance_new < distance:
                            free = (i, j, k)
                            distance = distance_new
        return free, distance

    def calculate_tsdf(self, index):
        i = index[0]
        j = index[1]
        k = index[2]
        if self.model.data[i, j, k]:
            free, distance = self.nearest_free((i, j, k))
            gradient = self.coordinates[free[0], free[1], free[2]] - self.coordinates[i, j, k]
            self.tsdf[i, j, k, :] = np.array([-distance] + list(gradient))
        else:
            obstacle, distance = self.nearest_obstacle((i, j, k))
            gradient = self.coordinates[i, j, k] - self.coordinates[obstacle[0], obstacle[1], obstacle[2]]
            self.tsdf[i, j, k, :] = np.array([distance] + list(gradient))

    def get_tsdf(self, q):
        # q = (x, y, z) in airsim coordinate
        assert q[0] >= self.voxel_origin[0]
        i = np.ceil((q[0] - self.voxel_origin[0]) / self.res)
        assert q[1] >= self.voxel_origin[1]
        j = np.ceil((q[1] - self.voxel_origin[1]) / self.res)
        assert q[2] >= self.voxel_origin[2]
        k = np.ceil((q[2] - self.voxel_origin[2]) / self.res)
        return self.tsdf[i, j, k, :]
