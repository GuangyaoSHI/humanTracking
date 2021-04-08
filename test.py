import binvox_rw
import numpy as np

# with open('map2.binvox', 'rb') as f:
#     model = binvox_rw.read_as_3d_array(f)
#
# # the point in airsim coordinate that generate voxel maps
# center = np.array([0, 0, 0])
# # resolution of each voxel
# res = 0.5
# # origin of voxel in airsim coordinate
# voxel_origin_x = -int(model.dims[0] / 2) * res + center[0]
# voxel_origin_y = -int(model.dims[1] / 2) * res + center[1]
# voxel_origin_z = -int(model.dims[2] / 2) * res + center[2]
# voxel_origin = (voxel_origin_x, voxel_origin_y, voxel_origin_z)
#
# tsdf = np.zeros(model.dims + [4])

import multiprocessing
import time


class TestPool():
    def __init__(self):
        indices = []
        for i in range(1000):
            for j in range(1000):
                for k in range(1000):
                    indices.append((i, j, k))
        start_time = time.time()
        p = multiprocessing.pool(multiprocessing.cpu_count())
        p.map(self.calcaulte, indices)
        p.close()
        p.join()
        print('time consumed is {}'.format(time.time() - start_time))

    def calculate(self, index):
        return index[0] * index[1] * index[2]


# https://zetcode.com/python/multiprocessing/
# https://stackoverflow.com/questions/9786102/how-do-i-parallelize-a-simple-python-loop