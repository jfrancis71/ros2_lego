import numpy as np
import torch
import scipy

class InertialNav():
    def __init__(self, num_grid_cells, num_orientation_cells, init):
        self.num_grid_cells = num_grid_cells
        self.num_orientation_cells = num_orientation_cells
        if init=="Uniform":
            self.current_probability_map = \
                torch.zeros([self.num_grid_cells, self.num_grid_cells, self.num_orientation_cells]) \
                    + \
                (1.0 / (self.num_grid_cells * self.num_grid_cells * self.num_orientation_cells))
        else:
            if init=="Origin":
                self.current_probability_map = \
                    torch.zeros([self.num_grid_cells, self.num_grid_cells, self.num_orientation_cells])
                self.current_probability_map[int(self.num_grid_cells/2), int(self.num_grid_cells/2), 0] = 1.0
            else:
                raise("Unknown init")
        self.position_kernel = np.zeros([11,11])
        self.orientation_kernel = np.zeros([self.num_orientation_cells])
        self.position_kernel[5,5] = 1.0
        self.orientation_kernel[0] = 1.0
        self.world_grid_length = 3.0
        self.world_cell_size = self.world_grid_length/self.num_grid_cells

    def inertial_update(self, last_inertial_position, current_inertial_position, last_inertial_orientation, current_inertial_orientation):
        s = self.current_probability_map * 0.0
        inertial_position_difference = current_inertial_position - last_inertial_position
        inertial_orientation_difference = (current_inertial_orientation - last_inertial_orientation)
        tensor_orientation = torch.tensor(current_inertial_orientation)
        inertial_forward = torch.inner(torch.stack([tensor_orientation.cos(), tensor_orientation.sin()]), inertial_position_difference)
        for r in range(self.num_orientation_cells):
            shifted = scipy.ndimage.shift(self.position_kernel, (0.0, -inertial_forward/self.world_cell_size), output=None, order=3, mode='constant', cval=0.0, prefilter=True)
            rotated = scipy.ndimage.rotate(shifted, 360 * r/self.num_orientation_cells, reshape=False)
            s[:,:,r] = torch.nn.functional.conv2d(self.current_probability_map[:,:,r:r+1].permute([2,0,1]), torch.tensor(rotated, dtype=torch.float).reshape([1,1,11,11]), padding="same").permute([1,2,0])[:,:,0]
        s = torch.tensor(
            scipy.ndimage.shift(s, (0,0, inertial_orientation_difference * self.num_orientation_cells/ (2*3.141)), mode='wrap'),
            dtype=torch.float)
        self.current_probability_map = s
        self.current_probability_map = torch.clip(self.current_probability_map, min=0.0)
        self.current_probability_map = self.current_probability_map / self.current_probability_map.sum()
        if inertial_position_difference.norm() > .001 or abs(inertial_orientation_difference) > .001:
            return True
        else:
            return False

    def update_from_sensor(self, update_from_sensor):
        self.current_probability_map *= update_from_sensor
        self.current_probability_map = torch.clip(self.current_probability_map, min=0.0)
        self.current_probability_map = self.current_probability_map / self.current_probability_map.sum()
