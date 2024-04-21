import math
import numpy as np
import torch
import scipy

class InertialNav():
    def __init__(self, num_grid_cells, num_orientation_cells, init):
        self.num_grid_cells = num_grid_cells
        self.num_orientation_cells = num_orientation_cells
        if init=="Uniform":
            self.current_probability_map = \
                torch.zeros([1, self.num_orientation_cells, self.num_grid_cells, self.num_grid_cells]) \
                    + \
                (1.0 / (self.num_grid_cells * self.num_grid_cells * self.num_orientation_cells))
        else:
            if init=="Origin":
                self.current_probability_map = \
                    torch.zeros([1, self.num_orientation_cells, self.num_grid_cells, self.num_grid_cells])
                self.current_probability_map[0, 0, int(self.num_grid_cells/2), int(self.num_grid_cells/2)] = 1.0
            else:
                raise("Unknown init")
        self.position_kernel = np.zeros([11,11])
        self.orientation_kernel = np.zeros([self.num_orientation_cells])
        self.position_kernel[5,5] = 1.0
        self.orientation_kernel[0] = 1.0
        self.world_grid_length = 3.0
        self.world_cell_size = self.world_grid_length/self.num_grid_cells
        forward = torch.distributions.normal.Normal(1.0, 0.1).sample([1000])
        half_angle_range = math.pi/self.num_orientation_cells
        angles = [torch.distributions.uniform.Uniform(2 * math.pi * a / self.num_orientation_cells - half_angle_range, 2 * math.pi * a / self.num_orientation_cells + half_angle_range).sample([1000]) for a in range(self.num_orientation_cells)]
        x_points = [[-forward * f * torch.cos(angles[a]) + torch.distributions.uniform.Uniform(-.5,+.5).sample([1000]) for a in range(self.num_orientation_cells)] for f in range(-5, 6)]
        y_points = [[-forward * f * torch.sin(angles[a]) + torch.distributions.uniform.Uniform(-.5,+.5).sample([1000]) for a in range(self.num_orientation_cells)] for f in range(-5, 6)]
        points = [[torch.stack([-y_points[f][a], x_points[f][a]], dim=1) for a in range(self.num_orientation_cells)] for f in range(11)]
        hist = [[torch.histogramdd(points[f][a], bins=[11,11], range=[-5.5,+5.5,-5.5,+5.5])[0] for a in range(self.num_orientation_cells)] for f in range(11)]
        self.density = [[hist[f][a]/1000.0 for a in range(self.num_orientation_cells)] for f in range(11)]
        self.identity = torch.eye(self.num_orientation_cells)

        rot_angles = torch.distributions.normal.Normal(1.0, 0.05).sample([1000])
        rot_points = [torch.remainder(2 * math.pi + (-rot_angles * a * 2 * math.pi/self.num_orientation_cells) + angles[0] + half_angle_range, 2*math.pi) for a in range(self.num_orientation_cells)]
        self.angle_hist = [torch.histogram(rot_points[a], bins=self.num_orientation_cells, range=[0, 2*math.pi])[0]/1000 for a in range(self.num_orientation_cells)]
        self.angle_mat = [ torch.stack([torch.roll(self.angle_hist[a], s) for s in range(self.num_orientation_cells)]) for a in range(-5,6)]
        self.angle_conv = [ self.angle_mat[a].reshape(self.num_orientation_cells, self.num_orientation_cells, 1, 1) for a in range(11)]
        print("Finished init")


    def inertial_update(self, last_inertial_position, current_inertial_position, last_inertial_orientation, current_inertial_orientation):
        s = self.current_probability_map * 0.0
        inertial_position_difference = current_inertial_position - last_inertial_position
        inertial_orientation_difference = (current_inertial_orientation - last_inertial_orientation)
#        inertial_orientation_difference = (current_inertial_orientation + 2 * math.pi) % (2 * math.pi) - (last_inertial_orientation + 2 * math.pi) % (
#                    2 * math.pi)
        print("Init in diff", inertial_orientation_difference)
        if inertial_orientation_difference > math.pi:
            inertial_orientation_difference = inertial_orientation_difference - 2*math.pi
        if inertial_orientation_difference < -math.pi:
            inertial_orientation_difference = inertial_orientation_difference + 2*math.pi
        print("Final Diff = ", inertial_orientation_difference)
        tensor_orientation = torch.tensor(current_inertial_orientation)
        inertial_forward = torch.inner(torch.stack([tensor_orientation.cos(), tensor_orientation.sin()]), inertial_position_difference)
        inertial_forward_cell = inertial_forward / self.world_cell_size
        old = self.current_probability_map
        d = torch.floor(inertial_forward_cell).int()
        s1 = torch.conv2d(self.current_probability_map, torch.stack(self.density[d+5], dim=0).unsqueeze(1), padding="same", groups=self.num_orientation_cells)
        s2 = torch.conv2d(self.current_probability_map,
                          torch.stack(self.density[d+5+1], dim=0).unsqueeze(1), padding="same",
                          groups=self.num_orientation_cells)
        fract = inertial_forward_cell - d
        s = s1*(1-fract) + s2*fract
        inertial_orientation_cell = inertial_orientation_difference * self.num_orientation_cells / (2*math.pi)
#        kw = torch.roll(self.identity, -round(inertial_orientation_cell))
#        s = torch.conv2d(s, kw.reshape(self.num_orientation_cells, self.num_orientation_cells, 1, 1))
        r = math.floor(inertial_orientation_cell)
        os1 = torch.conv2d(s, self.angle_conv[r+5])
        os2 = torch.conv2d(s, self.angle_conv[r+5+1])
        ofract = inertial_orientation_cell - r
        os = os1*(1-ofract) + os2*ofract
        self.current_probability_map = os
        self.current_probability_map = torch.clip(self.current_probability_map, min=0.0)
        self.current_probability_map = self.current_probability_map / self.current_probability_map.sum()
        if inertial_position_difference.norm() > .001 or abs(inertial_orientation_difference) > .02:
            print("inertial or diff (rads)=", inertial_orientation_difference, " cell diff=", inertial_orientation_cell, " r=", r)
#            print("CP=", self.current_probability_map[0, :, 50,50].detach().numpy())
            return True
        else:
            return False

    def update_from_sensor(self, update_from_sensor):
        self.current_probability_map *= update_from_sensor.permute(2,0,1).unsqueeze(0)
        self.current_probability_map = torch.clip(self.current_probability_map, min=0.0)
        self.current_probability_map = self.current_probability_map / self.current_probability_map.sum()

    def getpmap(self):
        return self.current_probability_map[0].permute(1, 2, 0)