import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pytorch3d.ops import ball_query

from . import patchnet as pn
from .utils.common import cal_Z_axis, l2_norm, RodsRotatFormula, sphere_query, var_to_invar, get_voxel_coordinate

class MiniSpinNet(nn.Module):
    def __init__(
        self,
        des_r: float = 3.0,
        num_points_per_patch: int = 512,
        rad_n: int = 3,
        azi_n: int = 20,
        ele_n: int = 7,
        delta: float = 0.8,
        voxel_sample: int = 10,
        is_aligned_to_global_z: bool = True,
    ):
        super(MiniSpinNet, self).__init__()
        self.des_r = des_r
        self.patch_sample = num_points_per_patch
        self.rad_n = rad_n
        self.azi_n = azi_n
        self.ele_n = ele_n
        self.delta = delta
        self.voxel_sample = voxel_sample
        self.is_aligned_to_global_z = is_aligned_to_global_z
        self.pnt_layer = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=(1, 1), stride=(1, 1)),
            nn.BatchNorm2d(16),
            nn.ReLU(True),
        )

        self.pool_layer = nn.Sequential(
            nn.Conv2d(32, 16, kernel_size=(1, 1), stride=(1, 1)),
            nn.BatchNorm2d(16),
            nn.ReLU(True),
            nn.Conv2d(16, 1, kernel_size=(1, 1), stride=(1, 1)),
            nn.BatchNorm2d(1),
            nn.ReLU(True),
        )

        self.conv_net = pn.Cylindrical_Net(inchan=16, dim=32)
        # self.conv_net = pn.Cylindrical_UNet(inchan=16, dim=32)

    def forward(self, pts, kpts, des_r, is_aligned_to_global_z=True, z_axis=None, is_aug=False):
        
        # extract patches
        init_patch = self.select_patches(pts, kpts, vicinity=des_r, patch_sample=self.patch_sample)
        init_patch = init_patch.squeeze(0)

        # print('init_patch:', init_patch) 
        # print('kpts:', kpts)

        # align with reference axis
        patches, rand_axis, R = self.axis_align(init_patch, is_aligned_to_global_z, z_axis)
        patches = self.normalize(patches, des_r)

        # print('patches:', patches) 

        # by default, we do not apply any SO(2) rotation augmentation
        aug_rotation = np.eye(3)[None].repeat(patches.shape[0], axis=0)
        aug_rotation = torch.FloatTensor(aug_rotation).to(patches.device)
        patches = patches @ aug_rotation.transpose(-1, -2)
        rand_axis = (rand_axis.unsqueeze(1) @ aug_rotation.transpose(-1, -2)).squeeze(1)

        # spatial point transformer
        inv_patches = self.SPT(patches, 1, self.delta / self.rad_n)

        # Vanilla SpinNet
        new_points = inv_patches.permute(0, 3, 1, 2)  # (B, C_in, npoint, nsample+1), input features
        x = self.pnt_layer(new_points)
        x = F.max_pool2d(x, kernel_size=(1, x.shape[-1])).squeeze(3)  # (B, C_in, npoint)
        del new_points
        x = x.view(x.shape[0], x.shape[1], self.rad_n, self.ele_n, self.azi_n)
        x, mid = self.conv_net(x)

        w = self.pool_layer(x)
        f = F.avg_pool2d(x * w, kernel_size=(x.shape[2], x.shape[3]))
        f = F.normalize(f.view(f.shape[0], -1), p=2, dim=1)
        x = F.normalize(x, p=2, dim=1)

        return {'desc': f,
                'equi': x,
                'rand_axis': rand_axis,
                'R': R,
                'patches': patches,
                'aug_rotation': aug_rotation}

    def select_patches(self, pts, refer_pts, vicinity, patch_sample=1024):
        # pts: (B, N, 3)
        # refer_pts: (B, K, 3), key points
        B, N, C = pts.shape

        # shuffle pts if pts is not orderless
        index = np.random.choice(N, N, replace=False)
        pts = pts[:, index]

        # Use PyTorch3D's ball_query instead of pnt2.ball_query and pnt2.grouping_operation
        # ball_query returns (dists, idx, nn) where nn contains the neighbor points
        dists, group_idx, new_points = ball_query(
            p1=refer_pts,  # query points (centers)
            p2=pts,        # points to search in
            K=patch_sample,  # maximum number of neighbors
            radius=vicinity,  # radius within which to search
            return_nn=True    # return the neighbor points directly
        )
        # dists as the squared distance
        # Sort distances in decreasing order for each reference point
        # sorted_dists, _ = torch.sort(dists, dim=-1, descending=True)
        # print('sorted_dists.shape:', sorted_dists.shape)
        # print('sorted_dists:', sorted_dists)
        
        # new_points from ball_query has shape (B, P1, K, D) where D is the point dimension
        # We want it in shape (B, P1, K, D) to match the original format
        # No permute needed since ball_query already returns the correct format
        
        # Create mask for invalid neighbors (where group_idx == -1)
        invalid_mask = (group_idx == -1).float()  # 1 where no valid neighbor found
        
        # Expand masks to match coordinate dimensions
        invalid_mask = invalid_mask.unsqueeze(3).repeat([1, 1, 1, C])
        
        # Create reference points repeated for each patch position
        new_pts = refer_pts.unsqueeze(2).repeat([1, 1, patch_sample, 1])
        
        # Fill invalid positions with reference points, keep valid neighbors as they are
        local_patches = new_points * (1 - invalid_mask) + new_pts * invalid_mask

        del invalid_mask
        del new_points
        del group_idx
        del new_pts
        del pts

        return local_patches

    def axis_align(self, input, is_aligned_to_global_z, z_axis=None):
        center = input[:, -1, :3]
        delta_x = input[:, :, :3] - center.unsqueeze(1)  # (B, npoint, 3), normalized coordinates        
        
        if not is_aligned_to_global_z:
            if z_axis is None:
                z_axis = cal_Z_axis(delta_x, ref_point=center)
                z_axis = l2_norm(z_axis, axis=1)
            else:
                z_axis = z_axis[0]
            R = RodsRotatFormula(z_axis, torch.FloatTensor([0, 0, 1]).expand_as(z_axis))
            delta_x = torch.matmul(delta_x, R)

            # for calculate gt lable
            rand_axis = torch.zeros_like(center)
            rand_axis[:, -1] = 1
            rand_axis = torch.cross(z_axis, rand_axis)
            rand_axis = F.normalize(rand_axis, p=2, dim=-1)

        else:
            rand_axis = torch.zeros_like(center)
            rand_axis[:, 0] = 1
            R = torch.eye(3).to(center.device)
            R = R[None].repeat([center.shape[0], 1, 1])

        return delta_x, rand_axis, R

    def SPT(self, delta_x, des_r, voxel_r):

        # partition the local surface along elevator, azimuth, radial dimensions
        S2_xyz = torch.FloatTensor(get_voxel_coordinate(radius=des_r,
                                                                     rad_n=self.rad_n,
                                                                     azi_n=self.azi_n,
                                                                     ele_n=self.ele_n))

        pts_xyz = S2_xyz.view(1, -1, 3).repeat([delta_x.shape[0], 1, 1]).cuda()
        # query points in sphere
        new_points = sphere_query(delta_x, pts_xyz, radius=voxel_r,
                                               nsample=self.voxel_sample)
        # transform rotation-variant coords into rotation-invariant coords
        new_points = var_to_invar(new_points, self.rad_n, self.azi_n, self.ele_n)

        return new_points

    def normalize(self, pts, radius):
        delta_x = pts / (torch.ones_like(pts).to(pts.device) * radius)

        return delta_x

    def get_parameter(self):
        return list(self.parameters())

