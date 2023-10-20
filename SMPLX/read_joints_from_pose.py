import torch
import numpy as np
from torch import nn
import pickle as pkl
import torch.nn.functional as F

class Struct(object):
    def __init__(self, **kwargs):
        for key, val in kwargs.items():
            setattr(self, key, val)


def to_np(array, dtype=np.float32):
    if 'scipy.sparse' in str(type(array)):
        array = array.todense()
    return np.array(array, dtype=dtype)


class Get_Joints(nn.Module):
    def __init__(self, path, batch_size=300) -> None:
        super().__init__()
        self.betas = nn.parameter.Parameter(torch.zeros([batch_size, 10], dtype=torch.float32), requires_grad=False)
        with open(path, "rb") as f:
            smpl_prior = pkl.load(f, encoding="latin1")
            data_struct = Struct(**smpl_prior)

        self.v_template = nn.parameter.Parameter(torch.from_numpy(to_np(data_struct.v_template)), requires_grad=False)
        self.shapedirs = nn.parameter.Parameter(torch.from_numpy(to_np(data_struct.shapedirs)), requires_grad=False)
        self.J_regressor = nn.parameter.Parameter(torch.from_numpy(to_np(data_struct.J_regressor)), requires_grad=False)
        posedirs = torch.from_numpy(to_np(data_struct.posedirs))
        num_pose_basis = posedirs.shape[-1]
        posedirs = posedirs.reshape([-1, num_pose_basis]).permute(1, 0)
        self.posedirs = nn.parameter.Parameter(posedirs, requires_grad=False)
        self.parents = nn.parameter.Parameter(torch.from_numpy(to_np(data_struct.kintree_table)[0]).long(), requires_grad=False)
        self.parents[0] = -1

        self.ident = nn.parameter.Parameter(torch.eye(3), requires_grad=False)
        self.K = nn.parameter.Parameter(torch.zeros([1, 3, 3]), requires_grad=False)
        self.zeros = nn.parameter.Parameter(torch.zeros([1, 1]), requires_grad=False)

    def blend_shapes(self, betas, shape_disps):
        blend_shape = torch.einsum('bl,mkl->bmk', [betas, shape_disps])
        return blend_shape

    def vertices2joints(self, J_regressor, vertices):
        return torch.einsum('bik,ji->bjk', [vertices, J_regressor])

    def batch_rodrigues(
        self,
        rot_vecs,
        epsilon = 1e-8,
    ):
        batch_size = rot_vecs.shape[0]
        angle = torch.norm(rot_vecs + epsilon, dim=1, keepdim=True)
        rot_dir = rot_vecs / angle
        cos = torch.unsqueeze(torch.cos(angle), dim=1)
        sin = torch.unsqueeze(torch.sin(angle), dim=1)
        # Bx1 arrays
        rx, ry, rz = torch.split(rot_dir, 1, dim=1)
        K = self.K.repeat(batch_size, 1, 1)
        zeros = self.zeros.repeat(batch_size, 1)
        K = torch.cat([zeros, -rz, ry, rz, zeros, -rx, -ry, rx, zeros], dim=1).view((batch_size, 3, 3))
        ident = self.ident.unsqueeze(0) 
        rot_mat = ident + sin * K + (1 - cos) * torch.bmm(K, K)
        return rot_mat

    def transform_mat(self, R, t):
        return torch.cat([F.pad(R, [0, 0, 0, 1]),
                        F.pad(t, [0, 0, 0, 1], value=1)], dim=2)

    def batch_rigid_transform(
        self,
        rot_mats,
        joints,
        parents,
    ):
        joints = torch.unsqueeze(joints, dim=-1)

        rel_joints = joints.clone()
        rel_joints[:, 1:] -= joints[:, parents[1:]]

        transforms_mat = self.transform_mat(
            rot_mats.reshape(-1, 3, 3),
            rel_joints.reshape(-1, 3, 1)).reshape(-1, joints.shape[1], 4, 4)

        transform_chain = [transforms_mat[:, 0]]
        for i in range(1, parents.shape[0]):
            # Subtract the joint location at the rest pose
            # No need for rotation, since it's identity when at rest
            curr_res = torch.matmul(transform_chain[parents[i]],
                                    transforms_mat[:, i])
            transform_chain.append(curr_res)

        transforms = torch.stack(transform_chain, dim=1)

        # The last column of the transformations contains the posed joints
        posed_joints = transforms[:, :, :3, 3]
        return posed_joints

    def forward(self, pose, trans=None):
        pose = pose.float()
        batch = pose.shape[0]
        betas = self.betas[:batch]
        v_shaped = self.v_template + self.blend_shapes(betas, self.shapedirs)
        J = self.vertices2joints(self.J_regressor, v_shaped)
        rot_mats = self.batch_rodrigues(pose.view(-1, 3)).view([batch, -1, 3, 3])
        J_transformed = self.batch_rigid_transform(rot_mats, J, self.parents)
        if trans is not None:
            J_transformed += trans.unsqueeze(dim=1)
        return J_transformed