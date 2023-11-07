import torch
# Recover global angle and positions for rotation data
# root_rot_velocity (B, seq_len, 1)
# root_linear_velocity (B, seq_len, 2)
# root_y (B, seq_len, 1)
# ric_data (B, seq_len, (joint_num - 1)*3)
# rot_data (B, seq_len, (joint_num - 1)*6)
# local_velocity (B, seq_len, joint_num*3)
# foot contact (B, seq_len, 4)
import numpy as np
from SMPLX.rotation_conversions import *

def qinv(q):
    assert q.shape[-1] == 4, 'q must be a tensor of shape (*, 4)'
    mask = torch.ones_like(q)
    mask[..., 1:] = -mask[..., 1:]
    return q * mask

def qrot(q, v):
    """
    Rotate vector(s) v about the rotation described by quaternion(s) q.
    Expects a tensor of shape (*, 4) for q and a tensor of shape (*, 3) for v,
    where * denotes any number of dimensions.
    Returns a tensor of shape (*, 3).
    """
    assert q.shape[-1] == 4
    assert v.shape[-1] == 3
    assert q.shape[:-1] == v.shape[:-1]

    original_shape = list(v.shape)
    # print(q.shape)
    q = q.contiguous().view(-1, 4)
    v = v.contiguous().view(-1, 3)

    qvec = q[:, 1:]
    uv = torch.cross(qvec, v, dim=1)
    uuv = torch.cross(qvec, uv, dim=1)
    return (v + 2 * (q[:, :1] * uv + uuv)).view(original_shape)


def recover_root_rot_pos(data):
    rot_vel = data[..., 0]
    r_rot_ang = torch.zeros_like(rot_vel).to(data.device)
    '''Get Y-axis rotation from rotation velocity'''
    r_rot_ang[..., 1:] = rot_vel[..., :-1]
    r_rot_ang = torch.cumsum(r_rot_ang, dim=-1)

    r_rot_quat = torch.zeros(data.shape[:-1] + (4,)).to(data.device)
    r_rot_quat[..., 0] = torch.cos(r_rot_ang)
    r_rot_quat[..., 2] = torch.sin(r_rot_ang)

    r_pos = torch.zeros(data.shape[:-1] + (3,)).to(data.device)
    r_pos[..., 1:, [0, 2]] = data[..., :-1, 1:3]
    '''Add Y-axis rotation to root position'''
    r_pos = qrot(qinv(r_rot_quat), r_pos)

    r_pos = torch.cumsum(r_pos, dim=-2)

    r_pos[..., 1] = data[..., 3]
    return r_rot_quat, r_pos

def quaternion_to_matrix(quaternions):
    """
    Convert rotations given as quaternions to rotation matrices.
    Args:
        quaternions: quaternions with real part first,
            as tensor of shape (..., 4).
    Returns:
        Rotation matrices as tensor of shape (..., 3, 3).
    """
    r, i, j, k = torch.unbind(quaternions, -1)
    two_s = 2.0 / (quaternions * quaternions).sum(-1)

    o = torch.stack(
        (
            1 - two_s * (j * j + k * k),
            two_s * (i * j - k * r),
            two_s * (i * k + j * r),
            two_s * (i * j + k * r),
            1 - two_s * (i * i + k * k),
            two_s * (j * k - i * r),
            two_s * (i * k - j * r),
            two_s * (j * k + i * r),
            1 - two_s * (i * i + j * j),
        ),
        -1,
    )
    return o.reshape(quaternions.shape[:-1] + (3, 3))

def quaternion_to_cont6d(quaternions):
    rotation_mat = quaternion_to_matrix(quaternions)
    cont_6d = torch.cat([rotation_mat[..., 0], rotation_mat[..., 1]], dim=-1)
    return cont_6d

def recover_from_rot(data, joints_num, skeleton):
    r_rot_quat, r_pos = recover_root_rot_pos(data)

    r_rot_cont6d = quaternion_to_cont6d(r_rot_quat)

    start_indx = 1 + 2 + 1 + (joints_num - 1) * 3
    end_indx = start_indx + (joints_num - 1) * 6
    cont6d_params = data[..., start_indx:end_indx]
    #     print(r_rot_cont6d.shape, cont6d_params.shape, r_pos.shape)
    cont6d_params = torch.cat([r_rot_cont6d, cont6d_params], dim=-1)
    cont6d_params = cont6d_params.view(-1, joints_num, 6)

    positions = skeleton.forward_kinematics_cont6d(cont6d_params, r_pos)

    return positions


def recover_from_ric(data, joints_num):
    if isinstance(data, np.ndarray):
        data = torch.from_numpy(data).float()
        dtype = "numpy"
    else:
        data = data.float()
        dtype = "tensor"

    r_rot_quat, r_pos = recover_root_rot_pos(data)
    positions = data[..., 4:(joints_num - 1) * 3 + 4]
    positions = positions.view(positions.shape[:-1] + (-1, 3))

    '''Add Y-axis rotation to local joints'''
    positions = qrot(qinv(r_rot_quat[..., None, :]).expand(positions.shape[:-1] + (4,)), positions)

    '''Add root XZ to joints'''
    positions[..., 0] += r_pos[..., 0:1]
    positions[..., 2] += r_pos[..., 2:3]

    '''Concate root and joints'''
    positions = torch.cat([r_pos.unsqueeze(-2), positions], dim=-2)

    if dtype == "numpy":
        positions = positions.numpy()

    return positions

def t2m_to_eval_rep(data, joint_num=22):
    bs, nframes, length = data.shape
    if isinstance(data, np.ndarray):
        data = torch.from_numpy(data).float()
    elif isinstance(data, torch.Tensor):
        data = data.float()
    joints = recover_from_ric(data, joint_num)
    translation = joints[:, :, 0, :] - joints[:, 0:1, 0, :]     ### [bs, nframes, 3]

    joints -= translation.unsqueeze(2)
    joints = torch.cat([translation.unsqueeze(2), joints], dim=2)   #### [bs, nframes, 23, 3]
    data = joints.reshape(bs, nframes, -1).cpu().numpy()
    return data

def recover_pose_from_t2m(data, njoints=22):
    joints = recover_from_ric(data, njoints)    
    trans = joints[:, 0, :] - joints[0:1, 0, :]

    pose = data[:, 4 + (njoints - 1) * 3:4 + (njoints - 1) * 9]
    pose = pose.reshape(pose.shape[0], njoints-1, 6)
    ptype = type(pose)
    if ptype == np.ndarray:
        pose = torch.from_numpy(pose).float()
        pose = rotation_6d_to_matrix(pose)
        pose = matrix_to_axis_angle(pose)
        pose = pose.numpy()
        root_vel = np.zeros([pose.shape[0], 1, 3])
        pose = np.concatenate([root_vel, pose], axis=1)
    elif ptype == torch.Tensor:
        pose = rotation_6d_to_matrix(pose)
        pose = matrix_to_axis_angle(pose)
        root_vel = torch.zeros([pose.shape[0], 1, 3])
        pose = torch.cat([root_vel, pose], dim=1) 

    pose = pose.reshape(pose.shape[0], -1)

    if njoints < 24:
        if ptype == np.ndarray:
            addition = np.zeros([pose.shape[0], 72-njoints*3])
            pose = np.concatenate([pose, addition], axis=1)
        elif ptype == torch.Tensor:
            addition = torch.zeros([pose.shape[0], 72-njoints*3], dtype=pose.dtype, device=pose.device)
            pose = torch.cat([pose, addition], dim=1)

    if ptype == np.ndarray:
        pose = np.concatenate([pose, trans], axis=1)
    elif ptype == torch.Tensor:
        pose = torch.cat([pose, trans], dim=1)

    return pose
