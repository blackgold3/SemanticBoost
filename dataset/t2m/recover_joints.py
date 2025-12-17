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
from dataset.utils.rotation_conversions import *

base_vec = torch.Tensor([[0, 0], [0, 1], [1, 0], [0, -1], [-1, 0], [1, 1], [1, -1], [-1, 1], [-1, -1]])
base_vec = base_vec / torch.norm(base_vec, p=2, dim=1).unsqueeze(1)      #### [9, 2]
base_vec[torch.isnan(base_vec)] = 0.0

base_body = torch.Tensor([[0, 0, 1]]).repeat(15, 1)     #### [15, 3]
base_head = torch.Tensor([[0, 0, 1], [-1, 0, 0], [1, 0, 0], [-1, 0, 1], [1, 0, 1], [0, 1, 1], [-1, 1, 0], [1, 1, 0], [-1, 1, 1], [1, 1, 1],
                        [0, -1, 1], [-1, -1, 0], [1, -1, 0], [-1, -1, 1], [1, -1, 1]])  ### [15, 3]
base_head = base_head / torch.norm(base_head, p=2, dim=1).unsqueeze(1)      #### [15, 3]
base_head[torch.isnan(base_head)] = 0.0
base_rela = base_head - base_body

base_body = torch.Tensor([[0, 0, 1]]).repeat(17, 1)     #### [17, 3]
base_leftarm = torch.Tensor([[0, 0, 0], [1, 0, 0], [-1, 0, 0], [0, 0, 1], [0, 0, -1], [1, 0, 1], [1, 0, -1], [-1, 0, 1], [-1, 0, -1],
                        [1, 1, 0], [-1, 1, 0], [0, 1, 1], [0, 1, -1], [1, 1, 1], [1, 1, -1], [-1, 1, 1], [-1, 1, -1]])  ### [17, 3]
base_leftarm = base_leftarm / torch.norm(base_leftarm, p=2, dim=1).unsqueeze(1)      #### [17, 3]
base_leftarm[torch.isnan(base_leftarm)] = 0.0
base_relf = base_leftarm - base_body

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

def t2m_to_root_head_leftarm_vec(smr_rep, njoints=22):
    bs, nframes, length = smr_rep.shape
    if isinstance(smr_rep, np.ndarray):
        smr_rep = torch.from_numpy(smr_rep).float()
    elif isinstance(smr_rep, torch.Tensor):
        smr_rep = smr_rep.float()
    joints = recover_from_ric(smr_rep, njoints)     ##### [bs, nframes, njoints, 3]

    ######## translation #############
    translation = joints[:, :, 0, :] - joints[:, 0:1, 0, :]   #### [bs, nframes, 3] ---> [bs, 9]
    mask = torch.abs(translation) < 0.2
    translation[mask] = 0
    translation = translation[:, :, [0, 2]]    
    translation = translation / torch.norm(translation, p=2, dim=2).unsqueeze(2)   #### [bs, nframes, 2]
    translation[torch.isnan(translation)] = 0.0
    translation = translation.unsqueeze(2).repeat(1, 1, 9, 1)   ##### [bs, nframes, 9, 2]
    curr_base = base_vec.to(translation.device).to(translation.dtype).unsqueeze(0).unsqueeze(0).repeat(bs, nframes, 1, 1)    
    distance = (translation - curr_base) ** 2
    distance = distance.sum(dim=3).argmin(dim=2)      #### [bs, nframes]
    trans_stat = torch.zeros([bs, 9]).to(translation.device).to(translation.dtype)
    for i in range(9):
        mask = distance == i
        mask = mask.sum(dim=1)      #### [bs]
        trans_stat[:, i] += mask 
    
    ############# head ############
    lshoulder = joints[:, :, 16]    #### [bs, nframes, 3]
    rshoulder =joints[:, :, 17]  
    neck = joints[:, :, 12]
    head = joints[:, :, 15]

    body_direct = torch.cross(rshoulder - neck, lshoulder - neck, dim=2) 
    body_direct = body_direct / torch.norm(body_direct, p=2, dim=2).unsqueeze(2)
    body_direct[torch.isnan(body_direct)] = 0.0     
    
    mshoulder = (lshoulder + rshoulder) / 2
    ms2n = mshoulder - neck
    h2n = head - neck
    head_direct = h2n * 0.56 + ms2n * 0.44
    head_direct = head_direct / torch.norm(head_direct, p=2, dim=2).unsqueeze(2)
    head_direct[torch.isnan(head_direct)] = 0.0     ### [bs, nframes, 3]  -> [bs, 15]
    rela_direct = head_direct - body_direct
    rela_direct = rela_direct.unsqueeze(2).repeat(1, 1, 15, 1)  ### [bs, nframes, 15, 3]
    curr_rela = base_rela.to(rela_direct.device).to(rela_direct.dtype).unsqueeze(0).unsqueeze(0).repeat(bs, nframes, 1, 1)    
    distance = (rela_direct - curr_rela) ** 2
    distance = distance.sum(dim=3).argmin(dim=2)      #### [bs, nframes]
    head_stat = torch.zeros([bs, 15]).to(rela_direct.device).to(rela_direct.dtype)
    for i in range(15):
        mask = distance == i
        mask = mask.sum(dim=1)      #### [bs]
        head_stat[:, i] += mask 

    reference = joints[:, :, 0, :]
    left_arms = joints[:, :, 18, :]
    left_wrists = joints[:, :, 20, :]

    left_arms = (left_arms + left_wrists) / 2
    rs2ls = (lshoulder - rshoulder) * 0.65
    reference = reference + rs2ls
    arm_direct = left_arms - reference
    arm_direct = arm_direct / torch.norm(arm_direct, p=2, dim=2).unsqueeze(2)
    arm_direct[torch.isnan(arm_direct)] = 0.0     ### [bs, nframes, 3]  -> [bs, 15]
    rela_arm = arm_direct - body_direct
    rela_arm = rela_arm.unsqueeze(2).repeat(1, 1, 17, 1)  ### [bs, nframes, 15, 3]
    curr_arm = base_relf.to(rela_arm.device).to(rela_arm.dtype).unsqueeze(0).unsqueeze(0).repeat(bs, nframes, 1, 1)    
    distancea_arm = (rela_arm - curr_arm) ** 2
    distancea_arm = distancea_arm.sum(dim=3).argmin(dim=2)      #### [bs, nframes]
    arm_stat = torch.zeros([bs, 17]).to(rela_arm.device).to(rela_arm.dtype)
    for i in range(17):
        mask = distancea_arm == i
        mask = mask.sum(dim=1)      #### [bs]
        arm_stat[:, i] += mask 


    return trans_stat, head_stat, arm_stat
