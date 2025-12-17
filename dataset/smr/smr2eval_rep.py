import numpy as np
from dataset.t2m.common.skeleton import Skeleton
from dataset.t2m.paramUtil import *
from dataset.t2m.motion_representation import process_file, uniform_skeleton
import torch
from dataset.smr.recover_smr import recover_from_ric
from dataset.t2m.common.quaternion import *

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

l_idx1, l_idx2 = 5, 8
fid_r, fid_l = [8, 11], [7, 10]
face_joint_indx = [2, 1, 17, 16]
n_raw_offsets = torch.from_numpy(t2m_raw_offsets)
kinematic_chain = t2m_kinematic_chain
# Get offsets of target skeleton
example_data = np.load("dataset/t2m/000021.npy")      #### standard skeleton
example_data = example_data.reshape(len(example_data), -1, 3)
example_data = torch.from_numpy(example_data)
tgt_skel = Skeleton(n_raw_offsets, kinematic_chain, 'cpu')
tgt_offsets = tgt_skel.get_offsets_joints(example_data[0])

def smr2t2m(smr_rep, njoints=22):
    bs, nframes, length = smr_rep.shape
    smr_rep = smr_rep.reshape(-1, length)
    if isinstance(smr_rep, np.ndarray):
        smr_rep = torch.from_numpy(smr_rep).float()
    elif isinstance(smr_rep, torch.Tensor):
        smr_rep = smr_rep.float()
    joints = recover_from_ric(smr_rep, njoints)

    if isinstance(joints, torch.Tensor):
        joints = joints.cpu().numpy()

    joints = joints.reshape(bs, nframes, njoints, 3)
    copy = joints[:, joints.shape[1]-1:, :, :]
    joints = np.concatenate([joints, copy], axis=1)
    t2ms = []
    for joint in joints:
        data = process_file(joint, 0.002, tgt_offsets, n_raw_offsets, kinematic_chain, 
                            l_idx1, l_idx2, face_joint_indx, fid_l, fid_r)

        data[np.isnan(data)] == 0.0
        data[np.isinf(data)] == 0.0
        t2ms.append(data)

    t2ms = np.stack(t2ms, axis=0)    
    return t2ms

def face2z(positions, tgt_offsets, n_raw_offsets, kinematic_chain, l_idx1, l_idx2, face_joint_indx):
    '''Uniform Skeleton'''
    positions = uniform_skeleton(positions, tgt_offsets, n_raw_offsets, kinematic_chain, l_idx1, l_idx2, face_joint_indx)

    '''Put on Floor'''
    floor_height = positions.min(axis=0).min(axis=0)[1]
    positions[:, :, 1] -= floor_height

    '''XZ at origin'''
    root_pos_init = positions[0]
    root_pose_init_xz = root_pos_init[0] * np.array([1, 0, 1])
    positions = positions - root_pose_init_xz

    '''All initially face Z+'''
    r_hip, l_hip, sdr_r, sdr_l = face_joint_indx
    across1 = root_pos_init[r_hip] - root_pos_init[l_hip]
    across2 = root_pos_init[sdr_r] - root_pos_init[sdr_l]
    across = across1 + across2
    across = across / np.sqrt((across ** 2).sum(axis=-1))[..., np.newaxis]

    forward_init = np.cross(np.array([[0, 1, 0]]), across, axis=-1)
    forward_init = forward_init / np.sqrt((forward_init ** 2).sum(axis=-1))[..., np.newaxis]

    target = np.array([[0, 0, 1]])
    root_quat_init = qbetween_np(forward_init, target)
    root_quat_init = np.ones(positions.shape[:-1] + (4,)) * root_quat_init

    positions = qrot_np(root_quat_init, positions)
    return positions

def smr_to_eval_joints(smr_rep, njoints=22):
    bs, nframes, length = smr_rep.shape
    if isinstance(smr_rep, np.ndarray):
        smr_rep = torch.from_numpy(smr_rep).float()
    elif isinstance(smr_rep, torch.Tensor):
        smr_rep = smr_rep.float()
    joints = recover_from_ric(smr_rep, njoints)
    new_joints = []
    for demo in joints:
        new_demo = face2z(demo.cpu().numpy(), tgt_offsets, n_raw_offsets, kinematic_chain, l_idx1, l_idx2, face_joint_indx)
        new_joints.append(new_demo)
    new_joints = np.stack(new_joints, axis=0)
    joints = torch.from_numpy(new_joints)
    translation = joints[:, :, 0, :] - joints[:, 0:1, 0, :]     ### [bs, nframes, 3]
    joints -= translation.unsqueeze(2)
    joints = torch.cat([translation.unsqueeze(2), joints], dim=2)   #### [bs, nframes, 23, 3]
    data = joints.reshape(bs, nframes, -1).cpu().numpy()
    return data

def smr_to_root_head_leftarm_vec(smr_rep, njoints=22):
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
