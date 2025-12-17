import numpy as np
from dataset.t2m.common.skeleton import Skeleton
from dataset.t2m.paramUtil import *
from dataset.t2m.motion_representation import process_file
import torch
from dataset.t2m.common.quaternion import *
from dataset.utils.rotation_conversions import *

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

def jp2t2m(jp_rep, njoints=22):
    bs, nframes, length = jp_rep.shape
    if isinstance(jp_rep, np.ndarray):
        jp_rep = torch.from_numpy(jp_rep).float()
    elif isinstance(jp_rep, torch.Tensor):
        jp_rep = jp_rep.float()

    translations = [torch.zeros([bs, 3], device=jp_rep.device, dtype=jp_rep.dtype)]
    final_speed = jp_rep[:, :-1, :3]   ### [nframes-1, 3]
    for i in range(final_speed.shape[1]):
        new_translation = translations[-1] + final_speed[:, i]
        translations.append(new_translation)
    
    translations = torch.stack(translations, dim=1) #### [bs, nframes, 3]
    # translations = jp_rep[:, :, :3]
    
    joints = torch.zeros([bs, nframes, 66], device=jp_rep.device, dtype=jp_rep.dtype)
    relative_joints = jp_rep[:, :, 3:66].reshape(bs, nframes, -1, 3)
    root = translations[:, :, None, :]
    relative_joints += root
    joints[:, :, 3:] = relative_joints.reshape(bs, nframes, -1)
    joints[:, :, :3] = root[:, :, 0, :]

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

def jp2pose(jp_rep):
    if isinstance(jp_rep, np.ndarray):
        jp_rep = torch.from_numpy(jp_rep).float()
    elif isinstance(jp_rep, torch.Tensor):
        jp_rep = jp_rep.float()

    translations = [torch.FloatTensor([0, 0, 0])]
    final_speed = jp_rep[:-1, :3]   ### [nframes-1, 3]
    for i in range(final_speed.shape[0]):
        new_translation = translations[-1] + final_speed[i]
        translations.append(new_translation)
    translations = torch.stack(translations, dim=0)   ### [nframes, 3]
    
    rotation = jp_rep[:, 66:].reshape(jp_rep.shape[0], -1, 6)   ### [nframes, 22, 6]
    rotation = rotation_6d_to_matrix(rotation)
    rotation = matrix_to_axis_angle(rotation)   ### [nframes, 22, 3]
    final_rotation = torch.zeros([rotation.shape[0], 24, 3], dtype=torch.float32)
    final_rotation[:, :22, :] = rotation
    final_rotation = final_rotation.reshape(rotation.shape[0], -1)  ### [nframes, 72]
    motion_tensor = torch.cat([final_rotation, translations], dim=1)  ### [nframes, 75]
    motion_tensor = motion_tensor.reshape(motion_tensor.shape[0], -1, 3)    ### [nframes, 25, 3]
    return motion_tensor
