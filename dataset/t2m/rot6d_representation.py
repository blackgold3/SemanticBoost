from os.path import join as pjoin
from dataset.t2m.common.skeleton import Skeleton
import numpy as np
from dataset.t2m.common.quaternion import *
from dataset.t2m.paramUtil import *
import torch
from tqdm import tqdm
import os
from utils.read_from_npy import npy2info, info2dict
from dataset.utils.rotation_conversions import *
from utils.reverse import reverse_pose as swap_left_right

def rot6d_representation(pose_path, joints_path, output_path, joints_num):
    '''
    amass = True 时 rotate = True
    '''
    os.makedirs(output_path, exist_ok=True)
    files = os.listdir(joints_path)
    for i in tqdm(range(len(files))):
        name = files[i]
        curr_pose = os.path.join(pose_path, name)
        curr_joints = os.path.join(joints_path, name)

        pose = np.load(curr_pose)
        pose, trans, gender, betas = npy2info(pose, 10)
        joints = np.load(curr_joints)
        joints = joints[:, :joints_num, :]

        pose = pose.reshape(pose.shape[0], -1, 3)
        pose = torch.from_numpy(pose)
        matrix = axis_angle_to_matrix(pose)
        rot6d = matrix_to_rotation_6d(matrix)[:, :joints_num, :].numpy()

        joints_vel = (joints[1::, :, :] - joints[:-1, :, :])
        rot6d_vel = rot6d[1::, :, :] - rot6d[:-1, :, :]

        nframes = rot6d.shape[0] - 1
        if nframes == 0:
            continue
        trans = trans[:-1]
        rot6d = rot6d[:-1].reshape(nframes, -1)
        joints = joints[:-1].reshape(nframes, -1)
        rot6d_vel = rot6d_vel.reshape(nframes, -1)
        joints_vel = joints_vel.reshape(nframes, -1)
        final_rep = np.concatenate([trans, rot6d, joints, rot6d_vel, joints_vel], axis=1)
        final_path = os.path.join(output_path, name)
        np.save(final_path, final_rep)

if __name__ == "__main__":
    pose_path = "/apdcephfs_cq2/share_1290939/kleinhe/humanml3d/full_pose_data"
    joints_path = "/apdcephfs_cq2/share_1290939/kleinhe/humanml3d/joints"
    output_path = '/apdcephfs_cq2/share_1290939/kleinhe/humanml3d/rot6d'
    device = "cuda"
    joints_num = 22

    rot6d_representation(pose_path, joints_path, output_path, joints_num)
