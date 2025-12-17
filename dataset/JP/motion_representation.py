from dataset.smr.common.skeleton import Skeleton
import numpy as np
from dataset.smr.common.quaternion import *
from dataset.smr.paramUtil import *
import torch
from tqdm import tqdm
import os
from dataset.utils.reverse import reverse_pose
from dataset.utils.rotation_conversions import *
from SMPLX import smplx
from dataset.utils.read_from_npy import npy2info, info2dict

@torch.no_grad()
def process_file(poses, joints_num, smpl_path, device="cuda", trans_matrix=None):
    ############### 解析动作数据 #################
    motions, trans, gender, betas = npy2info(poses, num_shapes=10)

    if motions.shape[1] == 72:
        mode = "smpl"
    elif motions.shape[1] == 156:
        mode = "smplh"
    elif motions.shape[1] == 165:
        mode = "smplx"

    ############ pose 归一化处理  ###############
    motions = motions.reshape(motions.shape[0], -1, 3)
    motions = torch.from_numpy(motions).float()
    if trans_matrix is not None:
        bak_rotation_matrix = torch.from_numpy(trans_matrix).float()
    else:
        bak_rotation_matrix = torch.eye(3).float()

    all_root_poses_matrix = axis_angle_to_matrix(motions[:, 0, :])
    aligned_root_poses_matrix = torch.matmul(bak_rotation_matrix, all_root_poses_matrix)
    motions[:, 0, :] = matrix_to_axis_angle(aligned_root_poses_matrix)
    motions = motions.reshape(motions.shape[0], -1)

    trans = torch.from_numpy(trans).float()
    trans = torch.matmul(bak_rotation_matrix, torch.transpose(trans, 0, 1))
    trans = torch.transpose(trans, 0, 1)

    ############### joints 生成 #################
    model = smplx.create(smpl_path, model_type=mode,
                        gender=gender, use_face_contour=True,
                        num_betas=10,
                        num_expression_coeffs=10,
                        ext="npz", use_pca=False, batch_size=motions.shape[0])
    model = model.to(device)
    inputs = info2dict(motions, trans, betas, mode, device)
    output = model(**inputs)
    joints = output.joints.cpu().detach().numpy()   ##### [nframes, njoints, 3]
    joints = joints[:, :joints_num, :]

    ########## position 归一化 ##########
    transtion = joints[:, 0, :] - joints[0:1, 0, :] #### [nframes, 3]
    speed = transtion[1:] - transtion[:-1]  ### [nframes-1, 3], 位移速度
    final_speed = np.zeros_like(transtion)  ### [nframes, 3]
    final_speed[:transtion.shape[0] - 1] = speed
    joints_relative = joints - joints[:, 0:1, :]    ### [nframes, njoints, 3]
    joints_relative = joints_relative[:, 1:, :].reshape(joints.shape[0], -1)    ### [nframes, 63]

    curr_poses = motions[:, :joints_num * 3]    #### [nframes, 66]
    curr_poses = curr_poses.reshape(curr_poses.shape[0], -1, 3)
    curr_poses = axis_angle_to_matrix(curr_poses)
    curr_poses = matrix_to_rotation_6d(curr_poses)  #### [nframes, 22, 6]
    curr_poses = curr_poses.reshape(curr_poses.shape[0], -1).numpy()

    data = np.concatenate([final_speed, joints_relative, curr_poses], axis=1)
    return data


def transfer_joints_to_jp(pose_dir, save_dir, joints_num=22, smpl_path="/apdcephfs_cq11/share_1567347/share_info/kleinhe/checkpoints/body_models", trans_matrix=None):
    os.makedirs(save_dir, exist_ok=True)
    ############### begin to transfer to humanml3d representations ########
    source_list = os.listdir(pose_dir)
    source_list = sorted(source_list)

    for source_file in tqdm(source_list):
        curr_pose = np.load(os.path.join(pose_dir, source_file))
        curr_poseR = reverse_pose(curr_pose)

        target_path = os.path.join(save_dir, source_file)
        target_pathM = os.path.join(save_dir, "M" + source_file)

        try:
            data = process_file(curr_pose, joints_num, smpl_path, trans_matrix=trans_matrix)
            np.save(target_path, data)
            dataR = process_file(curr_poseR, joints_num, smpl_path, trans_matrix=trans_matrix)
            np.save(target_pathM, dataR)
        except Exception as e:
            print(source_file)
            print(e)


if __name__ == "__main__":
    pose_dir = "/data/TTA/motionWE/results/ori"
    save_dir = "/data/TTA/motionWE/results/jp"
    trans_matrix = np.array([
        [1, 0, 0],
        [0, 0, 1],
        [0, -1, 0]
    ])

    transfer_joints_to_jp(pose_dir, save_dir, trans_matrix=trans_matrix)