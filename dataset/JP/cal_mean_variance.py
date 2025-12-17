import numpy as np
from os.path import join as pjoin
from tqdm import tqdm
import torch
from dataset.utils.rotation_conversions import *
def rotation_JP(motion, angel):
    new_motion = motion.copy()
    rotation_angel = np.deg2rad(angel)
    R_y = np.array([
        [np.cos(rotation_angel), 0, np.sin(rotation_angel)],
        [0, 1, 0],
        [-np.sin(rotation_angel), 0, np.cos(rotation_angel)]
    ]) 
    new_motion[:, :3] = np.matmul(R_y, new_motion[:, :3].T).T
    joints =  new_motion[:, 3:66].reshape(new_motion.shape[0], -1, 3).transpose(0, 2, 1)    ### [N, 3, 21]
    joints = np.matmul(R_y, joints).transpose(0, 2, 1).reshape(new_motion.shape[0], -1)  ### [N ,21, 3]
    new_motion[:, 3:66] = joints
    root = torch.from_numpy(new_motion[:, 66:72])
    root = rotation_6d_to_matrix(root)  ### [N, 3, 3]
    root = torch.from_numpy(R_y).float() @ root 
    root = matrix_to_rotation_6d(root)
    root = root.numpy()
    new_motion[:, 66:72] = root
    return new_motion

def mean_variance(train_file, save_dir, joints_num):
    with open(train_file, "r") as f:
        lines = f.readlines()
        lines = [line.strip() for line in lines]

    base_motion = 0
    base_std = 0
    count = 0
    for line in lines:
        line = line.split("#")[0]
        curr_matrix = np.load(line, allow_pickle=True)
        keys = curr_matrix.files
        for j in tqdm(range(len(keys))):
            key = keys[j]
            motion = curr_matrix[key]

            if np.isnan(motion).any():
                print("nan value in ==>", key)
                continue

            motion_90 = rotation_JP(motion, 90)
            motion_180 = rotation_JP(motion, 180)
            motion_270 = rotation_JP(motion, 270)

            count += (motion.shape[0] + motion_90.shape[0] + motion_180.shape[0] + motion_270.shape[0])
            base_motion += motion.sum(axis=0)
            base_motion += motion_90.sum(axis=0)
            base_motion += motion_180.sum(axis=0)
            base_motion += motion_270.sum(axis=0)

    Mean = base_motion / count
    count = 0

    for line in lines:
        line = line.split("#")[0]
        curr_matrix = np.load(line, allow_pickle=True)
        keys = curr_matrix.files
        for j in tqdm(range(len(keys))):
            key = keys[j]
            motion = curr_matrix[key]
            if np.isnan(motion).any():
                print("nan value in ==>", key)
                continue

            motion_90 = rotation_JP(motion, 90)
            motion_180 = rotation_JP(motion, 180)
            motion_270 = rotation_JP(motion, 270)

            count += (motion.shape[0] + motion_90.shape[0] + motion_180.shape[0] + motion_270.shape[0])
            base_std += ((motion - Mean) ** 2).sum(axis=0)
            base_std += ((motion_90 - Mean) ** 2).sum(axis=0)
            base_std += ((motion_180 - Mean) ** 2).sum(axis=0)
            base_std += ((motion_270 - Mean) ** 2).sum(axis=0)

    print("total frames is:%d"%(count))
    Std = np.sqrt(base_std / count)

    Mean[66:] = 0   #### rotation 不做归一化
    Std[66:] = 1
    Std[[0, 2]] = Std[[0, 2]].mean() / 1.0  #### [XZ 轨迹]
    Std[1:2] = Std[1:2].mean() / 1.0        #### [Y 轨迹]
    Std[3:66] = Std[3:66].mean() / 1.0      #### 相对关节点
   
    assert joints_num * 9 == Std.shape[-1]

    np.save(pjoin(save_dir, 'Mean-JP.npy'), Mean)
    np.save(pjoin(save_dir, 'Std-JP.npy'), Std)     

    return Mean, Std

