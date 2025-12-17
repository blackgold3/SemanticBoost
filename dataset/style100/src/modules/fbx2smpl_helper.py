import argparse

import numpy as np

from src.utils import *
from src.data_utils import BVH
from src.data_utils.Quaternions import Quaternions
from src.data_utils.Animation import Animation
import bpy
import os
from scipy.spatial.transform import Rotation

smpl_joint_names = ['Pelvis',  'L_Hip',    'R_Hip',    'Spine1',  'L_Knee',     'R_Knee',     \
                   'Spine2',  'L_Ankle',  'R_Ankle',  'Spine3',  'L_Foot',     'R_Foot',     \
                   'Neck',    'L_Collar', 'R_Collar', 'Head',    'L_Shoulder', 'R_Shoulder', \
                   'L_Elbow', 'R_Elbow',  'L_Wrist',  'R_Wrist', 'L_Hand',     'R_Hand']
corr_smpl2mix = {
    'Pelvis':'Hips',
    'L_Hip':'LeftHip',
    'R_Hip':'RightHip',
    'Spine1':'Chest',
    'L_Knee':'LeftKnee',
    'R_Knee':'RightKnee',
    'Spine2':'Chest2',
    'L_Ankle':'LeftAnkle',
    'R_Ankle':'RightAnkle',
    'Spine3':'Chest3',
    'L_Foot':'LeftToe',
    'R_Foot':'RightToe',
    'Neck':'Neck',
    'L_Collar':'LeftCollar',
    'R_Collar':'RightCollar',
    'Head':'Head',
    'L_Shoulder':'LeftShoulder',
    'R_Shoulder':'RightShoulder',
    'L_Elbow':'LeftElbow',
    'R_Elbow':'RightElbow',
    'L_Wrist':'LeftWrist',
    'R_Wrist':'RightWrist',
    'L_Hand':None,
    'R_Hand':None,
}

def fbx2smpl(src_name, dst_dir, work_dir):
    dst_path = os.path.join(dst_dir,src_name+'.npy')
    tmp_path = os.path.join(work_dir,src_name+'.bvh')
    os.makedirs(work_dir,exist_ok=True)
    os.makedirs(dst_dir, exist_ok=True)

    anim, joint_names, frame_time = BVH.load(tmp_path)
    joint_names = [name if ':' not in name else name.split(':')[-1] for name in joint_names]
    # print(joint_names)
    anim:Animation = anim

    root = anim.positions[:, 0]
    root = root - root[0]
    translation = root / 100

    frame_len = anim.rotations.shape[0]
    smpl_rot = [] # first in quat.
    for smpl_joint_name in smpl_joint_names:
        if smpl_joint_name in corr_smpl2mix and corr_smpl2mix[smpl_joint_name] is not None and corr_smpl2mix[smpl_joint_name] in joint_names:
            smpl_rot.append(anim.rotations[:,joint_names.index(corr_smpl2mix[smpl_joint_name])].qs)
        else:
            smpl_rot.append(Quaternions.id(frame_len).qs)
    smpl_rot = Quaternions(np.concatenate([ele[:,None,:] for ele in smpl_rot],axis=1)).euler().reshape(-1,3)
    smpl_rot = np.array(Rotation.from_euler('xyz',angles=smpl_rot).as_rotvec().data).reshape(frame_len,-1)

    smpl_rot = np.concatenate([smpl_rot, translation], axis=1)

    np.save(dst_path,smpl_rot,allow_pickle=False)
    # print(smpl_rot.shape)
    return 1

def auto_fbx2smpl(work_dir, dst_root, verbose=True):
    proc_files = 0
    for root, dirs, files in os.walk(work_dir):
        sub_dirs = root.split("/")[-1]
        if root == work_dir:
            continue
        target_root = os.path.join(dst_root, sub_dirs)
        work_root = os.path.join(work_dir, sub_dirs)
        for file in files:
            file:str = file
            src_name = file[:-4]
            cur_path = os.path.join(root,src_name+'.npy')
            proc_files += 1

            if os.path.exists(os.path.join(target_root,src_name+'.npy')):
                pass
            else:
                fbx2smpl(src_name=src_name,dst_dir=target_root,work_dir=work_root)
            print(f'smpl rotations for {cur_path} been processed.') if verbose else None
        print(f'[auto_fbx2smpl] {proc_files} fbx files found & processed under dir {work_root}')


if __name__ == '__main__':
    auto_fbx2smpl(src_dir="/data/TTA/MDM/Mixamoier/source", force=False)

