import argparse

import numpy as np

from src.utils import *
from src.data_utils import BVH
from src.data_utils.Quaternions import Quaternions
from src.data_utils.Animation import Animation
import bpy
import os
from scipy.spatial.transform import Rotation

mixamo_joint_names = ['Hips', 'Spine', 'LeftUpLeg', 'RightUpLeg', 'Spine1', 'LeftLeg', 'RightLeg', 'Spine2',
                          'LeftFoot',
                          'RightFoot', 'Neck', 'LeftShoulder', 'RightShoulder', 'LeftToeBase', 'RightToeBase', 'Head',
                          'LeftArm', 'RightArm', 'LeftForeArm', 'RightForeArm', 'LeftHand', 'RightHand']
mixamo_parents = [-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 7, 7, 8, 9, 10, 11, 12, 16, 17, 18, 19]
smpl_joint_names = ['Pelvis',  'L_Hip',    'R_Hip',    'Spine1',  'L_Knee',     'R_Knee',     \
                   'Spine2',  'L_Ankle',  'R_Ankle',  'Spine3',  'L_Foot',     'R_Foot',     \
                   'Neck',    'L_Collar', 'R_Collar', 'Head',    'L_Shoulder', 'R_Shoulder', \
                   'L_Elbow', 'R_Elbow',  'L_Wrist',  'R_Wrist', 'L_Hand',     'R_Hand']
corr_smpl2mix = {
    'Pelvis':'Hips',
    'L_Hip':'LeftUpLeg',
    'R_Hip':'RightUpLeg',
    'Spine1':'Spine',
    'L_Knee':'LeftLeg',
    'R_Knee':'RightLeg',
    'Spine2':'Spine1',
    'L_Ankle':'LeftFoot',
    'R_Ankle':'RightFoot',
    'Spine3':'Spine2',
    'L_Foot':'LeftToeBase',
    'R_Foot':'RightToeBase',
    'Neck':'Neck',
    'L_Collar':'LeftShoulder',
    'R_Collar':'RightShoulder',
    'Head':'Head',
    'L_Shoulder':'LeftArm',
    'R_Shoulder':'RightArm',
    'L_Elbow':'LeftForeArm',
    'R_Elbow':'RightForeArm',
    'L_Wrist':'LeftHand',
    'R_Wrist':'RightHand',
    'L_Hand':None,
    'R_Hand':None,
}

def fbx2smpl(src_dir,src_name,dst_dir,work_dir):
    src_path = os.path.join(src_dir,src_name+'.fbx')
    dst_path = os.path.join(dst_dir,src_name+'.npy')
    tmp_path = os.path.join(work_dir,src_name+'.tmp.bvh')
    os.makedirs(work_dir,exist_ok=True)
    os.makedirs(dst_dir, exist_ok=True)

    bpy.ops.import_scene.fbx(filepath=src_path, use_anim=True)
    bpy.ops.export_anim.bvh(filepath=os.path.join(work_dir,src_name+'.tmp.bvh'), root_transform_only=True)
    anim, joint_names, frame_time = BVH.load(tmp_path)
    joint_names = [name if ':' not in name else name.split(':')[-1] for name in joint_names]
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

def auto_fbx2smpl(src_dir, work_dir, dst_root, verbose=True):
    proc_files = 0
    for root, dirs, files in os.walk(src_dir):
        sub_dirs = root.split("/")[-1]
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
                fbx2smpl(src_dir=root,src_name=src_name,dst_dir=target_root,work_dir=work_root)
            print(f'smpl rotations for {cur_path} been processed.') if verbose else None
    print(f'[auto_fbx2smpl] {proc_files} fbx files found & processed under dir {src_dir}')


if __name__ == '__main__':
    auto_fbx2smpl(src_dir="/data/TTA/MDM/Mixamoier/source", force=False)

