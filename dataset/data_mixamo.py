import os
import numpy as np
from dataset.t2m.motion_representation import transfer_joints_to_t2m, extract_poses_from_smpls
from dataset.Mixamoier.handler_joints import extracted_pose_handler
from dataset.utils.split_train_test import split_files
import argparse
import json
import subprocess, platform
from dataset.smr.motion_representation import transfer_joints_to_smr
from dataset.utils.convert_fps import convert_to_target_fps
from tqdm import tqdm

parser = argparse.ArgumentParser(description='SMPLs Dataset Self-label and Convert to HumanML3D format')
############# part1 extract datasets to humanml3d datasets ###############
parser.add_argument("--new_root_dir", default="/apdcephfs_cq11/share_1567347/share_info/kleinhe/motionDataset/mixamo", type=str, help="new datasets base dir")
parser.add_argument('--model_path', type=str, default="/apdcephfs_cq11/share_1567347/share_info/kleinhe/checkpoints/body_models", help="SMPLs model path")
parser.add_argument("--input_path", type=str, default="/apdcephfs_cq11/share_1567347/share_info/kleinhe/raw_datasets/mixamo", help="SMPLs dataset")
parser.add_argument("--joints_dir", default="joints", type=str, help="SMPLs to 3D joints and save here")
parser.add_argument("--new_pose_dir", default="full_pose_data", type=str, help="SMPLs to 3D joints and save here")
parser.add_argument("--pose_dir", default="pose", type=str, help="SMPLs to 3D joints and save here")
parser.add_argument("--bvh", default="bvh", type=str, help="SMPLs BVH dirs")
parser.add_argument("--t2m_dir", default="new_joint_vecs", type=str, help="3D joints to HumanML3D format and save here")
parser.add_argument("--texts", default="texts", type=str, help="texts files will save here")
parser.add_argument("--smr_dir", default="smr_rep", type=str, help="3D joints to HumanML3D format and save here")
parser.add_argument("--target_fps", default=20, type=int, help="humanmact12 fps is 20, we need to align the smallest.")
############## activate which part ##############
# part 0 extract smpl 3d joints from source
# part 1 trans pose to joints
# part 2 extract humanml3d rep
# part 3 split train_and_test
parser.add_argument("--activate", default=[0, 0, 0, 0, 0], type=int, nargs="+",  help="2 parts, activate which part should set it = 1. ")

args = parser.parse_args()

joints_dir = os.path.join(args.new_root_dir, args.joints_dir)
new_pose_dir = os.path.join(args.new_root_dir, args.new_pose_dir)
pose_dir = os.path.join(args.new_root_dir, args.pose_dir)
temp_bvh_dir = os.path.join(args.new_root_dir, args.bvh)
final_t2m_dir = os.path.join(args.new_root_dir, args.t2m_dir)
texts_dir = os.path.join(args.new_root_dir, args.texts)
smr_dir = os.path.join(args.new_root_dir, args.smr_dir)
part0 = args.activate[0] == 1
part1 = args.activate[1] == 1
part2 = args.activate[2] == 1
part3 = args.activate[3] == 1
part4 = args.activate[4] == 1
###### part 0 #############
if part0:
    json_dict = {
        "source_dir":args.input_path,
        "bvh_dir":temp_bvh_dir,
        "pose":pose_dir
    }
    with open("dataset/Mixamoier/temp.json", "w") as f:
        json.dump(json_dict, f, indent=2)

    lines = []
    lines.append("cd dataset/Mixamoier\n")
    lines.append("/data/camera/blender-4.0.2-linux-x64/blender --background --python interf_fbx2smpl.py\n")
    lines.append("cd /data/TTA/motionWE\n")
    with open("dataset/Mixamoier/start.sh", "w") as f:
        f.writelines(lines)

    cmd = "bash dataset/Mixamoier/start.sh"
    subprocess.call(cmd, shell=platform.system() != 'Windows')


if part1:
    target_dir = os.path.join(pose_dir, "mixamo")
    os.makedirs(target_dir, exist_ok=True)
    for root, dirs, files in os.walk(pose_dir):
        for i in tqdm(range(len(files))):
            curr_path = os.path.join(root, files[i])
            
            if target_dir in curr_path:
                continue

            target_path = os.path.join(target_dir, files[i])
            curr_motion = np.load(curr_path)
            res_motion = convert_to_target_fps(curr_motion, 20, 20)
            np.save(target_path, res_motion)

    extracted_pose_handler(target_dir, texts_dir, new_pose_dir)
    extract_poses_from_smpls(args.model_path, new_pose_dir, joints_dir, "cuda", None, False, reverse=True)

if part2:
    print("-------------- > Transfer joints to HumanML3D data< --------------------------")
    transfer_joints_to_t2m(joints_dir, final_t2m_dir, 22)

if part3:
    transfer_joints_to_smr(new_pose_dir, joints_dir, smr_dir, 22)  

if part4:
    split_files(args.new_root_dir, smr_dir, texts_dir, smr_rep=True)