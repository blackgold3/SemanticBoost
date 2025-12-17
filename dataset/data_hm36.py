import os
from dataset.t2m.motion_representation import transfer_joints_to_t2m, extract_poses_from_smpls
from dataset.hm36.extract_joints import extract_joints_from_source
from dataset.utils.split_train_test import split_files
from dataset.utils.split_subdirs import split_files_to_subdirs
import argparse
from dataset.smr.motion_representation import transfer_joints_to_smr
import subprocess
import platform
import json
parser = argparse.ArgumentParser(description='SMPLs Dataset Self-label and Convert to HumanML3D format')
############# part1 extract datasets to humanml3d datasets ###############
parser.add_argument("--new_root_dir", default="/apdcephfs_cq11/share_1567347/share_info/kleinhe/motionDataset/hm36", type=str, help="new datasets base dir")
parser.add_argument('--model_path', type=str, default="/apdcephfs_cq11/share_1567347/share_info/kleinhe/checkpoints/body_models", help="SMPLs model path")
parser.add_argument("--input_path", type=str, default="/apdcephfs_cq11/share_1567347/share_info/kleinhe/raw_datasets/hm3.6/annot", help="SMPLs dataset")
parser.add_argument("--joints_dir", default="joints", type=str, help="SMPLs to 3D joints and save here")
parser.add_argument("--pose_dir", default="pose", type=str, help="SMPLs to 3D joints and save here")
parser.add_argument("--new_joints_dir", default="new_joints", type=str, help="SMPLs to 3D joints and save here")
parser.add_argument("--t2m_dir", default="new_joint_vecs", type=str, help="3D joints to HumanML3D format and save here")
parser.add_argument("--texts", default="texts", type=str, help="texts files will save here")
parser.add_argument("--smr_dir", default="smr_rep", type=str, help="3D joints to HumanML3D format and save here")
parser.add_argument("--target_fps", default=20, type=int, help="humanmact12 fps is 20, we need to align the smallest.")
############## activate which part ##############
# part 0 extract smpl 3d joints from source
# part 1 split several subdir
# part 2 concat files
# part 3 extract humanml3d rep
# part 4 extract smr rep
# part 5 split train_and_test
parser.add_argument("--activate", default=[0, 0, 0, 0, 0, 0], type=int, nargs="+",  help="2 parts, activate which part should set it = 1. ")

args = parser.parse_args()

joints_dir = os.path.join(args.new_root_dir, args.joints_dir)
pose_dir = os.path.join(args.new_root_dir, args.pose_dir)
new_joints_dir = os.path.join(args.new_root_dir, args.new_joints_dir)
final_t2m_dir = os.path.join(args.new_root_dir, args.t2m_dir)
texts_dir = os.path.join(args.new_root_dir, args.texts)
smr_dir = os.path.join(args.new_root_dir, args.smr_dir)
part0 = args.activate[0] == 1
part1 = args.activate[1] == 1
part2 = args.activate[2] == 1
part3 = args.activate[3] == 1
part4 = args.activate[4] == 1
part5 = args.activate[5] == 1
###### part 0 #############
if part0:
    hours = extract_joints_from_source(args.input_path, joints_dir, texts_dir, args.target_fps)
    print("total motiom time is:%.4f"%(hours))

if part1:
    os.makedirs(pose_dir, exist_ok=True)
    print("-------------- > split sub dirs < --------------------------")
    files = os.listdir(joints_dir)
    if "000000.npy" in files:
        dir_count = split_files_to_subdirs(joints_dir)
    else:
        dir_count = len(files)
        print("文件夹数量:", dir_count)

    for i in range(dir_count):
        target_path = ""
        jizhi_json = {
            "readable_name": "sever_hm36_joints2smpl_file%02d"%(i),
            "Token": "ayoT3ZAsqgdyIdISlbNNrg",
            "business_flag": "AILab_Game_Report_PaaS",
            "priority_level": "LOW",
            "elastic_level": 1, 
            "GPUName": "A100",
            "host_num": 1,
            "host_gpu_num": 1,
            "model_local_file_path": "/data/TTA/motionWE",
            "image_full_name": "mirrors.tencent.com/kleinhe/tta:sft",
            "cuda_version": "11.0",
            "start_cmd":"dataset/hm36/start.sh",
        }
        with open("dataset/hm36/start.sh", "w") as f:
            cmd = "python -m SMPLX.joints2smpl --model_path {} --source_path {} --target_path {}".format(args.model_path, os.path.join(joints_dir, "%02d"%(i)), pose_dir)
            f.writelines([cmd + "\n"])

        with open("dataset/hm36/jizhi.json", "w") as f:
            json.dump(jizhi_json, f, indent=2)

        command_start = "chmod 777 dataset/hm36/start.sh"
        subprocess.call(command_start, shell=platform.system() != 'Windows')

        command = "jizhi_client start -scfg dataset/hm36/jizhi.json"
        subprocess.call(command, shell=platform.system() != 'Windows')
    
    '''
    这里需要 python -m SMPLX.joints2smpl --model_path args.model_path --source_path joints_dir --target_path pose_dir
    结束后才进行 part2, part3 和 part4
    '''

if part2:
    extract_poses_from_smpls(args.model_path, pose_dir, new_joints_dir, "cuda", trans_matrix=None, amass=False)

if part3:
    print("-------------- > Transfer joints to HumanML3D data< --------------------------")
    transfer_joints_to_t2m(new_joints_dir, final_t2m_dir, 22)

if part4:
    transfer_joints_to_smr(pose_dir, new_joints_dir, smr_dir, 22)  

if part5:
    split_files(args.new_root_dir, smr_dir, texts_dir, smr_rep=True)