import os
from dataset.amass.raw_pose_processing import extract_amass_to_smpls
from dataset.t2m.motion_representation import extract_poses_from_smpls, transfer_joints_to_t2m
from dataset.smr.motion_representation import transfer_joints_to_smr
import argparse
from tqdm import tqdm
import random
import numpy as np
from dataset.utils.dual_with_text import dual_with_text

parser = argparse.ArgumentParser(description='SMPLs Dataset Self-label and Convert to HumanML3D format')
############# part1 extract datasets to humanml3d datasets ###############
parser.add_argument("--new_root_dir", default="/apdcephfs_cq3/share_1330077/motionGPM/datasets/KIT-ML/", type=str, help="new datasets base dir")
parser.add_argument('--model_path', type=str, default="/apdcephfs_cq3/share_1330077/motionGPM/body_models", help="SMPLs model path")
parser.add_argument("--joints_dir", default="new_joints", type=str, help="SMPLs to 3D joints and save here")
parser.add_argument("--t2m_dir", default="new_joint_vecs", type=str, help="3D joints to HumanML3D format and save here")
parser.add_argument("--texts", default="Cmodality", type=str, help="texts files will save here")
############## activate which part ##############
# part 0 extract texts to npz
# part 1 get split files for smr rep
parser.add_argument("--activate", default=[0, 0], type=int, nargs="+",  help="5 parts, activate which part should set it = 1. ")

args = parser.parse_args()

device = "cuda"
joints_num = 21
joints_dir = os.path.join(args.new_root_dir, args.joints_dir)
final_t2m_dir = os.path.join(args.new_root_dir, args.t2m_dir)
texts_dir = os.path.join(args.new_root_dir, args.texts)
part0 = args.activate[0] == 1
part1 = args.activate[1] == 1
############################## part 0. extract texts information to json file ##########
if part0:
    os.makedirs(texts_dir, exist_ok=True)
    ori_texts_path = os.path.join(args.new_root_dir, "texts")
    files = os.listdir(ori_texts_path)
    for i in tqdm(range(len(files))):
        curr_path = files[i]
        name = curr_path.replace(".txt", ".npz")
        curr_path = os.path.join(ori_texts_path, curr_path)
        saved_path = os.path.join(texts_dir, name)
        with open(curr_path, "r") as f:
            lines = f.readlines()
        
        texts = []
        for line in lines:
            line = line.replace("\n", "")
            texts.append(line)
        
        curr_dict = {
            "action":"",
            "text":texts,
            "audio":""
        }
    
        np.savez(saved_path, **curr_dict)

if part1:
    def split_amass(target_motion, target_text, extension):
        print("-------------- > get split files < --------------------------")
        train_dict = {}
        test_dict = {}
        trainval_dict = {}

        train_file = os.path.join("dataset/kit/train.txt")
        test_file = os.path.join("dataset/kit/test.txt")
        with open(train_file, "r") as f:
            train_names = f.readlines()
            train_names = [name.strip() for name in train_names]

        with open(test_file, "r") as f:
            test_names = f.readlines()
            test_names = [name.strip() for name in test_names]

        files = os.listdir(target_motion)
        files = sorted(files)
        for i in tqdm(range(len(files))):
            curr_name = files[i].replace(".npy", "")
            curr_path = os.path.join(target_motion, curr_name + ".npy")
            curr_text = os.path.join(target_text, curr_name + ".npz")   

            motion = np.load(curr_path)
            captions = dual_with_text(curr_text)
            if captions is None:
                continue

            save_key = "KIT_{}".format(curr_name) 
            if curr_name in test_names:
                test_dict[save_key] = {
                    "motion":motion,
                    "text":captions
                }

            if curr_name in train_names:
                train_dict[save_key] = {
                    "motion":motion,
                    "text":captions
                }            
            
            trainval_dict[save_key] = {
                "motion":motion,
                "text":captions            
            }

        np.savez(os.path.join(args.new_root_dir, "train{}.npz".format(extension)), **train_dict)
        np.savez(os.path.join(args.new_root_dir, "test{}.npz".format(extension)), **test_dict)
        np.savez(os.path.join(args.new_root_dir, "trainval{}.npz".format(extension)), **trainval_dict)

    split_amass(final_t2m_dir, texts_dir, "")
   