import os
from dataset.amass.raw_pose_processing import extract_amass_to_smpls
from dataset.t2m.motion_representation import extract_poses_from_smpls, transfer_joints_to_t2m
from dataset.smr.motion_representation import transfer_joints_to_smr
from dataset.JP.motion_representation import transfer_joints_to_jp
import argparse
from tqdm import tqdm
import numpy as np

parser = argparse.ArgumentParser(description='SMPLs Dataset Self-label and Convert to HumanML3D format')
############# part1 extract datasets to humanml3d datasets ###############
parser.add_argument("--new_root_dir", default="/apdcephfs_cq11/share_1567347/share_info/kleinhe/motionDataset/Amass/", type=str, help="new datasets base dir")
parser.add_argument('--model_path', type=str, default="/apdcephfs_cq11/share_1567347/share_info/kleinhe/checkpoints/body_models", help="SMPLs model path")
parser.add_argument("--input_path", type=str, default="/apdcephfs_cq11/share_1567347/share_info/kleinhe/raw_datasets/amass_smplh", help="SMPLs dataset")
parser.add_argument("--pose_dir1", default="pose_data", type=str, help="Extract npz to npy and save here, dir1 keep the files structure")
parser.add_argument("--pose_dir2", default="full_pose_data", type=str, help="dir2 extract all files to a single dir")
parser.add_argument("--joints_dir", default="joints", type=str, help="SMPLs to 3D joints and save here")
parser.add_argument("--t2m_dir", default="new_joint_vecs", type=str, help="3D joints to HumanML3D format and save here")
parser.add_argument("--smr_dir", default="smr_rep", type=str, help="3D joints to HumanML3D format and save here")
parser.add_argument("--jp_dir", default="jp_rep", type=str, help="3D joints to HumanML3D format and save here")
parser.add_argument("--texts", default="texts", type=str, help="texts files will save here")
parser.add_argument("--target_fps", default=20, type=int, help="humanmact12 fps is 20, we need to align the smallest.")
parser.add_argument("--only_t2m", action="store_true", default=False)
############# amass full settings ###############
parser.add_argument("--trainval", default=1, type=int, help="if dual with amass full")
############## activate which part ##############
# part 0 extract texts to npz
# part 1 extract files to smpl npy
# part 2 extract joints from smpl and transpose to humanml3d rep
# part 3 extract smr rep
# part 4 get split files for smr rep
parser.add_argument("--activate", default=[0, 0, 0, 0, 0], type=int, nargs="+",  help="5 parts, activate which part should set it = 1. ")

args = parser.parse_args()

device = "cuda"
joints_num = 22
pose_dir1 = os.path.join(args.new_root_dir, args.pose_dir1)
pose_dir2 = os.path.join(args.new_root_dir, args.pose_dir2)
joints_dir = os.path.join(args.new_root_dir, args.joints_dir)
final_t2m_dir = os.path.join(args.new_root_dir, args.t2m_dir)
smr_dir = os.path.join(args.new_root_dir, args.smr_dir)
jp_dir = os.path.join(args.new_root_dir, args.jp_dir)
texts_dir = os.path.join(args.new_root_dir, args.texts)
part0 = args.activate[0] == 1
part1 = args.activate[1] == 1
part2 = args.activate[2] == 1
part3 = args.activate[3] == 1
part4 = args.activate[4] == 1
############################### part10 extract smpl pose data from amass ##################
if part0:
    print("-------------- > Extract Amass datasets belongs to HumanML3D < --------------------------")
    index_path = "dataset/amass/index.csv"
    num_frames1 = extract_amass_to_smpls(args.input_path, pose_dir1, pose_dir2, args.target_fps, False, index_path, 0, None)
    index = 14616
    
    if args.trainval == 1:
        os.makedirs(texts_dir, exist_ok=True)
        print("-------------- > Extract Amass datasets no label < --------------------------")
        num_frames2 = extract_amass_to_smpls(args.input_path, pose_dir1, pose_dir2, args.target_fps, False, None, index, texts_dir)

############################### part2. smpl pose to joints and t2m representations ##################
if part1:
    print("-------------- > Transfer smpl pose to joints < --------------------------")

    trans_matrix = np.array([
        [1, 0, 0],
        [0, 0, -1],
        [0, 1, 0]]
    ) 

    extract_poses_from_smpls(args.model_path, pose_dir2, joints_dir, device, trans_matrix=trans_matrix, amass=True)

    print("-------------- > Transfer joints to HumanML3D data< --------------------------")
    transfer_joints_to_t2m(joints_dir, final_t2m_dir, joints_num)

############################### part3. SMR ##################
if part2:
    transfer_joints_to_smr(pose_dir2, joints_dir, smr_dir, 22)

############################### part4. JP ##################
if part3:
    trans_matrix = np.array([
        [1, 0, 0],
        [0, 0, 1],
        [0, -1, 0]
    ])

    transfer_joints_to_jp(pose_dir2, jp_dir, 22, args.model_path, trans_matrix=trans_matrix)

def split_amass(target_motion, target_text, extension, trainval=1):
    print("-------------- > get split files < --------------------------")
    train_dict = {}
    test_dict = {}

    train_text = {}
    test_text = {}

    train_file = os.path.join("dataset/amass/train.txt")
    test_file = os.path.join("dataset/amass/test.txt")
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
        curr_text = os.path.join(target_text, curr_name + ".txt")   

        motion = np.load(curr_path)
        with open(curr_text, "r") as f:
            text = f.readlines()
            text = [text[i].strip() for i in range(len(text))]
        
        save_key = "Amass_{}".format(curr_name) 
        if curr_name in test_names:
            test_dict[save_key] = motion
            test_text[save_key] = text

        if curr_name in train_names or trainval == 1:
            train_dict[save_key] = motion
            train_text[save_key] = text
        
    np.savez(os.path.join(args.new_root_dir, "train-motion{}.npz".format(extension)), **train_dict)
    np.savez(os.path.join(args.new_root_dir, "test-motion{}.npz".format(extension)), **test_dict)
    np.savez(os.path.join(args.new_root_dir, "train-text{}.npz".format(extension)), **train_text)
    np.savez(os.path.join(args.new_root_dir, "test-text{}.npz".format(extension)), **test_text)

if part4:
    if args.only_t2m:
        split_amass(final_t2m_dir, texts_dir, "-t2m", args.trainval)
    else:
        # split_amass(smr_dir, texts_dir, "-smr", args.trainval)
        split_amass(jp_dir, texts_dir, "-JP", args.trainval)
