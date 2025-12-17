import os
from dataset.t2m.cal_mean_variance import mean_variance
from dataset.smr.cal_mean_variance import mean_variance as smean_svariance
from dataset.JP.cal_mean_variance import mean_variance as jmean_jvariance
import argparse
from tqdm import tqdm
import numpy as np

parser = argparse.ArgumentParser(description='SMPLs Dataset Self-label and Convert to HumanML3D format')
############# part1 extract datasets to humanml3d datasets ###############
parser.add_argument("--root_path", type=str, default="/apdcephfs_cq11/share_1567347/share_info/kleinhe/motionDataset", help="SMPLs datasets")
parser.add_argument("--save_dir", type=str, default="train_amass_JP_cond", help="SMPLs dataset")
parser.add_argument("--datasets", default=["HumanML3D", "hm36", "mixamo", "100STYLE"], type=str, nargs="+",  help="5 parts, activate which part should set it = 1. ")
parser.add_argument("--rep", default="smr", type=str, choices=["t2m", "smr", "JP"])
############## activate which part ##############
# part 0 split files
# part 1 calculate mean and std
# part 2 total hours calculate
parser.add_argument("--activate", default=[0, 0, 0], type=int, nargs="+",  help="5 parts, activate which part should set it = 1. ")

args = parser.parse_args()

device = "cuda"
joints_num = 22
part0 = args.activate[0] == 1
part1 = args.activate[1] == 1
part2 = args.activate[2] == 1

root_path = args.root_path
target_path = os.path.join(root_path, args.save_dir)
os.makedirs(target_path, exist_ok=True)
datasets = args.datasets

if args.rep == "t2m":
    extension = ""
elif args.rep == "smr":
    extension = "-smr"
elif args.rep == "JP":
    extension = "-JP"

test_path = "test-motion{}.npz".format(extension)
train_path = "train-motion{}.npz".format(extension)
test_path_text = "test-text{}.npz".format(extension)
train_path_text = "train-text{}.npz".format(extension)
test_path_cond = "test-cond{}.npz".format(extension)
train_path_cond = "train-cond{}.npz".format(extension)
target_train_path = "train{}.txt".format(extension)
target_test_path = "test{}.txt".format(extension)

if part0:
    print("============= part 0 get split files ============")
    train_lines = []
    test_lines = []
    for i in range(len(datasets)):
        curr_test_path_motion = os.path.join(root_path, datasets[i], test_path)   
        curr_test_path_text = os.path.join(root_path, datasets[i], test_path_text)
        curr_test_path_cond = os.path.join(root_path, datasets[i], test_path_cond)
        if os.path.exists(curr_test_path_motion) and os.path.exists(curr_test_path_text):
            test_lines.append("{}#{}#{}".format(curr_test_path_motion, curr_test_path_text, curr_test_path_cond) + "\n")

        curr_train_path_motion = os.path.join(root_path, datasets[i], train_path)   
        curr_train_path_text = os.path.join(root_path, datasets[i], train_path_text)   
        curr_train_path_cond = os.path.join(root_path, datasets[i], train_path_cond)   
        train_lines.append("{}#{}#{}".format(curr_train_path_motion, curr_train_path_text, curr_train_path_cond) + "\n")                     

    with open(os.path.join(target_path, target_test_path), "w") as f:
        f.writelines(test_lines)
    with open(os.path.join(target_path, target_train_path), "w") as f:
        f.writelines(train_lines)

if part1:
    print("============= part 1 calculate mean and std ============")
    final_file = os.path.join(target_path, target_train_path)
    if args.rep == "t2m":
        mean_variance(final_file, target_path, joints_num)
    elif args.rep == "smr":
        smean_svariance(final_file, target_path, joints_num)
    elif args.rep == "JP":
        jmean_jvariance(final_file, target_path, joints_num)

if part2:
    ############### calculate total frames ###########
    print("============= part 2 get total time ============")
    final_file = os.path.join(target_path, target_train_path)
    nframes = 0
    with open(final_file, "r") as f:
        lines = f.readlines()
        lines = [line.strip() for line in lines]

    for line in lines:
        line = line.split("#")[0]
        curr_matrix = np.load(line, allow_pickle=True)
        keys = curr_matrix.files
        for j in tqdm(range(len(keys))):
            key = keys[j]
            motion = curr_matrix[key]
            nframes += motion.shape[0]

    hours = nframes / 20 / 3600
    print("total frames is :%d, total time is %.4f hours"%(nframes, hours))