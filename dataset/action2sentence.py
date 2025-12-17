import argparse
import os
from dataset.utils.dual_with_text import action2text
import random
import re
from tqdm import tqdm
import torch

def contains_chinese(text):
    pattern = re.compile(r'[\u4e00-\u9fa5]')
    match = re.search(pattern, text)
    return match is not None

parser = argparse.ArgumentParser(description='chatgpt label augmentation')
############# part1 extract datasets to humanml3d datasets ###############
parser.add_argument("--data_root", default="/apdcephfs_cq3/share_1330077/motionGPM/datasets", type=str)
parser.add_argument('--dataset', type=str, default="mixamo")
parser.add_argument("--text_dir", type=str, default="texts")
args = parser.parse_args()

root_path = os.path.join(args.data_root, args.dataset, args.text_dir)
files = os.listdir(root_path)

files = sorted(files)

for i in tqdm(range(len(files))):
    index = files[i].replace("M", "").replace(".txt", "")
    index = int(index)
    if args.dataset == "HumanML3D" and index < 14616:
        continue

    curr_path = os.path.join(root_path, files[i])

    if args.dataset == "100STYLE":
        with open(curr_path, "r") as f:
            lines = f.readlines()[:2]
            action = random.choice(lines).strip().split("#")[0]
    else:
        with open(curr_path, "r") as f:
            line = f.readlines()[0]
            line = line.strip()
        lines = [line + " \n"]
        action = line.split("#")[0]

    if args.dataset == "mixamo":
        human = ["person", "man", "woman"]
        index = torch.randint(0, 3, [3]).tolist()

        new_action = "A {} is performing ".format(human[index[0]]) + action
        new_line = action2text(new_action)
        lines.append(new_line + "\n")      

        new_action = "A {} performs ".format(human[index[1]]) + action
        new_line = action2text(new_action)
        lines.append(new_line + "\n")      

        new_action = "A {} does ".format(human[index[2]]) + action
        new_line = action2text(new_action)
        lines.append(new_line + "\n")    
    else:
        human = ["person", "man", "woman"]
        index = torch.randint(0, 3, [2]).tolist()

        temp_action1 = action.split()
        temp_action2 = action.split()

        if temp_action1[0].endswith("s"):
            temp_action1[0] += "es"
        else:
            temp_action1[0] += "s"
        temp_action1 = " ".join(temp_action1)
        new_action = "A {} ".format(human[index[0]]) + temp_action1
        new_line = action2text(new_action)
        lines.append(new_line + "\n")             

        if temp_action2[0].endswith("e"):
            temp_action2[0] = temp_action2[0][:-1]
        temp_action2[0] += "ing" 
        temp_action2 = " ".join(temp_action2)
        new_action = "A {} is ".format(human[index[1]]) + temp_action2
        new_line = action2text(new_action)
        lines.append(new_line + "\n")


    with open(curr_path, "w") as f:
        f.writelines(lines)

    
