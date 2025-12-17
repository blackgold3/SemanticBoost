import numpy as np
import os
from dataset.utils.reverse import swap_text
from dataset.utils.dual_with_text import action2text
import csv
from dataset.utils.convert_fps import convert_to_target_fps
import re
import torch

def extracted_pose_handler(pose_dir, text_dir, new_pose_dir):
    os.makedirs(new_pose_dir, exist_ok=True)
    os.makedirs(text_dir, exist_ok=True)

    frames_cut = {}
    with open('dataset/style100/Frame_Cuts.csv', newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            style = row["STYLE_NAME"]
            del(row["STYLE_NAME"])
            frames_cut[style] = row

    style_desc = {}
    with open('dataset/style100/Dataset_List.csv', newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            style = row["Style Name"]
            style_desc[style] = row["Description"]   

    action_map = {
        "br":"run backwards",
        "bw":"walk backwards",
        "fr":"run forward",
        "fw":"walks forward",
        "id":"stand",
        "sr":"run sidestep",
        "sw":"walk sidestep",
        "tr1":"transition",
        "tr2":"transition",
        "tr3":"transition"
    }

    index = 0
    for root, dirs, files in os.walk(pose_dir):
        for path in files:
            if not path.endswith(".npy"):
                continue
            else:
                name = path.split(".")[0]
                style, movement = name.split("_")
                action = action_map[movement.lower()]
                styles = [style_desc[style].lower()]

                split_style = re.findall('[A-Z][^A-Z]*', style)
                split_style = " ".join(split_style).lower()
                split_style = "style " + split_style
                styles.append(split_style)

                actions = [action + " with " + styles[i] for i in range(len(styles))]
                reverse_actions = [swap_text(actions[i]) for i in range(len(actions))]

                curr_path = os.path.join(root, path)
                pose = np.load(curr_path)
                start_key = movement.upper() + "_START"
                end_key = movement.upper() + "_STOP"
                start_frame = int(frames_cut[style][start_key])
                end_frame = int(frames_cut[style][end_key])
                real_pose = pose[start_frame:end_frame]
                real_pose = convert_to_target_fps(real_pose, 60, 20)

                split_poses = torch.from_numpy(real_pose)
                split_poses = torch.split(split_poses, 250, dim=0)

                for curr_pose in split_poses:
                    pose_save_path = os.path.join(new_pose_dir, "%06d.npy"%(index))
                    text_save_path = os.path.join(text_dir, "%06d.txt"%(index))
                    reverse_text_save_path = os.path.join(text_dir, "M%06d.txt"%(index))

                    np.save(pose_save_path, curr_pose.numpy())

                    with open(text_save_path, "w") as f:
                        save_actions = [action2text(action) + "\n" for action in actions]
                        f.writelines(save_actions)
                    
                    with open(reverse_text_save_path, "w") as f:
                        save_reverse_actions = [action2text(action) + "\n" for action in reverse_actions]
                        f.writelines(save_reverse_actions)

                    index += 1

                    if index % 100 == 0:
                        print("Has been dual with %d"%(index))