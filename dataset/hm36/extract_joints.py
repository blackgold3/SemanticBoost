import numpy as np
from dataset.hm36.hm36tosmpl import convert_hm36_to_smpl
import os
from dataset.utils.reverse import swap_text
from dataset.utils.dual_with_text import action2text
import torch
from tqdm import tqdm

action = {
    "02":"direct traffic",
    "03":"discuss with others",
    "04":"eat",
    "05":"greet",
    "06":"talk on the phone",
    "07":"strike a pose",
    "08":"make purchases",
    "09":"sit on the chair",
    "10":"do activities while seated",
    "11":"smoke",
    "12":"take a photo",
    "13":"wait someone",
    "14":"walk",
    "15":"walk the dog",
    "16":"walk together with others",
}

def extract_joints_from_source(source, joints_path, text_path, target_fps=20):
    os.makedirs(joints_path, exist_ok=True)
    os.makedirs(text_path, exist_ok=True)

    default_fps = 60
    ratio = default_fps // target_fps

    trans_matrix = np.array([
        [1, 0, 0],
        [0, 0, -1],
        [0, 1, 0]
    ])

    index = 0
    nframes = 0

    for root, dirs, files in tqdm(os.walk(source)):
        fdir = root.split("/")[-1]   
        try:
            action_id = fdir.split("_")[3]
        except:
            continue
        action_name = action[action_id]
        Maction = swap_text(action_name)

        curr_path = os.path.join(root, "matlab_meta.txt")

        with open(curr_path, "r") as f:
            lines = f.readlines()

        poses = []
        for i in range(9, len(lines)):
            pose1 = lines[i].replace("\n", "")
            pose1 = pose1.split()[1::]
            pose1 = [float(pose1[i]) for i in range(96)]
            pose1 = np.asarray(pose1)
            poses.append(pose1)

        poses = np.stack(poses, axis=0)
        poses /= 1000
        poses = poses.reshape(poses.shape[0], -1, 3)    ### 【nframes, 32, 3】

        poses = convert_hm36_to_smpl(poses)         ### [nframes, 22, 3]
        poses = poses[::ratio, :, :]
        poses = np.dot(poses, trans_matrix)     #### [nframes, 22, 3]
        nframes += poses.shape[0]

        poses = torch.from_numpy(poses)
        poses = poses.split(200, dim=0)

        for pose in poses:
            pose = pose.numpy()
            save_path = os.path.join(joints_path, "%06d.npy"%(index))
            condition_path = os.path.join(text_path, "%06d.txt"%(index))
            condition_pathM = os.path.join(text_path, "M%06d.txt"%(index))

            np.save(save_path, pose)

            with open(condition_path, "w") as f:
                f.writelines([action2text(action_name) + " \n"])

            with open(condition_pathM, "w") as f:
                f.writelines([action2text(Maction) + " \n"])      

            index += 1
            
    
    return nframes / target_fps / 3600