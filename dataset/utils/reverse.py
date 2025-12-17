import numpy as np
import torch
from dataset.utils.read_from_npy import npy2info, info2dict

def swap_text(text):
    words = text.split()
    for idx in range(len(words)):
        if words[idx].startswith("left"):
            words[idx] = words[idx].replace("left","right")
        elif words[idx].startswith("right"):
            words[idx] = words[idx].replace("right","left")
        elif words[idx].startswith("clockwise"):
            words[idx] = words[idx].replace("clockwise","counterclockwise")
        elif words[idx].startswith("counterclockwise"):
            words[idx] = words[idx].replace("counterclockwise","clockwise")
        elif words[idx].startswith("counter-clockwise"):
            words[idx] = words[idx].replace("counter-clockwise","clockwise")
        elif words[idx] == "counter" and idx != len(words) - 1 and words[idx + 1] == "clockwise":
            words[idx] = ""
    temp_text_M = " ".join(words)
    return temp_text_M


def reverse_joints(data):
    data = data.copy()

    if len(data.shape) != 3:
        data = torch.from_numpy(data)
        curr_data = curr_data.reshape(data.shape[0], -1, 3)
        assert len(curr_data.shape) == 3 and curr_data.shape[-1] == 3
    else:
        curr_data = torch.from_numpy(data)      #### humanact12

    curr_data = curr_data.detach().clone()

    model_type = {52:"smplh", 55:"smplx", 22:"smpl", 24:"smpl"}[curr_data.shape[1]]

    right_chain = [2, 5, 8, 11, 14, 17, 19, 21]
    left_chain = [1, 4, 7, 10, 13, 16, 18, 20]
    
    if model_type == "smplx":
        left_eye_chain = [23]
        right_eye_chain = [24]
        left_hand_chain = [25, 26, 27, 37, 38, 39, 28, 29, 30, 34, 35, 36, 31, 32, 33]
        right_hand_chain = [46, 47, 48, 49, 50, 51, 43, 44, 45, 40, 41, 42, 52, 53, 54]
    elif model_type == "smplh":
        left_hand_chain = [22, 23, 24, 34, 35, 36, 25, 26, 27, 31, 32, 33, 28, 29, 30]
        right_hand_chain = [43, 44, 45, 46, 47, 48, 40, 41, 42, 37, 38, 39, 49, 50, 51]

    tmp = curr_data[:, right_chain]
    curr_data[:, right_chain] = curr_data[:, left_chain]
    curr_data[:, left_chain] = tmp

    if curr_data.shape[1] > 24:
        tmp = curr_data[:, right_hand_chain]
        curr_data[:, right_hand_chain] = curr_data[:, left_hand_chain]
        curr_data[:, left_hand_chain] = tmp
    
    if curr_data.shape[1] > 52:
        temp = curr_data[:, right_eye_chain]
        curr_data[:, right_eye_chain] = curr_data[:, left_eye_chain]
        curr_data[:, left_eye_chain] = temp
    
    curr_data[..., 0] *= -1
    curr_data = curr_data.numpy()
    return curr_data

def reverse_pose(data):
    left_chain = {
        "smpl": [1, 4, 7, 10, 13, 16, 18, 20],
        "smplh": [1, 4, 7, 10, 13, 16, 18, 20, 22, 23, 24, 34, 35, 36, 25, 26, 27, 31, 32, 33, 28, 29, 30],
        "smplx": [1, 4, 7, 10, 13, 16, 18, 20, 23, 25, 26, 27, 37, 38, 39, 28, 29, 30, 34, 35, 36, 31, 32, 33]
    }
    right_chain = {
        "smpl": [2, 5, 8, 11, 14, 17, 19, 21],
        "smplh": [2, 5, 8, 11, 14, 17, 19, 21, 43, 44, 45, 46, 47, 48, 40, 41, 42, 37, 38, 39, 49, 50, 51],
        "smplx": [2, 5, 8, 11, 14, 17, 19, 21, 24, 46, 47, 48, 49, 50, 51, 43, 44, 45, 40, 41, 42, 52, 53, 54]
    }

    data = data.copy()
    curr_pose, curr_trans, curr_gender, curr_betas = npy2info(data, 10)

    if curr_gender == "female":
        curr_gender = np.zeros([curr_pose.shape[0], 1])
    elif curr_gender == "female":
        curr_gender = np.ones([curr_pose.shape[0], 1])
    else:
        curr_gender = np.ones([curr_pose.shape[0], 1]) * 2

    if curr_pose.shape[1] == 72:
        model_type = "smpl"
    elif curr_pose.shape[1] == 156:
        model_type = "smplh"
    elif curr_pose.shape[1] == 165:
        model_type = "smplx"
    else:
        raise ValueError("Wrong Input type")

    curr_pose = curr_pose.reshape(curr_pose.shape[0], -1, 3)
    tmp = curr_pose[:, right_chain[model_type]]
    curr_pose[:, right_chain[model_type]] = curr_pose[:, left_chain[model_type]]
    curr_pose[:, left_chain[model_type]] = tmp
    curr_pose[:, :, 1:3] *= -1
    curr_pose = curr_pose.reshape(curr_pose.shape[0], -1)

    curr_trans[:, 0] *= -1
    target_file = np.concatenate([curr_pose, curr_trans, curr_gender], axis=1)
    return target_file