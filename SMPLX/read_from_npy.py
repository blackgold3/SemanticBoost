import numpy as np
import torch

def npy2info(motions, num_shapes=10):
    if isinstance(motions, str):
        motions = np.load(motions)
        
    trans = None
    gnum = 2

    if isinstance(motions, np.ndarray):
        betas = np.zeros([motions.shape[0], num_shapes]).astype(motions.dtype)
    else:
        betas = torch.zeros([motions.shape[0], num_shapes], dtype=motions.dtype)

    if len(motions.shape) == 3:
        motions = motions.reshape(motions.shape[0], -1)

    if motions.shape[1] in [73, 157, 166]:
        gnum = motions[:, -1:][0]
        motions = motions[:, :-1]
    elif motions.shape[1] in [75, 159, 168]:
        gnum = 2
        trans = motions[:, -3::]
        motions = motions[:, :-3]
    elif motions.shape[1] in [76, 160, 169]:
        gnum = motions[:, -1:][0]
        trans = motions[:, -4:-1:]
        motions = motions[:, :-4]     
    elif motions.shape[1] in [72 + num_shapes, 156 + num_shapes, 165 + num_shapes]:
        betas = motions[:, -num_shapes::]
        gnum = 2
        motions = motions[:, :-num_shapes]
    elif motions.shape[1] in [73 + num_shapes, 157 + num_shapes, 166 + num_shapes]:
        betas = motions[:, -num_shapes::]
        gnum = motions[:, -num_shapes-1:-num_shapes:][0]
        motions = motions[:, :-num_shapes-1]      
    elif motions.shape[1] in [75 + num_shapes, 159 + num_shapes, 168 + num_shapes]:
        betas = motions[:, -num_shapes::]
        gnum = 2
        trans = motions[:, -num_shapes-3:-num_shapes:]
        motions = motions[:, :-num_shapes-3]      
    elif motions.shape[1] in [76 + num_shapes, 160 + num_shapes, 169 + num_shapes]:
        betas = motions[:, -num_shapes::]
        gnum = motions[:, -num_shapes-1:-num_shapes:][0]
        trans = motions[:, -num_shapes-4:-num_shapes-1:]
        motions = motions[:, :-num_shapes-4]    

    if gnum == 0:
        gender = "female"
    elif gnum == 1:
        gender = "male"
    else:
        gender = "neutral"

    return motions, trans, gender, betas

def info2dict(pose, trans=None, betas=None, mode="smpl", device="cuda", index=-1):
    if isinstance(pose, np.ndarray):
        pose = torch.from_numpy(pose)

    if trans is not None and isinstance(trans, np.ndarray):
        trans = torch.from_numpy(trans)
    
    if betas is not None and isinstance(betas, np.ndarray):
        betas = torch.from_numpy(betas)
    elif betas is None:
        betas = torch.zeros([pose.shape[0], 10])

    if index != -1:
        pose = pose[index:index+1]

        if trans is not None:
            trans = trans[index:index+1]

        betas = betas[index:index+1]

    if mode == "smplx":
        inputs = {
            "global_orient": pose[:, :3].float().to(device),
            "body_pose": pose[:, 3:66].float().to(device),
            "jaw_pose": pose[:, 66:69].float().to(device),
            "leye_pose": pose[:, 69:72].float().to(device),
            "reye_pose": pose[:, 72:75].float().to(device),
            "left_hand_pose":pose[:, 75:120].float().to(device),
            "right_hand_pose":pose[:, 120:].float().to(device),
        }
    elif mode == "smplh":
        inputs = {
            "global_orient": pose[:, :3].float().to(device),
            "body_pose": pose[:, 3:66].float().to(device),
            "left_hand_pose":pose[:, 66:111].float().to(device),
            "right_hand_pose":pose[:, 111:].float().to(device),
        } 
    elif mode == "smpl":
        inputs = {
            "global_orient": pose[:, :3].float().to(device),
            "body_pose": pose[:, 3:].float().to(device),
        }      

    if trans is not None:
        inputs["transl"] = trans[:, :].float().to(device)
    else:
        print("No Translation Information")

    inputs["betas"] = betas[:, :].float().to(device)

    return inputs