import torch
import numpy as np
from SMPLX.visualize_joint2smpl.simplify_loc2rot import joints2smpl
from dataset.smr.recover_smr import recover_pose_from_smr
from dataset.t2m.recover_joints import recover_from_ric as recover_joints
from dataset.utils.read_from_npy import npy2info, info2dict
from SMPLX import smplx
from dataset.utils.rotation_conversions import *
def fit2smpl(motion, smpl_path, device="cuda"):
    '''
    输入 [nframes, 22, 3]
    '''
    print(">>>>>>>>>>>>>>> fit joints to smpl >>>>>>>>>>>>>>>>>>>>")
    frames = motion.shape[0]
    j2s = joints2smpl(num_frames=frames, device=device, model_path=smpl_path)
    motion_tensor, translation = j2s.joint2smpl(motion)
    return motion_tensor, translation

def motion_rep_process(motion, rep, smpl_path, device="cuda", out_format="pose"):
    '''
    输入各种表达的motion
    输出 numpy 类型的 SMPL pose 表达
    '''

    if rep == "pose":   #### 直接可视化 SMPL 旋转信息
        return motion
    
    if rep == "t2m":    #### T2M 表征
        if not isinstance(motion, np.ndarray):
            motion = motion.detach().cpu().numpy()
        joints = recover_joints(motion, 22)
        motion_tensor, translation = fit2smpl(joints, smpl_path, device)
        motion_tensor = np.concatenate([motion_tensor, translation], axis=1)
        motion_tensor = motion_tensor.reshape(motion_tensor.shape[0], -1)    #### [nframes, 75]
        return motion_tensor
    
    if rep == "joints":
        motion_tensor, translation = fit2smpl(motion, smpl_path, device)
        motion_tensor = np.concatenate([motion_tensor, translation], axis=1)
        motion_tensor = motion_tensor.reshape(motion_tensor.shape[0], -1)    #### [nframes, 75] 

    if rep == "JP":
        if isinstance(motion, torch.Tensor):
            motion = motion.detach().cpu().numpy()
            
        translations = [np.array([0, 0, 0])]
        final_speed = motion[:-1, :3]   ### [nframes-1, 3]
        for i in range(final_speed.shape[0]):
            new_translation = translations[-1] + final_speed[i]
            translations.append(new_translation)
        translations = np.stack(translations, axis=0)   ### [nframes, 3]

        if out_format == "pose":
            rotation = motion[:, 66:].reshape(motion.shape[0], -1, 6)   ### [nframes, 22, 6]
            rotation = torch.from_numpy(rotation)
            rotation = rotation_6d_to_matrix(rotation)
            rotation = matrix_to_axis_angle(rotation)   ### [nframes, 22, 3]
            final_rotation = np.zeros([rotation.shape[0], 24, 3])
            final_rotation[:, :22, :] = rotation.numpy()
            final_rotation = final_rotation.reshape(rotation.shape[0], -1)  ### [nframes, 72]
            motion_tensor = np.concatenate([final_rotation, translations], axis=1)  ### [nframes, 75]
        elif out_format == "joints":
            joints = motion[:, 3:66].reshape(motion.shape[0], -1, 3)
            translation = translations[:, None, :]
            joints += translation
            joints = np.concatenate([translation, joints], axis=1)  ### [nframes, 22, 3]
            motion_tensor, translation = fit2smpl(joints, smpl_path, device)
            motion_tensor = np.concatenate([motion_tensor, translation], axis=1)
            motion_tensor = motion_tensor.reshape(motion_tensor.shape[0], -1)    #### [nframes, 75]             
        elif out_format == "mixture":
            joints = motion[:, 3:66].reshape(motion.shape[0], -1, 3)
            translation = translations[:, None, :]
            joints += translation
            joints = np.concatenate([translation, joints], axis=1)  ### [nframes, 22, 3]
            motion_tensor, translation = fit2smpl(joints, smpl_path, device)
            motion_tensor = np.concatenate([motion_tensor, translation], axis=1)

            rotation = motion[:, 66:].reshape(motion.shape[0], -1, 6)   ### [nframes, 22, 6]
            rotation = torch.from_numpy(rotation)
            rotation = rotation_6d_to_matrix(rotation)
            rotation = matrix_to_axis_angle(rotation)   ### [nframes, 22, 3]

            replace_index = [7, 8, 10, 11, 12, 15, 20, 21]
            motion_tensor = motion_tensor.reshape(motion_tensor.shape[0], -1, 3)
            motion_tensor[:, replace_index, :] = rotation[:, replace_index, :]
            motion_tensor = motion_tensor.reshape(motion_tensor.shape[0], -1)    #### [nframes, 75]          

    if rep == "smr":
        if not isinstance(motion, np.ndarray):
            motion = motion.detach().cpu().numpy()
        pose, joints = recover_pose_from_smr(motion, 22)
        pose = pose.reshape(pose.shape[0], -1, 3)   ### [nframes, 25, 3]
        motion_tensor, translation = fit2smpl(joints, smpl_path, device)
        motion_tensor = np.concatenate([motion_tensor, translation], axis=1) 
        motion_tensor = motion_tensor.reshape(motion_tensor.shape[0], -1, 3)   ### [nframes, 25, 3]### [nframes, 25, 3]
        replace = [12, 15, 20, 21]
        motion_tensor[:, replace, :] = pose[:, replace, :]
        motion_tensor = motion_tensor.reshape(motion_tensor.shape[0], -1)
    
    return motion_tensor

def pose2mesh(motions, smpl_path, device="cuda"):
    motions, trans, gender, betas = npy2info(motions, num_shapes=10)
    betas = None
    gender = "neutral"

    if motions.shape[1] == 72:
        mode = "smpl"
    elif motions.shape[1] == 156:
        mode = "smplh"
    elif motions.shape[1] == 165:
        mode = "smplx"
    
    print("Visualize Mode -> ", mode, "SMPL_PATH ->", smpl_path)
    model = smplx.create(smpl_path, model_type=mode,
                        gender=gender, use_face_contour=True,
                        num_betas=10,
                        num_expression_coeffs=10,
                        ext="npz", use_pca=False, batch_size=motions.shape[0])
    model = model.to(device)
    inputs = info2dict(motions, trans, betas, mode, device)
    output = model(**inputs)
    vertices = output.vertices.cpu().detach().numpy()
    joints = output.joints.cpu().detach().numpy()

    return vertices, joints, model.faces