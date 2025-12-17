from os.path import join as pjoin
from dataset.t2m.common.skeleton import Skeleton
import numpy as np
from dataset.t2m.common.quaternion import *
from dataset.t2m.paramUtil import *
import torch
from tqdm import tqdm
import os
from SMPLX import smplx
from dataset.utils.read_from_npy import npy2info, info2dict
from dataset.utils.reverse import reverse_joints as swap_left_right

def uniform_skeleton(positions, target_offset, n_raw_offsets, kinematic_chain, l_idx1, l_idx2, face_joint_indx):
    src_skel = Skeleton(n_raw_offsets, kinematic_chain, 'cpu')
    src_offset = src_skel.get_offsets_joints(torch.from_numpy(positions[0]))
    src_offset = src_offset.numpy()
    tgt_offset = target_offset.numpy()
    # print(src_offset)
    # print(tgt_offset)
    '''Calculate Scale Ratio as the ratio of legs'''
    src_leg_len = np.abs(src_offset[l_idx1]).max() + np.abs(src_offset[l_idx2]).max()
    tgt_leg_len = np.abs(tgt_offset[l_idx1]).max() + np.abs(tgt_offset[l_idx2]).max()

    scale_rt = tgt_leg_len / src_leg_len
    # print(scale_rt)
    src_root_pos = positions[:, 0]
    tgt_root_pos = src_root_pos * scale_rt

    '''Inverse Kinematics'''
    quat_params = src_skel.inverse_kinematics_np(positions, face_joint_indx)
    # print(quat_params.shape)

    '''Forward Kinematics'''
    src_skel.set_offset(target_offset)
    new_joints = src_skel.forward_kinematics_np(quat_params, tgt_root_pos)
    return new_joints

def process_file(positions, feet_thre, tgt_offsets, n_raw_offsets, kinematic_chain, l_idx1, l_idx2, face_joint_indx, fid_l, fid_r):
    '''Uniform Skeleton'''
    positions = uniform_skeleton(positions, tgt_offsets, n_raw_offsets, kinematic_chain, l_idx1, l_idx2, face_joint_indx)

    '''Put on Floor'''
    floor_height = positions.min(axis=0).min(axis=0)[1]
    positions[:, :, 1] -= floor_height


    '''XZ at origin'''
    root_pos_init = positions[0]
    root_pose_init_xz = root_pos_init[0] * np.array([1, 0, 1])
    positions = positions - root_pose_init_xz

    '''All initially face Z+'''
    r_hip, l_hip, sdr_r, sdr_l = face_joint_indx
    across1 = root_pos_init[r_hip] - root_pos_init[l_hip]
    across2 = root_pos_init[sdr_r] - root_pos_init[sdr_l]
    across = across1 + across2
    across = across / np.sqrt((across ** 2).sum(axis=-1))[..., np.newaxis]

    forward_init = np.cross(np.array([[0, 1, 0]]), across, axis=-1)
    forward_init = forward_init / np.sqrt((forward_init ** 2).sum(axis=-1))[..., np.newaxis]

    target = np.array([[0, 0, 1]])
    root_quat_init = qbetween_np(forward_init, target)
    root_quat_init = np.ones(positions.shape[:-1] + (4,)) * root_quat_init

    positions = qrot_np(root_quat_init, positions)

    '''New ground truth positions'''
    global_positions = positions.copy()

    """ Get Foot Contacts """

    def foot_detect(positions, thres):
        velfactor, heightfactor = np.array([thres, thres]), np.array([3.0, 2.0])

        feet_l_x = (positions[1:, fid_l, 0] - positions[:-1, fid_l, 0]) ** 2
        feet_l_y = (positions[1:, fid_l, 1] - positions[:-1, fid_l, 1]) ** 2
        feet_l_z = (positions[1:, fid_l, 2] - positions[:-1, fid_l, 2]) ** 2
        #     feet_l_h = positions[:-1,fid_l,1]
        #     feet_l = (((feet_l_x + feet_l_y + feet_l_z) < velfactor) & (feet_l_h < heightfactor)).astype(np.float)
        feet_l = ((feet_l_x + feet_l_y + feet_l_z) < velfactor).astype(np.float32)

        feet_r_x = (positions[1:, fid_r, 0] - positions[:-1, fid_r, 0]) ** 2
        feet_r_y = (positions[1:, fid_r, 1] - positions[:-1, fid_r, 1]) ** 2
        feet_r_z = (positions[1:, fid_r, 2] - positions[:-1, fid_r, 2]) ** 2
        #     feet_r_h = positions[:-1,fid_r,1]
        #     feet_r = (((feet_r_x + feet_r_y + feet_r_z) < velfactor) & (feet_r_h < heightfactor)).astype(np.float)
        feet_r = (((feet_r_x + feet_r_y + feet_r_z) < velfactor)).astype(np.float32)
        return feet_l, feet_r
    #
    feet_l, feet_r = foot_detect(positions, feet_thre)

    '''Quaternion and Cartesian representation'''
    r_rot = None

    def get_rifke(positions):
        '''Local pose'''
        positions[..., 0] -= positions[:, 0:1, 0]
        positions[..., 2] -= positions[:, 0:1, 2]
        '''All pose face Z+'''
        positions = qrot_np(np.repeat(r_rot[:, None], positions.shape[1], axis=1), positions)
        return positions

    def get_cont6d_params(positions):
        skel = Skeleton(n_raw_offsets, kinematic_chain, "cpu")
        quat_params = skel.inverse_kinematics_np(positions, face_joint_indx, smooth_forward=True)

        '''Quaternion to continuous 6D'''
        cont_6d_params = quaternion_to_cont6d_np(quat_params)
        # (seq_len, 4)
        r_rot = quat_params[:, 0].copy()
        #     print(r_rot[0])
        '''Root Linear Velocity'''
        # (seq_len - 1, 3)
        velocity = (positions[1:, 0] - positions[:-1, 0]).copy()
        #     print(r_rot.shape, velocity.shape)
        velocity = qrot_np(r_rot[1:], velocity)
        '''Root Angular Velocity'''
        # (seq_len - 1, 4)
        r_velocity = qmul_np(r_rot[1:], qinv_np(r_rot[:-1]))
        # (seq_len, joints_num, 4)
        return cont_6d_params, r_velocity, velocity, r_rot

    cont_6d_params, r_velocity, velocity, r_rot = get_cont6d_params(positions)
    positions = get_rifke(positions)

    '''Root height'''
    root_y = positions[:, 0, 1:2]

    '''Root rotation and linear velocity'''
    r_velocity = np.arcsin(r_velocity[:, 2:3])
    l_velocity = velocity[:, [0, 2]]
    root_data = np.concatenate([r_velocity, l_velocity, root_y[:-1]], axis=-1)

    '''Get Joint Rotation Representation'''
    rot_data = cont_6d_params[:, 1:].reshape(len(cont_6d_params), -1)

    '''Get Joint Rotation Invariant Position Represention'''
    ric_data = positions[:, 1:].reshape(len(positions), -1)

    '''Get Joint Velocity Representation'''

    local_vel = qrot_np(np.repeat(r_rot[:-1, None], global_positions.shape[1], axis=1),
                        global_positions[1:] - global_positions[:-1])
    local_vel = local_vel.reshape(len(local_vel), -1)

    data = root_data
    data = np.concatenate([data, ric_data[:-1]], axis=-1)
    data = np.concatenate([data, rot_data[:-1]], axis=-1)
    data = np.concatenate([data, local_vel], axis=-1)
    data = np.concatenate([data, feet_l, feet_r], axis=-1)

    return data

def extract_poses_from_smpls(model_dir, file_paths, output_path, device="cuda", trans_matrix=None, amass=False, reverse=True):
    os.makedirs(output_path, exist_ok=True)
    for root, folder, files in os.walk(file_paths):
        for i in tqdm(range(len(files))):
            input_path = os.path.join(root, files[i])
            target_path = os.path.join(output_path, files[i])
            ext = input_path.split(".")[-1]
            if ext == "npz":
                poses = np.load(input_path)
                gender = str(poses["gender"])
                poses = poses["poses"]
                trans = poses["trans"]
                betas = poses["betas"]
            else:
                poses, trans, gender, betas = npy2info(input_path, 10)

            if poses.shape[1] == 72:
                mode_type = "smpl"
                gender = "neutral"
                if amass:       #### humanact12
                    poses = poses.reshape(poses.shape[0], -1, 3)
                    trans_matrix_humanact12 = np.array([
                            [1, 0, 0],
                            [0, 0, 1],
                            [0, 1, 0]
                        ]).astype(poses.dtype)      ##### humanact12 data has a wrong transpose matrix, need to fix

                    poses = poses.dot(trans_matrix_humanact12)
                    poses = poses.dot(trans_matrix)
                    np.save(target_path, poses)    
                    ########### data augment #################
                    if reverse:
                        data_m = swap_left_right(poses)
                        target_path_M = os.path.join(output_path, "M" + files[i])
                        np.save(target_path_M, data_m)
                    continue

            elif poses.shape[1] == 156:
                mode_type = "smplh"
            elif poses.shape[1] == 165:
                mode_type = "smplx"

            model = smplx.create(model_dir, model_type=mode_type,
                                gender=gender, use_face_contour=True,
                                num_betas=10,
                                num_expression_coeffs=10,
                                ext="npz", use_pca=False, batch_size=poses.shape[0])
            model = model.eval().to(device)

            inputs = info2dict(poses, trans, betas, mode_type, device)
            output = model(**inputs)
            joint_loc = output.joints.detach().cpu().numpy()

            if mode_type == "smpl":
                joint_loc = joint_loc[:, :24, :]
            elif mode_type == "smplh":
                joint_loc = joint_loc[:, :52, :]
            elif mode_type == "smplx":
                joint_loc = joint_loc[:, :55, :]

            if trans_matrix is not None:
                joint_loc = np.dot(joint_loc, trans_matrix)
            
            np.save(target_path, joint_loc)
            ########### data augment #################
            if reverse:
                data_m = swap_left_right(joint_loc)
                target_path_M = os.path.join(output_path, "M" + files[i])
                np.save(target_path_M, data_m)

def transfer_joints_to_t2m(joints_path, save_dir, joints_num=22):
    os.makedirs(save_dir, exist_ok=True)
    l_idx1, l_idx2 = 5, 8
    fid_r, fid_l = [8, 11], [7, 10]
    face_joint_indx = [2, 1, 17, 16]
    data_dir = joints_path
    n_raw_offsets = torch.from_numpy(t2m_raw_offsets)
    kinematic_chain = t2m_kinematic_chain
    # Get offsets of target skeleton
    example_data = np.load("dataset/t2m/000021.npy")      #### standard skeleton
    example_data = example_data.reshape(len(example_data), -1, 3)
    example_data = torch.from_numpy(example_data)
    tgt_skel = Skeleton(n_raw_offsets, kinematic_chain, 'cpu')
    tgt_offsets = tgt_skel.get_offsets_joints(example_data[0])

    ############### begin to transfer to humanml3d representations ########
    source_list = os.listdir(data_dir)
    for source_file in tqdm(source_list):
        # target_path = pjoin(save_dir, source_file)
        # if os.path.exists(target_path):
        #     continue
        source_data = np.load(os.path.join(data_dir, source_file))[:, :joints_num, :]
        try:
            data = process_file(source_data, 0.002, tgt_offsets, n_raw_offsets, kinematic_chain, 
                                l_idx1, l_idx2, face_joint_indx, fid_l, fid_r)
            np.save(pjoin(save_dir, source_file), data)
        except Exception as e:
            print(source_file)
            print(e)


if __name__ == "__main__":
    model_dir = "/data/TTA/data/body_models"
    smpls_path = "/data/TTA/data/refer_joints"
    output_path = "/data/TTA/data/NewDataset/humanml3d/smplh_joints"
    save_dir2 = '/data/TTA/data/NewDataset/humanml3d/new_joint_vecs/'
    os.makedirs(output_path, exist_ok=True)
    os.makedirs(save_dir2, exist_ok=True)
    device = "cuda"
    joints_num = 22

    extract_poses_from_smpls(model_dir, smpls_path, output_path, device)
    transfer_joints_to_t2m(output_path, save_dir2, joints_num)
