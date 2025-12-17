import os
import subprocess
import platform
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp
import numpy as np
import torch
from dataset.utils.rotation_conversions import *
from sever_ft_control_inbetween import sever_motion_concat as server_ft_inbetween
from sever_motion_concat_inpainting import sever_motion_concat as server_inpainting
from dataset.smr.motion_representation import convert2smr
from dataset.utils.convert_fps import convert_to_target_fps
from sample.t2m_post_process import pose2mesh, motion_rep_process
from dataset.utils.rotation_conversions import *
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp

class sever_smplx_concat(object):
    def __init__(self, mode, checkpoints, target_dir, model_path):
        self.mode = mode
        self.target_dir = target_dir
        os.makedirs(self.target_dir, exist_ok=True)
        self.begin_capture = 20
        self.end_capture = 20

        if self.mode == "inpainting":
            self.blend_size = 10
            self.smpl_server = server_inpainting(checkpoints, self.blend_size, self.begin_capture, self.end_capture, 0, target_dir + "/middle_res")
        elif self.mode == "inbetween":
            self.blend_size = 0
            self.smpl_server = server_ft_inbetween(checkpoints, self.begin_capture, self.end_capture, 0, target_dir + "/middle_res")
        
        self.device = "cuda"
        self.model_path = model_path
        self.gender = 2

    def data_handler(self, smplx_motion, target_name):
        '''
        smplx_motion: npz 文件
        '''
        pose = smplx_motion["poses"]
        trans = smplx_motion["trans"]
        try:
            fps = int(smplx_motion['mocap_framerate'])
        except:
            fps = 20

        try:
            gender = smplx_motion["gender"].tolist()
            if gender == "female":
                self.gender = 0
            elif gender == "male":
                self.gender = 1
            else:
                self.gender = 2
        except:
            self.gender = 2

        ##################### rotation 到正面 ##################
        pose = pose.reshape(pose.shape[0], -1, 3)
        motions = torch.from_numpy(pose).float()
        first_frame_root_pose_matrix = axis_angle_to_matrix(motions[0][0])
        all_root_poses_matrix = axis_angle_to_matrix(motions[:, 0, :])
        aligned_root_poses_matrix = torch.matmul(torch.transpose(first_frame_root_pose_matrix, 0, 1),
                                                    all_root_poses_matrix)
        motions[:, 0, :] = matrix_to_axis_angle(aligned_root_poses_matrix)
        motions = motions.reshape(motions.shape[0], -1)
        motions = motions.numpy()
        pose = motions
        trans = torch.from_numpy(trans).float()
        trans = torch.matmul(torch.transpose(first_frame_root_pose_matrix, 0, 1).float(),
                                torch.transpose(trans, 0, 1))
        trans = torch.transpose(trans, 0, 1).numpy()

        ############################### smplx 帧率转换和信息提取 ##################
        smplx_rep = np.concatenate([pose, trans], axis=1)   ### [nframes, 168]
        smplx_rep = convert_to_target_fps(smplx_rep, fps, 20)   ### [nframes, 168]
        trans = smplx_rep[:, 165:]
        smplx_extra_pose = smplx_rep[:, 66:165] ### [9 + 45 + 45 = 99]

        ############################## smr 转换 #################
        nframes = smplx_rep.shape[0]
        gender_rep = np.array([self.gender], dtype=np.float32)[None, :].repeat(nframes, axis=0)   ### [nframes, 1]
        smplx_pose = np.concatenate([smplx_rep, gender_rep], axis=1)    ### [66 + 9 + 90 + 3 + 1 = 169]

        vertices, joints, faces = pose2mesh(smplx_pose, self.model_path, self.device)
        smr_rep = convert2smr(smplx_pose, joints, 22)
        smr_path = os.path.join(self.target_dir, target_name) + ".npy"
        np.save(smr_path, smr_rep)
        return smplx_pose, smplx_extra_pose, smr_path, smr_rep

    def slerp_between_quaternions(self, q1, q2, num_frames):
        times = [0, 1]
        rotations = R.from_quat([q1, q2])
        slerp = Slerp(times, rotations)
        interpolation_times = np.linspace(0, 1, num_frames)
        interpolated_rotations = slerp(interpolation_times)
        interpolated_quats = interpolated_rotations.as_quat()
        
        return interpolated_quats

    def __call__(self, file1_path, file2_path, content_length, only_smr_rep=False):
        ########## 源文件处理，转换 fps, 保存SMPL pose 部分的 smr 参数 ###############
        file1 = np.load(file1_path)
        file2 = np.load(file2_path)
        print("============= smplx 动作预处理 ============")
        motion1, motion1_smplx_extra, motion1_smr_path, motion1_smr_rep = self.data_handler(file1, "file1")
        motion2, motion2_smplx_extra, motion2_smr_path, motion2_smr_rep = self.data_handler(file2, "file2")

        #################### 帧数处理   ################
        if self.mode == "inpainting":       ##### 去掉 blend_size 区域
            motion1 = motion1[:-self.blend_size]
            motion2 = motion2[self.blend_size:]
            motion1_smr_rep = motion1_smr_rep[:-self.blend_size + 1]
            motion2_smr_rep = motion2_smr_rep[self.blend_size - 1:]
            motion1_smplx_extra = motion1_smplx_extra[:-self.blend_size]
            motion2_smplx_extra = motion2_smplx_extra[self.blend_size:]
            blend_size = self.blend_size
        else:
            blend_size = self.blend_size
    
        #################### 生成部分，smr 表示 ########################
        print("============= 过渡身体动作生成 ============")
        syntheiszed_motion = self.smpl_server(motion1_smr_path, motion2_smr_path, content_length, None, False)
        print("============= 过渡身体动作转换 ============")

        if only_smr_rep:
            syntheiszed_motion = np.concatenate([motion1_smr_rep, syntheiszed_motion, motion2_smr_rep], axis=0) ### [nframes, 75]

        syntheiszed_motion = motion_rep_process(syntheiszed_motion, "smr", self.model_path, self.device)    #### [nframes, 75]

        ################# 手部插值  ##################
        print("============= 过渡手部插值 ============")
        begin_pose = motion1_smplx_extra[-1:]   #### [1, 99], 动作1的最后一帧
        end_pose = motion2_smplx_extra[0:1]     #### [1, 99]，动作2的第一帧

        begin_pose = torch.from_numpy(begin_pose).reshape([-1, 3])
        begin_pose = axis_angle_to_quaternion(begin_pose).numpy()       #### [1, 33, 4]

        end_pose = torch.from_numpy(end_pose).reshape([-1, 3])
        end_pose = axis_angle_to_quaternion(end_pose).numpy().reshape(-1, 4)

        rotation_joints = []
        for i in range(begin_pose.shape[0]):
            curr_begin = begin_pose[i]
            curr_end = end_pose[i]
            curr_insert_quanti = self.slerp_between_quaternions(curr_begin, curr_end, content_length + 2 + blend_size * 2)
            rotation_joints.append(curr_insert_quanti) 

        rotation_joints = np.stack(rotation_joints, axis=0) #### [33, content+2, 4]
        rotation_joints = rotation_joints[:, 1:-1, :]   #### [33, content, 4]
        rotation_joints = torch.from_numpy(rotation_joints).permute(1, 0, 2).reshape(-1, 4)
        rotation_joints = quaternion_to_axis_angle(rotation_joints) ### [content * 33, 3]
        rotation_joints = rotation_joints.reshape(content_length + blend_size * 2, -1, 3).reshape(content_length + blend_size * 2, -1)

        if only_smr_rep:
            rotation_joints = np.concatenate([motion1_smplx_extra, rotation_joints, motion2_smplx_extra], axis=0)

        ############################# 拼接 ##############
        print("============= 动作拼接 ============")
        synthesized_trans = syntheiszed_motion[:, -3:]  ########### [content_length, 3]
        synthesized_body = syntheiszed_motion[:, :66]   ###### [content_length, 66]
        gender_rep = np.array([self.gender], dtype=np.float32)[None, :].repeat(synthesized_body.shape[0], axis=0)   ### [nframes, 1]
        syntheiszed_smplx = np.concatenate([synthesized_body, rotation_joints, synthesized_trans, gender_rep], axis=1)  ### [nframes, 169]

        if only_smr_rep:
            final_motion = syntheiszed_smplx
        else:
            '''
            translation 对齐
            '''
            motion1_translation = motion1[:, 165:168]
            syntheiszed_translation = syntheiszed_smplx[:, 165:168]
            motion2_translation = motion2[:, 165:168]
            synthesized_translation_sub = syntheiszed_translation - syntheiszed_translation[0]
            syntheiszed_translation = synthesized_translation_sub + motion1_translation[-1:]
            motion2_translation_sub = motion2_translation - motion2_translation[0]
            motion2_translation = motion2_translation_sub + syntheiszed_translation[-1:]
            syntheiszed_smplx[:, 165:168] = syntheiszed_translation
            motion2[:, 165:168] = motion2_translation

            '''
            按照头尾帧的偏移量
            缩放每一帧的旋转变换
            '''
            motion1_body_pose = motion1[:, :66].reshape(motion1.shape[0], -1, 3)
            synthesized_body_pose = syntheiszed_smplx[:, :66].reshape(syntheiszed_smplx.shape[0], -1, 3)
            motion2_body_pose = motion2[:, :66].reshape(motion2.shape[0], -1, 3)

            real_begin_A = R.from_euler("XYZ", motion1_body_pose[-1])  #### [22, 3, 3]
            real_end_B = R.from_euler("XYZ", motion2_body_pose[0])   ### [22, 3, 3]
            curr_begin_C = R.from_euler("XYZ", synthesized_body_pose[0])
            curr_end_D = R.from_euler("XYZ", synthesized_body_pose[-1])

            q_start = real_begin_A * curr_begin_C.inv()    #### 起点偏移旋转
            q_end = real_end_B * curr_end_D.inv()          #### 终点偏移旋转, [22, 4]

            slerps = []
            times = np.linspace(0, 1, synthesized_body_pose.shape[0])        
            for j in range(synthesized_body_pose.shape[1]): 
                curr_start = q_start.as_quat()[j]
                curr_end = q_end.as_quat()[j]
                curr_roatation = R.from_quat([curr_start, curr_end])   ### [2, 4]
                curr_slerp = Slerp([0, 1], curr_roatation)
                interp_rots = curr_slerp(times)
                slerps.append(interp_rots.as_quat())
            slerps = np.stack(slerps, axis=0).transpose(1, 0, 2)    ### [22, 40, 4] -> [40, 22, 4]

            adjusted_frames = np.zeros([synthesized_body_pose.shape[0], synthesized_body_pose.shape[1], 3])
            for i in range(synthesized_body_pose.shape[0]): ### 20
                curr_rotation = R.from_euler("XYZ", synthesized_body_pose[i])   ### [22, 4]
                curr_interp = R.from_quat(slerps[i])
                real_rotation = curr_interp * curr_rotation
                adjusted_frames[i] = real_rotation.as_euler("XYZ", degrees=False)
                
            adjusted_frames = adjusted_frames.reshape(synthesized_body_pose.shape[0], -1)
            syntheiszed_smplx[:, :66] = adjusted_frames

            '''
            最终的motion 拼接
            '''
            final_motion = np.concatenate([motion1, syntheiszed_smplx, motion2], axis=0)

        ######################## 保存 ##################
        print("=========== final motion size =====", final_motion.shape)
        save_path = os.path.join(self.target_dir, "final.npy")
        np.save(save_path, final_motion)
        cmd = "python -m sample.visual_file --source {} --mode pose --target_dir {} --target_name final --render_mode pyrender --out_mode video".format(save_path, self.target_dir)
        subprocess.call(cmd, shell=platform.system() != 'Windows')

        return final_motion


if __name__ == "__main__":
    model_path = "/apdcephfs_cq11/share_1567347/share_info/kleinhe/checkpoints/body_models"

    # checkpoints = "/apdcephfs_cq11/share_1567347/share_info/kleinhe/checkpoints/text2motion/train_4alldata_text2motion_smr/S250000_F0.2012_T0.5225.pth"
    # handler = sever_smplx_concat("inpainting", checkpoints, "results/handler", model_path)
    
    checkpoints = "/apdcephfs_cq11/share_1567347/share_info/kleinhe/checkpoints/text2motion/train_ft_control_inbetween_20_20_lowlr/S040000_F0.2014_T0.5115.pth"
    handler = sever_smplx_concat("inbetween", checkpoints, "results/handler", model_path)

    file1 = "results/files/first.npz"
    file2 = "results/files/last.npz"
    handler(file1, file2, 20, only_smr_rep=False)