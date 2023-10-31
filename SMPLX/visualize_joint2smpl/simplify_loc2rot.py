import numpy as np
import os
import torch
from SMPLX import smplx
import h5py
from SMPLX.visualize_joint2smpl.joints2smpl.src.smplify import SMPLify3D
from tqdm import tqdm
import argparse
from motion.dataset.smooth import bezier_smooth

class joints2smpl:

    def __init__(self, num_frames, device, model_path=None, json_dict=None):
        self.smpl_dir = model_path
        self.device = device
        # self.device = torch.device("cpu")
        self.batch_size = num_frames
        self.num_joints = 22  # for HumanML3D
        self.joint_category = "AMASS"
        self.num_smplify_iters = 15
        self.fix_foot = False
        
        smplmodel = smplx.create(self.smpl_dir, model_type="smpl", gender="neutral", ext="pkl",
                                 batch_size=self.batch_size).to(self.device)

        # ## --- load the mean pose as original ----
        smpl_mean_file = os.path.join(json_dict["joints2smpl"], "neutral_smpl_mean_params.h5")

        file = h5py.File(smpl_mean_file, 'r')
        self.init_mean_pose = torch.from_numpy(file['pose'][:]).unsqueeze(0).repeat(self.batch_size, 1).float().to(self.device)
        self.init_mean_shape = torch.from_numpy(file['shape'][:]).unsqueeze(0).repeat(self.batch_size, 1).float().to(self.device)
        self.cam_trans_zero = torch.Tensor([0.0, 0.0, 0.0]).unsqueeze(0).to(self.device)
        #

        # # #-------------initialize SMPLify
        self.smplify = SMPLify3D(smplxmodel=smplmodel,
                            joints_category=self.joint_category,
                            num_iters=self.num_smplify_iters,
                            device=self.device)


    def npy2smpl(self, npy_path):
        out_path = npy_path.replace('.npy', '_rot.npy')
        motions = np.load(npy_path, allow_pickle=True)[None][0]
        # print_batch('', motions)
        n_samples = motions['motion'].shape[0]
        all_thetas = []
        for sample_i in tqdm(range(n_samples)):
            thetas, _ = self.joint2smpl(motions['motion'][sample_i].transpose(2, 0, 1))  # [nframes, njoints, 3]
            all_thetas.append(thetas.cpu().numpy())
        motions['motion'] = np.concatenate(all_thetas, axis=0)
        print('motions', motions['motion'].shape)

        print(f'Saving [{out_path}]')
        np.save(out_path, motions)
        exit()


    def joint2smpl(self, input_joints, init_params=None):
        if len(input_joints.shape) == 2:
            input_joints = input_joints.reshape(input_joints.shape[0], -1, 3)

        pred_pose = torch.zeros(self.batch_size, 72).to(self.device)
        pred_betas = torch.zeros(self.batch_size, 10).to(self.device)
        pred_cam_t = torch.zeros(self.batch_size, 3).to(self.device)
        keypoints_3d = torch.zeros(self.batch_size, self.num_joints, 3).to(self.device)

        # joints3d = input_joints[idx]  # *1.2 #scale problem [check first]
        keypoints_3d = torch.Tensor(input_joints).to(self.device).float()
        
        root_loc = torch.tensor(keypoints_3d[:, 0:1])   #### N * 1 * 3
        root_loc = root_loc - root_loc[[0], :, :]   ### N * 1 * 3
        root_loc = root_loc.squeeze(1).detach().cpu().numpy()
        # if idx == 0:
        if init_params is None:
            pred_betas = self.init_mean_shape
            pred_pose = self.init_mean_pose
            pred_cam_t = self.cam_trans_zero
        else:
            pred_betas = init_params['betas']
            pred_pose = init_params['pose']
            pred_cam_t = init_params['cam']

        if self.joint_category == "AMASS":
            confidence_input = torch.ones(self.num_joints)
            # make sure the foot and ankle
            if self.fix_foot == True:
                confidence_input[7] = 1.5
                confidence_input[8] = 1.5
                confidence_input[10] = 1.5
                confidence_input[11] = 1.5
        else:
            print("Such category not settle down!")

        new_opt_pose = self.smplify(
            pred_pose.detach(),
            pred_betas.detach(),
            pred_cam_t.detach(),
            keypoints_3d,
            conf_3d=confidence_input.to(self.device),
        )

        thetas = new_opt_pose.reshape(self.batch_size, 24 * 3)
        vecs = thetas.detach().cpu().numpy()

        return vecs, root_loc


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", type=str, required=True, help='Blender file or dir with blender files')
    parser.add_argument("--cuda", type=bool, default=True, help='')
    parser.add_argument("--device", type=int, default=0, help='')
    params = parser.parse_args()

    simplify = joints2smpl(device_id=params.device, cuda=params.cuda)

    if os.path.isfile(params.input_path) and params.input_path.endswith('.npy'):
        simplify.npy2smpl(params.input_path)
    elif os.path.isdir(params.input_path):
        files = [os.path.join(params.input_path, f) for f in os.listdir(params.input_path) if f.endswith('.npy')]
        for f in files:
            simplify.npy2smpl(f)