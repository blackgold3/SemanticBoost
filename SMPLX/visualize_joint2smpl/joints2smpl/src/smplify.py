import torch
import os, sys
import pickle

sys.path.append(os.path.dirname(__file__))
from customloss import (camera_fitting_loss_3d,
                        body_fitting_loss_3d, 
                        )
from prior import MaxMixturePrior
from SMPLX.visualize_joint2smpl.joints2smpl.src import config
from tqdm import tqdm

@torch.no_grad()
def guess_init_3d(model_joints, 
                  j3d, 
                  joints_category="orig"):
    """Initialize the camera translation via triangle similarity, by using the torso joints        .
    :param model_joints: SMPL model with pre joints
    :param j3d: 25x3 array of Kinect Joints
    :returns: 3D vector corresponding to the estimated camera translation
    """
    # get the indexed four
    gt_joints = ['RHip', 'LHip', 'RShoulder', 'LShoulder']
    gt_joints_ind = [config.JOINT_MAP[joint] for joint in gt_joints]
    
    if joints_category=="orig":
        joints_ind_category = [config.JOINT_MAP[joint] for joint in gt_joints]
    elif joints_category=="AMASS":
        joints_ind_category = [config.AMASS_JOINT_MAP[joint] for joint in gt_joints] 
    else:
        print("NO SUCH JOINTS CATEGORY!") 

    sum_init_t = (j3d[:, joints_ind_category] - model_joints[:, gt_joints_ind]).sum(dim=1)
    init_t = sum_init_t / 4.0
    return init_t


# SMPLIfy 3D
class SMPLify3D():
    """Implementation of SMPLify, use 3D joints."""

    def __init__(self,
                 smplxmodel,
                 step_size=1.0,
                 num_iters=100,
                 joints_category="orig",
                 device=torch.device('cuda:0'),
                 ):

        # Store options
        self.device = device
        self.step_size = step_size

        self.num_iters = num_iters
        # GMM pose prior
        self.pose_prior = MaxMixturePrior(prior_folder=config.GMM_MODEL_DIR,
                                          num_gaussians=8,
                                          dtype=torch.float32).to(device)
        # reLoad SMPL-X model
        self.smpl = smplxmodel

        self.model_faces = smplxmodel.faces_tensor.view(-1)

        # select joint joint_category
        self.joints_category = joints_category
        
        if joints_category=="orig":
            self.smpl_index = config.full_smpl_idx
            self.corr_index = config.full_smpl_idx 
        elif joints_category=="AMASS":
            self.smpl_index = config.amass_smpl_idx
            self.corr_index = config.amass_idx
        else:
            self.smpl_index = None 
            self.corr_index = None
            print("NO SUCH JOINTS CATEGORY!")

    # ---- get the man function here ------
    def __call__(self, init_pose, init_betas, init_cam_t, j3d, conf_3d=1.0, seq_ind=0):
        """Perform body fitting.
        Input:
            init_pose: SMPL pose estimate
            init_betas: SMPL betas estimate
            init_cam_t: Camera translation estimate
            j3d: joints 3d aka keypoints
            conf_3d: confidence for 3d joints
			seq_ind: index of the sequence
        Returns:
            vertices: Vertices of optimized shape
            joints: 3D joints of optimized shape
            pose: SMPL pose parameters of optimized shape
            betas: SMPL beta parameters of optimized shape
            camera_translation: Camera translation
        """
        # Split SMPL pose to body pose and global orientation
        body_pose = init_pose[:, 3:].detach().clone()
        global_orient = init_pose[:, :3].detach().clone()
        betas = init_betas.detach().clone()

        # use guess 3d to get the initial
        smpl_output = self.smpl(global_orient=global_orient,
                                body_pose=body_pose,
                                betas=betas)
        model_joints = smpl_output.joints

        init_cam_t = guess_init_3d(model_joints, j3d, self.joints_category).unsqueeze(1).detach()
        camera_translation = init_cam_t.clone()
        
        preserve_pose = init_pose[:, 3:].detach().clone()
       # -------------Step 1: Optimize camera translation and body orientation--------
        # Optimize only camera translation and body orientation
        body_pose.requires_grad = False
        betas.requires_grad = True
        global_orient.requires_grad = True
        camera_translation.requires_grad = True

        camera_opt_params = [betas, global_orient, camera_translation]

        camera_optimizer = torch.optim.LBFGS(camera_opt_params, max_iter=10,
                                                lr=self.step_size, line_search_fn='strong_wolfe')
        cycle = tqdm(range(10))
        for i in cycle:
            def closure():
                camera_optimizer.zero_grad()
                smpl_output = self.smpl(global_orient=global_orient,
                                        body_pose=body_pose,
                                        betas=betas)
                model_joints = smpl_output.joints
                loss = camera_fitting_loss_3d(model_joints, camera_translation,
                                                init_cam_t, j3d, self.joints_category)

                loss.backward()
                return loss
            
            camera_optimizer.step(closure)

        # Fix camera translation after optimizing camera
        # --------Step 2: Optimize body joints --------------------------
        # Optimize only the body pose and global orientation of the body
        body_pose.requires_grad = True
        global_orient.requires_grad = True
        camera_translation.requires_grad = True
        betas.requires_grad = True
        body_opt_params = [body_pose, betas, global_orient, camera_translation]

        cycle = tqdm(range(self.num_iters))
        body_optimizer = torch.optim.LBFGS(body_opt_params, max_iter=self.num_iters,
                                            lr=self.step_size, line_search_fn='strong_wolfe')
        for i in cycle:
            def closure():
                body_optimizer.zero_grad()
                smpl_output = self.smpl(global_orient=global_orient,
                                        body_pose=body_pose,
                                        betas=betas)
                model_joints = smpl_output.joints
                model_vertices = smpl_output.vertices

                loss = body_fitting_loss_3d(body_pose, preserve_pose, betas, model_joints[:, self.smpl_index], camera_translation,
                                            j3d[:, self.corr_index], self.pose_prior,
                                            joints3d_conf=conf_3d,
                                            joint_loss_weight=600.0,
                                            pose_preserve_weight=5.0,
                                            use_collision=False, 
                                            model_vertices=model_vertices, model_faces=self.model_faces,
                                            search_tree=None, pen_distance=None, filter_faces=None)
                loss.backward()
                return loss

            body_optimizer.step(closure)


        pose = torch.cat([global_orient, body_pose], dim=-1).detach()
        return pose
