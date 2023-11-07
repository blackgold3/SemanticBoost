from torch import nn
import torch
import numpy as np
from SMPLX.visualize_joint2smpl.simplify_loc2rot import joints2smpl
from mdm.hybrik_loc2rot import HybrIKJointsToRotmat
from SMPLX import smplx
from SMPLX.read_from_npy import npy2info, info2dict
from SMPLX.rotation_conversions import *
from mdm.dataset.recover_smr import *
from mdm.dataset.recover_joints import recover_from_ric as recover_joints
from mdm.dataset.paramUtil import t2m_kinematic_chain
from mdm.plot3d import plot_3d_motion
import os
import subprocess
import platform
from PIL import Image
from mdm.sample import Predictor as mdm_predictor
from TADA.anime import Animation

class Visualize(nn.Module):
    def __init__(self, **kargs):
        super(Visualize, self).__init__()
        self.mode = kargs.get("mode", "camd")
        self.predictor = mdm_predictor(**kargs)
        self.rep = self.predictor.rep
        self.smpl_path = kargs.get("smpl_path")
        self.device = kargs.get("device", "cpu")
        self.rotate = kargs.get("rotate", 0)
        self.pose_generator = HybrIKJointsToRotmat()
        self.path = kargs["path"]

        self.tada_base = kargs.get("tada_base", None)
        self.tada_role = kargs.get("tada_role", None)

        if self.tada_base is not None and self.tada_role is not None:
            self.anime = Animation(self.tada_role, self.tada_base, self.device)
            self.face = None
        else:
            self.face = np.load(os.path.join(self.path["dataset_dir"], "smplh.faces"))
            self.anime = None

    def fit2smpl(self, motion, mode="fast"):
        print(">>>>>>>>>>>>>>> fit joints to smpl >>>>>>>>>>>>>>>>>>>>")
        if mode == "slow":
            frames = motion.shape[0]
            j2s = joints2smpl(num_frames=frames, device=self.device, model_path=self.smpl_path, json_dict=self.path)
            motion_tensor, translation = j2s.joint2smpl(motion)
        else:
            translation = motion[:, 0:1, :] - motion[0, 0:1, :]
            motion = self.pose_generator(motion)
            motion = torch.from_numpy(motion)
            hand = torch.eye(3).unsqueeze(0).unsqueeze(0).repeat(motion.shape[0], 2, 1, 1)
            motion = torch.cat([motion, hand], dim=1)
            motion_tensor = matrix_to_axis_angle(motion)
            motion_tensor = motion_tensor.numpy()

        return motion_tensor, translation

    def predict(self, sentence, path, render_mode="pyrender", joint_path=None, smpl_path=None):
        motion_tensor = self.predictor.predict(sentence, 1, path)
        if self.rep == "t2m":
            motion_tensor = motion_tensor[0].detach().cpu().numpy()         #### [nframes, 263]

            if joint_path is not None:
                np.save(joint_path, motion_tensor)

            if render_mode == "joints":
                motion_tensor = motion_tensor
            else:
                motion_tensor, translation = self.fit2smpl(motion_tensor, render_mode.split("_")[-1])
                motion_tensor = np.concatenate([motion_tensor, translation], axis=1)    
                motion_tensor = motion_tensor.reshape(motion_tensor.shape[0], -1)

            if smpl_path is not None:
                np.save(smpl_path, motion_tensor)

        elif self.rep == "smr":
            motion_tensor = motion_tensor[0][0].detach().cpu().numpy()
            joints = recover_from_ric(motion_tensor, 22)

            if joint_path is not None:
                np.save(joint_path, joints)

            if render_mode == "joints":
                motion_tensor = joints
            else:
                pose = recover_pose_from_smr(motion_tensor, 22)
                pose = pose.reshape(pose.shape[0], -1, 3)
                motion_tensor, translation = self.fit2smpl(joints, render_mode.split("_")[-1])
                motion_tensor = np.concatenate([motion_tensor, translation], axis=1)
                motion_tensor = motion_tensor.reshape(motion_tensor.shape[0], -1, 3)
                replace = [12, 15, 20, 21]
                motion_tensor[:, replace, :] = pose[:, replace, :]
                motion_tensor = motion_tensor.reshape(motion_tensor.shape[0], -1)
        
            if smpl_path is not None:
                np.save(smpl_path, motion_tensor)

        return motion_tensor.astype(np.float32)

    def joints_process(self, joints, text, width=1024, height=1024):
        os.makedirs("temp", exist_ok=True)
        plot_3d_motion(t2m_kinematic_chain, joints, text, figsize=(width/100, height/100))
        files = os.listdir("temp")
        files = sorted(files)
        pics = []
        for i in range(len(files)):
            pic = Image.open(os.path.join("temp", files[i]))
            pic = np.asarray(pic)
            pics.append(pic.copy())
        
        cmd = "rm -r temp"
        subprocess.call(cmd, shell=platform.system() != 'Windows')
        pics = np.stack(pics, axis=0)
        return pics


    def pyrender_process(self, vertices, height=1024, weight=1024, follow=False):
        import trimesh
        from trimesh import Trimesh
        import pyrender
        from pyrender.constants import RenderFlags
        import os
        os.environ['PYOPENGL_PLATFORM'] = "egl"
        from shapely import geometry
        from tqdm import tqdm
    
        faces = self.face

        vertices = vertices.astype(np.float32)
        MINS = np.min(np.min(vertices, axis=0), axis=0)
        MAXS = np.max(np.max(vertices, axis=0), axis=0)

        #################### position initial at zero point
        vertices[:, :, 0] -= (MAXS + MINS)[0] / 2       #### x 
        vertices[:, :, 1] -= (MAXS + MINS)[1] / 2       #### y
        vertices[:, :, 2] -= (MAXS + MINS)[2] / 2       #### z

        MINS = np.min(np.min(vertices, axis=0), axis=0)
        MAXS = np.max(np.max(vertices, axis=0), axis=0)
        MEANS = np.mean(vertices, axis=1)   ### [frames, 3]
        pics = []

        ############### ground initial ###########
        r = pyrender.OffscreenRenderer(weight, height)

        for i in tqdm(range(vertices.shape[0])):
            end_color = np.array([255, 192, 203]) / 255.0

            bg_color = [0, 0, 0, 0.5]
            scene = pyrender.Scene(bg_color=bg_color, ambient_light=(0.4, 0.4, 0.4))

            if self.anime is None:
                mesh = Trimesh(vertices=vertices[i, :, :].tolist(), faces=faces) 
                base_color = end_color.tolist()
                material = pyrender.MetallicRoughnessMaterial(
                    metallicFactor=0.5, roughnessFactor=0.7,
                    alphaMode='OPAQUE',
                    baseColorFactor=base_color
                )
                mesh = pyrender.Mesh.from_trimesh(mesh, material=material)
            else:
                mesh = Trimesh(vertices=vertices[i, :, :].tolist(), faces=faces, visual=self.anime.trimesh_visual, process=False)
                mesh = pyrender.Mesh.from_trimesh(mesh, smooth=True, material=None)   

            scene.add(mesh)

            ########################### ground ##################
            c = np.pi / 2
            light = pyrender.DirectionalLight(color=[1,1,1], intensity=200)
            light_pose = np.eye(4)
            light_pose[:3, 3] = [0, -1, 1]
            scene.add(light, pose=light_pose.copy())
            light_pose[:3, 3] = [0, 1, 1]
            scene.add(light, pose=light_pose.copy())
            light_pose[:3, 3] = [1, 1, 2]
            scene.add(light, pose=light_pose.copy())

            ################ camera ##############
            if follow:
                camera = pyrender.PerspectiveCamera(yfov=(np.pi / 3.0))
                c = -np.pi / 12
                mean = MEANS[i]
                scene.add(camera, pose=[[ 1, 0, 0, mean[0]],
                                        [ 0, np.cos(c), -np.sin(c), mean[1]+ 0.5],
                                        [ 0, np.sin(c), np.cos(c), mean[2] + 2.5],
                                        [ 0, 0, 0, 1]
                                        ])
            else:
                camera = pyrender.PerspectiveCamera(yfov=(np.pi / 3.0))
                c = -np.pi / 12
                scene.add(camera, pose=[[ 1, 0, 0, 0],
                                        [ 0, np.cos(c), -np.sin(c), (MAXS[1] + MINS[1])/ 2 + 0.5],
                                        [ 0, np.sin(c), np.cos(c), MAXS[2] + 2.5],
                                        [ 0, 0, 0, 1]
                                        ])
                
            pic, _ = r.render(scene, flags=RenderFlags.RGBA)
            pics.append(pic)

        pics = np.stack(pics, axis=0)
        return pics

    @torch.no_grad()
    def get_mesh(self, motions):
        if self.anime is not None:
            vertices, faces = self.anime.forward_mdm(motions)
            joints = vertices
            self.face = faces
        else:
            motions, trans, gender, betas = npy2info(motions, 10)

            betas = None
            gender = "neutral"

            if motions.shape[1] == 72:
                mode = "smpl"
            elif motions.shape[1] == 156:
                mode = "smplh"
            elif motions.shape[1] == 165:
                motions = np.concatenate([motions[:, :66], motions[:, 75::]], axis=1)
                mode = "smplh"

            if self.rotate != 0:
                motions = motions.reshape(motions.shape[0], -1, 3)
                motions = torch.from_numpy(motions).float()
                first_frame_root_pose_matrix = axis_angle_to_matrix(motions[0][0])
                all_root_poses_matrix = axis_angle_to_matrix(motions[:, 0, :])
                aligned_root_poses_matrix = torch.matmul(torch.transpose(first_frame_root_pose_matrix, 0, 1),
                                                            all_root_poses_matrix)
                motions[:, 0, :] = matrix_to_axis_angle(aligned_root_poses_matrix)
                motions = motions.reshape(motions.shape[0], -1)
                motions = motions.numpy()

            print("Visualize Mode -> ", mode)
            model = smplx.create(self.smpl_path, model_type=mode,
                                gender=gender, use_face_contour=True,
                                num_betas=10,
                                num_expression_coeffs=10,
                                ext="npz", use_pca=False, batch_size=motions.shape[0])
            model = model.eval().to(self.device)

            inputs = info2dict(motions, trans, betas, mode, self.device)

            output = model(**inputs)

            vertices = output.vertices.cpu().numpy()
            joints = output.joints.cpu().numpy()

        return vertices, joints