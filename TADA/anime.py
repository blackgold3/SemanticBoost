import os
import json
import pickle as pkl
import random
import argparse
import cv2
import torch
from TADA import smplx
import imageio
import numpy as np
from tqdm import tqdm
from PIL import Image
from TADA.lib.common.remesh import subdivide_inorder
from TADA.lib.common.utils import SMPLXSeg
from TADA.lib.common.lbs import warp_points
from TADA.lib.common.obj import compute_normal
import trimesh
import pyrender
from shapely import geometry
import moviepy.editor as mpy
os.environ['PYOPENGL_PLATFORM'] = "egl"

def build_new_mesh(v, f, vt, ft):
    # build a correspondences dictionary from the original mesh indices to the (possibly multiple) texture map indices
    f_flat = f.flatten()
    ft_flat = ft.flatten()
    correspondences = {}

    # traverse and find the corresponding indices in f and ft
    for i in range(len(f_flat)):
        if f_flat[i] not in correspondences:
            correspondences[f_flat[i]] = [ft_flat[i]]
        else:
            if ft_flat[i] not in correspondences[f_flat[i]]:
                correspondences[f_flat[i]].append(ft_flat[i])

    # build a mesh using the texture map vertices
    new_v = np.zeros((v.shape[0], vt.shape[0], 3))
    for old_index, new_indices in correspondences.items():
        for new_index in new_indices:
            new_v[:, new_index] = v[:, old_index]

    # define new faces using the texture map faces
    f_new = ft
    return new_v, f_new

class Animation:
    def __init__(self, ckpt_path, workspace_dir, device="cuda"):
        self.device = device
        self.SMPLXSeg = SMPLXSeg(workspace_dir)
        # load data
        init_data = np.load(os.path.join(workspace_dir, "init_body/data.npz"))
        self.dense_faces = torch.as_tensor(init_data['dense_faces'], device=self.device)
        self.dense_lbs_weights = torch.as_tensor(init_data['dense_lbs_weights'], device=self.device)
        self.unique = init_data['unique']
        self.vt = init_data['vt']
        self.ft = init_data['ft']

        model_params = dict(
            model_path=os.path.join(workspace_dir, "smplx/SMPLX_NEUTRAL_2020.npz"),
            model_type='smplx',
            create_global_orient=True,
            create_body_pose=True,
            create_betas=True,
            create_left_hand_pose=True,
            create_right_hand_pose=True,
            create_jaw_pose=True,
            create_leye_pose=True,
            create_reye_pose=True,
            create_expression=True,
            create_transl=False,
            use_pca=False,
            flat_hand_mean=False,
            num_betas=300,
            num_expression_coeffs=100,
            num_pca_comps=12,
            dtype=torch.float32,
            batch_size=1,
        )
        self.body_model = smplx.create(**model_params).to(device='cuda')
        self.smplx_face = self.body_model.faces.astype(np.int32)

        ckpt_file = os.path.join(workspace_dir, "MESH", ckpt_path, "params.pt")
        albedo_path = os.path.join(workspace_dir, "MESH", ckpt_path, "mesh_albedo.png")
        self.load_ckpt_data(ckpt_file, albedo_path)


    def load_ckpt_data(self, ckpt_file, albedo_path):
        model_data = torch.load(ckpt_file)
        self.expression = model_data["expression"] if "expression" in model_data else None
        self.jaw_pose = model_data["jaw_pose"] if "jaw_pose" in model_data else None

        self.betas = model_data['betas']
        self.v_offsets = model_data['v_offsets']
        self.v_offsets[self.SMPLXSeg.eyeball_ids] = 0.
        self.v_offsets[self.SMPLXSeg.hands_ids] = 0.

        # tex to trimesh texture
        vt = self.vt.copy()
        vt[:, 1] = 1 - vt[:, 1]
        albedo = Image.open(albedo_path)
        
        self.raw_albedo = torch.from_numpy(np.array(albedo))
        self.raw_albedo = self.raw_albedo / 255.0
        self.raw_albedo = self.raw_albedo.permute(2, 0, 1)
        
        self.trimesh_visual = trimesh.visual.TextureVisuals(
            uv=vt,
            image=albedo,
            material=trimesh.visual.texture.SimpleMaterial(
                image=albedo,
                diffuse=[255, 255, 255, 255],
                ambient=[255, 255, 255, 255],
                specular=[0, 0, 0, 255],
                glossiness=0)
        )

    def forward_mdm(self, motion):
        try:
            mdm_body_pose = motion["poses"]
            translate = torch.from_numpy(motion["trans"])
        except:
            translate = torch.from_numpy(motion[:, -3:])
            mdm_body_pose = motion[:, :-3]
            mdm_body_pose = mdm_body_pose.reshape(mdm_body_pose.shape[0], -1, 3)

        translate = translate.to(self.device)
        scan_v_posed = []
        for i, (pose, t) in tqdm(enumerate(zip(mdm_body_pose, translate))):
            body_pose = torch.as_tensor(pose[None, 1:22, :], device=self.device)
            global_orient = torch.as_tensor(pose[None, :1, :], device=self.device)
            output = self.body_model(
                betas=self.betas,
                global_orient=global_orient,
                jaw_pose=self.jaw_pose,
                body_pose=body_pose,
                expression=self.expression,
                return_verts=True
            )

            v_cano = output.v_posed[0]
            # re-mesh
            v_cano_dense = subdivide_inorder(v_cano, self.smplx_face[self.SMPLXSeg.remesh_mask], self.unique).squeeze(0)
            # add offsets
            vn = compute_normal(v_cano_dense, self.dense_faces)[0]
            v_cano_dense += self.v_offsets * vn
            # do LBS
            v_posed_dense = warp_points(v_cano_dense, self.dense_lbs_weights, output.joints_transform[:, :55])
            # translate
            v_posed_dense += t - translate[0]

            scan_v_posed.append(v_posed_dense)
        
        scan_v_posed = torch.cat(scan_v_posed).detach().cpu().numpy()
        new_scan_v_posed, new_face = build_new_mesh(scan_v_posed, self.dense_faces, self.vt, self.ft)
        new_scan_v_posed = new_scan_v_posed.astype(np.float32)

        return new_scan_v_posed, new_face

