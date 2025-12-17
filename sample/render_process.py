import numpy as np
from trimesh import Trimesh
import pyrender
from pyrender.constants import RenderFlags
import os
os.environ['PYOPENGL_PLATFORM'] = "egl"
from tqdm import tqdm
from pyrr import matrix44, Vector3
from dataset.utils.rotation_conversions import *
import matplotlib.pyplot as plt

def pyrender_process(vertices, height=1080, weight=1920, face_path="dataset/smplh.faces"):
    '''
    vertices [nframes, 6890, 3]
    '''
    if isinstance(face_path, np.ndarray):
        faces = face_path
    elif os.path.exists(face_path):
        faces = np.load(face_path)
    else:
        faces = None

    vertices = vertices.astype(np.float32)
    nframes = vertices.shape[0]

    fov_y = np.pi / 3
    '''
    相机参数
    '''
    MINS = np.min(np.min(vertices, axis=0), axis=0)
    MAXS = np.max(np.max(vertices, axis=0), axis=0)
    c = -np.pi / 12
    camera=np.array([
        [ 1, 0, 0, 0],
        [ 0, 1, 0, (MAXS[1] + MINS[1])/ 2],
        [ 0, 0, 1, MAXS[2] + 3.0],
        [ 0, 0, 0, 1]
    ])

    camera = camera[None, :, :].repeat(nframes, axis=0)
    
    pics = []
    ############### ground initial ###########
    r = pyrender.OffscreenRenderer(weight, height)

    colors = plt.cm.get_cmap("tab10")

    for i in tqdm(range(nframes)):
        bg_color = [1, 1, 1, 0.5]
        scene = pyrender.Scene(bg_color=bg_color, ambient_light=(0.4, 0.4, 0.4))
        mesh = Trimesh(vertices=vertices[i, :, :].tolist(), faces=faces)
        base_color = colors(1)
        material = pyrender.MetallicRoughnessMaterial(
            metallicFactor=0.5, roughnessFactor=0.7,
            alphaMode='OPAQUE',
            baseColorFactor=base_color
        ) 
        mesh = pyrender.Mesh.from_trimesh(mesh, material=material)   
        scene.add(mesh)

        ########################### ground ##################
        light = pyrender.DirectionalLight(color=[1,1,1], intensity=200)
        light_pose = np.eye(4)
        light_pose[:3, 3] = [0, -1, 1]
        scene.add(light, pose=light_pose.copy())
        light_pose[:3, 3] = [0, 1, 1]
        scene.add(light, pose=light_pose.copy())
        light_pose[:3, 3] = [1, 1, 2]
        scene.add(light, pose=light_pose.copy())

        ################ camera ##############
        curr_camera = pyrender.PerspectiveCamera(yfov=fov_y)
        scene.add(curr_camera, pose=camera[i])
        pic, _ = r.render(scene, flags=RenderFlags.RGBA)
        pics.append(pic)

    pics = np.stack(pics, axis=0)
    return pics