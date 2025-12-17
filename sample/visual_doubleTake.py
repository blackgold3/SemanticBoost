import os
import torch
import numpy as np
import platform
import subprocess
import argparse
import imageio.v2 as imageio
from sample.t2m_model_infer import ModelInfer
from sample.render_process import pyrender_process
from sample.t2m_post_process import pose2mesh, motion_rep_process

def mask_png(frames):
    for frame in frames:
        im = imageio.imread(frame)
        im[im[:, :, 3] < 1, :] = 255
        imageio.imwrite(frame, im[:, :, 0:3])
    return

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='visualize demo')
    ############################ basic_setings ########################
    parser.add_argument('--prompts', type=str, required=True, nargs="+", help="text2motion 条件")
    parser.add_argument('--nframes', type=int, required=True, nargs="+", help="生成长度")
    parser.add_argument('--checkpoints', type=str, required=True)
    parser.add_argument('--handshake_size', type=int, default=20, help="doubleTake衔接区域长度")
    parser.add_argument('--blend_size', type=int, default=10, help="doubleTake过渡区域长度")
    parser.add_argument('--fps', type=int, default=20, help="渲染帧率")
    parser.add_argument("--target_dir", type=str, default="results/files", help="output dir name")
    parser.add_argument("--target_name", type=str, default=None, help="output file name")
    parser.add_argument("--model_path", default="/apdcephfs_cq11/share_1567347/share_info/kleinhe/checkpoints/body_models", type=str)
    parser.add_argument("--render_mode", default="pyrender", type=str, choices=["blender", "pyrender"])
    parser.add_argument("--blender_path", default="/apdcephfs_cq11/share_1567347/share_info/kleinhe/blender-4.0.2-linux-x64", type=str)
    parser.add_argument("--out_mode", default="video", type=str, choices=["sequence", "video"])
    parser.add_argument("--device", default="cuda", type=str)
    parser.add_argument("--save", action="store_true", help="是否导出生成的表征文件")
    parser.add_argument("--export", action="store_true", help="是否导出fbx 文件")
    parser.add_argument("--base_fbx", default="/apdcephfs_cq11/share_1567347/share_info/kleinhe/T2M/SMPL_m_unityDoubleBlends_lbs_10_scale5_207_v1.0.0.fbx", help="导出 fbx的mesh文件")
    opt = parser.parse_args()

    ############# 初始化 ##############
    model = ModelInfer(opt.checkpoints, opt.device)
    os.makedirs(opt.target_dir, exist_ok=True)
    if opt.target_name is None:
        opt.target_name = "doubkeTake"

    ############# 条件构造和模型推理 ##############
    motions = model.doubleTake(opt.prompts, opt.nframes, opt.handshake_size, opt.blend_size)
    smpl_pose = motion_rep_process(motions, model.rep, opt.model_path, opt.device)
    ################### 渲染 #####################
    vertices, joints, faces = pose2mesh(smpl_pose, opt.model_path, opt.device)
    if opt.render_mode == "pyrender":
        if opt.out_mode == "sequence":
            raise NotImplementedError("Pyrender not support sequence picture")
        else:
            pics = pyrender_process(vertices, height=1080, weight=1920, face_path=faces)
            target_path = os.path.join(opt.target_dir, opt.target_name + ".mp4")
            imageio.mimsave(target_path, pics, fps=opt.fps)   
    elif opt.render_mode == "blender":
        mesh_path = os.path.join(opt.target_dir, opt.target_name + ".npy")
        np.save(mesh_path, vertices)
        if opt.out_mode == "sequence":
            pic_path = mesh_path.replace(".npy", ".png")
            if os.path.exists(pic_path):
                os.remove(pic_path)
    
        cmd = "{}/blender --background --python sample/blender/render.py -- --cfg=sample/blender/configs/render.yaml --npy={} --mode={} --joint_type=humanml3d --fps={}".format(opt.blender_path, mesh_path, opt.out_mode, opt.fps)
        print(cmd)
        subprocess.call(cmd, shell=platform.system() != 'Windows')
        if opt.out_mode == "sequence":
            frames = [mesh_path.replace(".npy", ".png")]
            mask_png(frames)
        os.remove(mesh_path)

    if opt.save:
        pose_path = os.path.join(opt.target_dir, opt.target_name + ".npy")
        print("=========== 保存生成文件 ", motions.shape, "===============")
        np.save(pose_path, motions.detach().cpu().numpy())

    if opt.export:
        pose_path = os.path.join(opt.target_dir, opt.target_name + "_pose.npy")
        fbx_path = os.path.join(opt.target_dir, opt.target_name + ".fbx")
        np.save(pose_path, smpl_pose)
        cmd = "{}/blender --background --python sample/fbx_output.py --input {}  --output {} --smpl2fbx {}".format(opt.blender_path, pose_path, fbx_path, opt.base_fbx)
        subprocess.call(cmd, shell=platform.system() != 'Windows')
        os.remove(pose_path)