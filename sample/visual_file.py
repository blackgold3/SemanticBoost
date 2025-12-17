import os
import numpy as np
import platform
import subprocess
import argparse
import imageio.v2 as imageio
from sample.render_process import pyrender_process
from sample.t2m_post_process import pose2mesh, motion_rep_process

'''
[
    [c_x[0], c_y[0], c_z[0], c_e[0]],
    [c_x[1], c_y[1], c_z[1], c_e[1]],
    [c_x[2], c_y[2], c_z[2], c_e[2]],
    [0, 0, 0, 1]
]

'''

def mask_png(frames):
    for frame in frames:
        im = imageio.imread(frame)
        im[im[:, :, 3] < 1, :] = 255
        imageio.imwrite(frame, im[:, :, 0:3])
    return

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='visualize demo')
    ############################ basic_setings ########################
    parser.add_argument('--source', type=str, default="/apdcephfs/private_kleinhe/TTA/motionWE/results/12_zhao_2_1_1.npz", help="文件路径")
    parser.add_argument('--mode', type=str, default="pose", choices=['pose', 't2m', "joints", "smr", "JP"], help="已有文件的类型")
    parser.add_argument('--fps', type=int, default=30, help="渲染帧率")
    parser.add_argument("--target_dir", type=str, default="results/test", help="output dir name")
    parser.add_argument("--target_name", type=str, default=None, help="output file name")
    parser.add_argument("--model_path", default="/apdcephfs_cq11/share_1567347/share_info/kleinhe/checkpoints/body_models", type=str)
    parser.add_argument("--render_mode", default="pyrender", type=str, choices=["blender", "pyrender"])
    parser.add_argument("--blender_path", default="/apdcephfs_cq11/share_1567347/share_info/kleinhe/blender-4.0.2-linux-x64", type=str)
    parser.add_argument("--out_mode", default="video", type=str, choices=["sequence", "video"])
    parser.add_argument("--device", default="cuda", type=str)
    parser.add_argument("--export", action="store_true", help="是否导出fbx 文件")
    parser.add_argument("--base_fbx", default="/apdcephfs_cq11/share_1567347/share_info/kleinhe/T2M/SMPL_m_unityDoubleBlends_lbs_10_scale5_207_v1.0.0.fbx", help="导出 fbx的mesh文件")
    opt = parser.parse_args()

    if os.path.exists(opt.source):
        curr_file = np.load(opt.source)
    else:
        raise ValueError("No input file")

    os.makedirs(opt.target_dir, exist_ok=True)
    if opt.target_name is None:
        opt.target_name = opt.source.split("/")[-1].split(".")[0]

    # pose = curr_file["poses"]
    # trans = curr_file["trans"]
    # fps = curr_file["mocap_frame_rate"]
    # curr_file = np.concatenate([pose, trans], axis=-1)

    smpl_pose = motion_rep_process(curr_file, opt.mode, opt.model_path, opt.device)
    vertices, joints, faces = pose2mesh(smpl_pose, opt.model_path, opt.device)

    if opt.render_mode == "pyrender":
        if opt.out_mode == "sequence":
            raise NotImplementedError("Pyrender not support sequence picture")
        else:
            pics = pyrender_process(vertices, height=1080, weight=1920, face_path=faces)
            # target_path = os.path.join(opt.target_dir, opt.target_name + ".mp4")
            # imageio.mimsave(target_path, pics, fps=opt.fps)   
            target_path = os.path.join(opt.target_dir, opt.target_name + ".gif")
            imageio.mimsave(target_path, pics, duration= int(1000 / opt.fps), loop=0)    
            
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

    if opt.export:
        pose_path = os.path.join(opt.target_dir, opt.target_name + "_pose.npy")
        fbx_path = os.path.join(opt.target_dir, opt.target_name + ".fbx")
        np.save(pose_path, smpl_pose)
        cmd = "{}/blender --background --python sample/fbx_output.py --input {}  --output {} --smpl2fbx {}".format(opt.blender_path, pose_path, fbx_path, opt.base_fbx)
        subprocess.call(cmd, shell=platform.system() != 'Windows')
        os.remove(pose_path)