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
    parser.add_argument('--checkpoints', type=str, required=True)
    parser.add_argument('--prompt', type=str, default=None, help="text2motion 条件")
    parser.add_argument('--mode', type=str, default="nocond", choices=["nocond", "text"], help="模型是text2motion模型时, 可以在这里设置CFG参数强行无条件, nocond 模型不受影响")
    parser.add_argument('--nframes', type=int, default=200, help="生成长度")
    parser.add_argument('--help_file', type=str, default=None, help="inpainting的参考文件")
    parser.add_argument('--capture_begin', type=int, default=[0], nargs="+", help="参考文件的参考起点")   
    parser.add_argument('--capture_end', type=int, default=[0], nargs="+", help="参考文件的参考终点")   
    parser.add_argument('--target_begin', type=int, default=[0], nargs="+", help="生成文件的控制起点")   
    parser.add_argument('--target_end', type=int, default=[0], nargs="+", help="生成文件的控制终点")   
    parser.add_argument('--blend_size', type=int, default=10, help="混合区域长度")   
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
    parser.add_argument("--novis", action="store_true", help="不渲染，只保存模型结果，用于下游任务等")
    parser.add_argument("--base_fbx", default="/apdcephfs_cq11/share_1567347/share_info/kleinhe/T2M/SMPL_m_unityDoubleBlends_lbs_10_scale5_207_v1.0.0.fbx", help="导出 fbx的mesh文件")
    opt = parser.parse_args()

    ############# 初始化 ##############
    model = ModelInfer(opt.checkpoints, opt.device)
    os.makedirs(opt.target_dir, exist_ok=True)
    if opt.target_name is None:
        opt.target_name = "inpainting"

    ############# 条件构造和模型推理 ##############
    if opt.prompt is None:
        opt.prompt = ""
    model_kwargs = {'y':{'text': opt.prompt, 'lengths':opt.nframes}}
    if opt.mode == "nocond":
        model_kwargs['y']['scale'] = torch.zeros(1, device=opt.device)
    else:
        model_kwargs['y']['scale'] = torch.ones(1, device=opt.device) * model.args.guidance_param

    assert len(opt.target_begin) == len(opt.target_end) == len(opt.capture_begin) == len(opt.capture_end)
    final_inpainting_mask = 0
    final_inpainted_motion = 0
    for i in range(len(opt.capture_begin)):
        inpainted_motion, inpainting_mask = model.inpainting_mask(opt.help_file, opt.nframes, opt.capture_begin[i], opt.capture_end[i], opt.target_begin[i], opt.target_end[i], opt.blend_size)
        final_inpainting_mask += inpainting_mask
        final_inpainted_motion += inpainted_motion

    model_kwargs['y']['inpainted_motion'] = final_inpainted_motion
    model_kwargs['y']['inpainting_mask'] = final_inpainting_mask
    motions = model(opt.nframes, model_kwargs)

    ################### 渲染 #####################
    if not opt.novis:
        smpl_pose = motion_rep_process(motions, model.rep, opt.model_path, opt.device)
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

    if opt.save:
        pose_path = os.path.join(opt.target_dir, opt.target_name + ".npy")
        print("=========== 保存生成文件 ", motions.shape, "===============")
        np.save(pose_path, motions.detach().cpu().numpy())

    if opt.export:
        pose_path = os.path.join(opt.target_dir, opt.target_name + "_pose.npy")
        fbx_path = os.path.join(opt.target_dir, opt.target_name + ".fbx")

        try:        ###### 如果没开可视化，直接导出 fbx 会报没有 smpl_pose 的错误
            np.save(pose_path, smpl_pose)
        except:
            smpl_pose = motion_rep_process(motions, model.rep, opt.model_path, opt.device)
            np.save(pose_path, smpl_pose)

        cmd = "{}/blender --background --python sample/fbx_output.py --input {}  --output {} --smpl2fbx {}".format(opt.blender_path, pose_path, fbx_path, opt.base_fbx)
        subprocess.call(cmd, shell=platform.system() != 'Windows')
        os.remove(pose_path)