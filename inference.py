import torch
from mdm.visual_api import Visualize  
import cv2
import os, sys
import time
import json
import imageio
import argparse
from tqdm import tqdm
import platform
import subprocess
import moviepy.editor as mpy

def interface(prompt, mode="camd", render_mode="pyrender", out_size=1024, tada_role=None, length="120", follow=True, export=False):
    os.makedirs("results/motion", exist_ok=True)
    os.makedirs("results/joints", exist_ok=True)
    os.makedirs("results/smpls", exist_ok=True)
    os.makedirs("results/fbxs", exist_ok=True)

    if mode == "mdm" or "|" in prompt:
        speedup = 0
    else:
        speedup = 1

    name = prompt.replace("/", "_").replace(" ", "_").replace(",", "_").replace("#", "_").replace("|", "_").replace(".npy", "").replace(".txt", "").replace(".csv", "").replace(".", "").replace("'", "_")
    name = "_".join(name.split("_")[:25])
    out_path = os.path.join("results/motion", name + ".mp4")
    gif_path = os.path.join("results/motion", name + ".gif")
    joint_path = os.path.join("results/joints", name + ".npy")
    smpl_path = os.path.join("results/smpls", name + ".npy")
    fbx_path = os.path.join("results/fbxs", name + ".fbx")

    '''
    prompt 输入为 length, prompt, 如果只输入 prompt, length 默认为 196
    mode 指不同的模型
    '''

    render_mode = {
        "3dfast":"pyrender_fast",
        "3dslow":"pyrender_slow",
        "joints":"joints"
    }[render_mode]
    assert render_mode in ["joints", "pyrender_fast", "pyrender_slow"]
    path = None

    with open("mdm/path.json", "r") as f:
        json_dict = json.load(f)
   
    t1 = time.time()

    kargs = {
        "mode":mode,
        "device":"cuda" if torch.cuda.is_available() else "cpu",
        "rotate":0,
        "condition":"text",
        "smpl_path":json_dict["smpl_path"],
        "skip_steps":0,
        "path":json_dict,
        "tada_base":json_dict["tada_base"],
        "tada_role":tada_role,
        "speedup":speedup,
        "length":length
    }
    visual = Visualize(**kargs)

    t2 = time.time()

    output = visual.predict(prompt, path, render_mode, joint_path, smpl_path)

    t3 = time.time()

    if render_mode == "joints":
        pics = visual.joints_process(output, prompt, out_size, out_size)
    elif render_mode.startswith("pyrender"):
        meshes, _ = visual.get_mesh(output)
        pics = visual.pyrender_process(meshes, out_size, out_size, follow=follow)
    
    vid = mpy.ImageSequenceClip([x[:, :, :] for x in pics], fps=20)
    vid.write_videofile(out_path, remove_temp=True)

    # imageio.mimsave(gif_path, pics, duration= 1000 / 20, loop=0)

    t4 = time.time()

    cost_init = t2 - t1 
    cost_infer = t3 - t2
    cost_render = t4 - t3

    if os.path.exists(json_dict["blender_path"]) and export:
        cmd = "{}/blender --background --python mdm/fbx_output.py --input {}  --output {} --smpl2fbx {}".format(json_dict["blender_path"], smpl_path, fbx_path, json_dict["smpl2fbx_m"])
        subprocess.call(cmd, shell=platform.system() != 'Windows')
        t5 = time.time()
        cost_export = t5 - t4
    else:
        t5 = t4
        cost_export = 0

    cost_init = t2 - t1 
    cost_infer = t3 - t2
    cost_render = t4 - t3
       
    print("initial model cost time: %.4f, infer and fit cost time: %.4f, render cost time: %.4f, export fbx cost time: %.4f, total cost time: %.4f"%(cost_init, cost_infer, cost_render, cost_export, t5 - t1))

    return out_path, joint_path, smpl_path, fbx_path

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='visualize demo')
    ############################ basic_setings ########################
    parser.add_argument('--prompt', type=str, default="A person walks counter clockwise along a circle.")
    parser.add_argument('--mode', type=str, default="ncamd", choices=['camd', 'camd-augment', "mdm", "ncamd", "ncamd-augment"], help="choose model")
    parser.add_argument("--render", default="3dslow", type=str, choices=["3dslow", "3dfast", "joints"])
    parser.add_argument("--size", default=1024, type=int)
    parser.add_argument("--role", default=None, type=str)
    parser.add_argument("--length", default="180", type=str)
    parser.add_argument("-f", "--follow", action="store_true", help="if camera follow motion during render")
    parser.add_argument("-e", "--export", action="store_true", help="if export fbx file")

    opt = parser.parse_args()

    out_path = interface(opt.prompt, mode=opt.mode, render_mode=opt.render, out_size=opt.size, tada_role=opt.role, length=opt.length, follow=opt.follow, export=opt.export)