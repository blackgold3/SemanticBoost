import torch
from motion.visual_api import Visualize  
import cv2
import os, sys
import time
import json
import imageio
import argparse
from tqdm import tqdm

def interface(prompt, mode="camd", render_mode="pyrender", out_size=1024, tada_role=None, speedup=1, export2fbx=0, blender_path=None):
    os.makedirs("results/motion", exist_ok=True)
    os.makedirs("results/joints", exist_ok=True)
    os.makedirs("results/smpls", exist_ok=True)
    os.makedirs("results/fbxs", exist_ok=True)

    name = prompt.replace("/", "_").replace(" ", "_").replace(",", "_").replace("#", "_").replace("|", "_").replace(".npy", "").replace(".txt", "").replace(".csv", "").replace(".", "").replace("'", "_")
    name = "_".join(name.split("_")[:25])
    out_path = os.path.join("results/motion", name + ".mp4")
    joint_path = os.path.join("results/joints", name + ".npy")
    smpl_path = os.path.join("results/smpls", name + ".npy")
    fbx_path = os.path.join("results/fbxs", name + ".fbx")

    '''
    prompt 输入为 length, prompt, 如果只输入 prompt, length 默认为 196
    mode 指不同的模型
    '''

    assert render_mode in ["joints", "pyrender_fast", "pyrender_slow"]
    path = None

    with open("motion/path.json", "r") as f:
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
        "speedup":speedup
    }
    visual = Visualize(**kargs)

    t2 = time.time()

    output = visual.predict(prompt, path, render_mode, joint_path, smpl_path)

    t3 = time.time()

    if render_mode == "joints":
        pics = visual.joints_process(output, prompt, out_size, out_size)
    elif render_mode.startswith("pyrender"):
        meshes, _ = visual.get_mesh(output)
        pics = visual.pyrender_process(meshes, out_size, out_size)
    
    video=cv2.VideoWriter(out_path,cv2.VideoWriter_fourcc(*'MP4V'),20,(out_size, out_size))
    for pic in tqdm(pics):
        pic = pic[:, :, :3]
        pic = cv2.cvtColor(pic, cv2.COLOR_BGR2RGB)
        video.write(pic)   #写入视频
    video.release()

    # imageio.mimsave(gif_path, pics, duration= 1000 / 20, loop=0)

    t4 = time.time()

    cost_init = t2 - t1 
    cost_infer = t3 - t2
    cost_render = t4 - t3

    if export2fbx != 0:
        import platform
        import subprocess

        cmd = "{} --background --python motion/fbx_output.py --input {}  --output {} --smpl2fbx {}".format(blender_path, smpl_path, fbx_path, json_dict["smpl2fbx_m"])
        subprocess.call(cmd, shell=platform.system() != 'Windows')


    print("initial model cost time: %.4f, infer and fit cost time: %.4f, render cost time: %.4f, total cost time: %.4f"%(cost_init, cost_infer, cost_render, t4 - t1))

    return out_path

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='visualize demo')
    ############################ basic_setings ########################
    parser.add_argument('--prompt', type=str, default="A person stands up from a chair, then walks counter clockwise along a circle, finally sits back on the chair.")
    parser.add_argument('--mode', type=str, default="ncamd", choices=['camd', 'camd-augment', "mdm", "ncamd", "ncamd-augment"], help="choose model")
    parser.add_argument("--render_mode", default="pyrender_slow", type=str, choices=["pyrender_slow", "pyrender_fast", "joints"])
    parser.add_argument("--size", default=1024, type=int)
    parser.add_argument("--tada_role", default=None, type=str)
    parser.add_argument("--speedup", default=1, type=int, help="if load tensorRT model.")
    parser.add_argument("--export2fbx", default=0, type=int, help="export2fbx == 0 do not export fbx, else export fbx file.")
    parser.add_argument("--blender_path", default="/data/TTA/blender/blender", type=str, help="export fbx mush through blender api, we test with 2.93")
    
    opt = parser.parse_args()

    out_path = interface(opt.prompt, mode=opt.mode, render_mode=opt.render_mode, out_size=opt.size, tada_role=opt.tada_role, speedup=opt.speedup, export2fbx=opt.export2fbx, blender_path=opt.blender_path)