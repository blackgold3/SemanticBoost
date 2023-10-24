import torch
from motion.visual_api import Visualize  
import moviepy.editor as mpy
import os, sys
import time
import json
import imageio
import argparse

def interface(prompt, mode="cadm", render_mode="pyrender", out_size=1024, tada_role=None):
    os.makedirs("results/motion", exist_ok=True)
    os.makedirs("results/joints", exist_ok=True)
    os.makedirs("results/smpls", exist_ok=True)

    name = prompt.replace("/", "_").replace(" ", "_").replace(",", "_").replace("#", "_").replace("|", "_").replace(".npy", "").replace(".txt", "").replace(".csv", "").replace(".", "").replace("'", "_")
    name = "_".join(name.split("_")[:25])
    out_path = os.path.join("results/motion", name + ".mp4")
    gif_path = os.path.join("results/motion", name + ".gif")
    joint_path = os.path.join("results/joints", name + ".npy")
    smpl_path = os.path.join("results/smpls", name + ".npy")

    '''
    prompt 输入为 length, prompt, 如果只输入 prompt, length 默认为 196
    mode 指不同的模型
    '''

    assert mode in ["cadm", "cadm-augment", "mdm"]
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
        "tada_role":tada_role
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
    
    vid = mpy.ImageSequenceClip([x[:, :, :] for x in pics], fps=20)
    vid.write_videofile(out_path, remove_temp=True)
    imageio.mimsave(gif_path, pics, duration= 1000 / 20, loop=0)

    t4 = time.time()

    cost_init = t2 - t1 
    cost_infer = t3 - t2
    cost_render = t4 - t3

    print("initial model cost time: %.4f, infer and fit cost time: %.4f, render cost time: %.4f, total cost time: %.4f"%(cost_init, cost_infer, cost_render, t4 - t1))

    return out_path

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='visualize demo')
    ############################ basic_setings ########################
    parser.add_argument('--prompt', type=str, default="120, A person walks while raising his right hand up. During the process, the person moves to the south, the person looks forward downward, his left forearm moves to body's left front up, left back up repeatly.")
    parser.add_argument('--mode', type=str, default="cadm-augment", choices=['cadm', 'cadm-augment', "mdm"], help="choose model")
    parser.add_argument("--render_mode", default="pyrender_slow", type=str, choices=["pyrender_slow", "pyrender_fast", "joints"])
    parser.add_argument("--size", default=1024, type=int)
    parser.add_argument("--tada_role", default=None, type=str)
    opt = parser.parse_args()


    out_path = interface(opt.prompt, mode=opt.mode, render_mode=opt.render_mode, out_size=opt.size, tada_role=opt.tada_role)