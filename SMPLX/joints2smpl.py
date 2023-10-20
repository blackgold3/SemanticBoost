import torch
from SMPLX.visualize_joint2smpl.simplify_loc2rot import joints2smpl
import argparse
import numpy as np
import os
from tqdm import tqdm

parser = argparse.ArgumentParser(description='transfer joint3d to smpls')
parser.add_argument("--model_path", default="/data/TTA/data/body_models")
parser.add_argument('--source_path', default="/data/TTA/data/humanact12/group_000")
parser.add_argument("--target_path", default="/data/TTA/data/humanact_smplh/group_000")
parser.add_argument("--mode", default="joints", choices=["t2m", "joints"])
args = parser.parse_args()
device = "cuda"

if os.path.isdir(args.source_path):
    os.makedirs(args.target_path, exist_ok=True)
    files = os.listdir(args.source_path)
    target_files = files
else:
    files = [args.source_path]
    args.source_path = ""

    if args.target_path.split(".")[-1] != "npy":
        os.makedirs(args.target_path)
        target_files = [files[0].split("/")[-1]]
    else:
        target_files = [args.target_path]
        args.target_path = ""

for i in range(len(files)):
    curr_path = os.path.join(args.source_path, files[i])
    target_path = os.path.join(args.target_path, target_files[i])
    if os.path.exists(target_path):
        continue

    curr_file = np.load(curr_path)       #### [nframe, 263]
    curr_file = torch.from_numpy(curr_file)

    if args.mode == "t2m":
        from dataset.t2m.recover_joints import recover_from_ric
        motions = recover_from_ric(curr_file, 22)    #### [nframes, 22, 3]
        motions = motions.detach().cpu().numpy()
    else:
        motions = curr_file.detach().cpu().numpy()
 
    frames, njoints, nfeats = motions.shape
    MINS = motions.min(axis=0).min(axis=0)
    MAXS = motions.max(axis=0).max(axis=0)
    height_offset = MINS[1]
    motions[:, :, 1] -= height_offset
    model = joints2smpl(frames, 0, True, model_path=args.model_path)
    target, trans = model.joint2smpl(motions)

    target = np.concatenate([target, trans], axis=1)

    np.save(target_path, target)
    if i % 10 == 0:
        print("save %d npys"%(i))