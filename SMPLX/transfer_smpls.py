import argparse
import numpy as np
import pickle
from SMPLX.transfer_model.write_obj import write_obj
from SMPLX.transfer_model.utils import read_deformation_transfer
from SMPLX.transfer_model.data import build_dataloader
from SMPLX.transfer_model.transfer_model import run_fitting
from SMPLX.transfer_model.merge_output import merge
from SMPLX.smplx import build_layer
import os
import torch
from tqdm import tqdm
import subprocess
import platform
import time

def load_npz(path):
    return np.load(path)

def load_pickle(path):
    with open(path, "rb") as f:
        res = pickle.load(f, encoding="latin1")
    return res

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='transfer between smpls')
    parser.add_argument('--source', default="smpl")
    parser.add_argument("--target", default="smplh")
    parser.add_argument("--model_path", default="/data/TTA/data/body_models")
    parser.add_argument("--extra_dir", default="/data/TTA/data/extra_dir", help="https://smpl-x.is.tue.mpg.de/download.php")
    parser.add_argument("--source_path", default="/data/TTA/data/humanact_smpl")
    parser.add_argument("--target_path", default="/data/TTA/data/humanact_smplh")
    parser.add_argument("--batch_size", default=500, type=int)
    args = parser.parse_args()
    device = "cuda"

    if args.target == "smplx" or args.source == "smplx":
        deformation_transfer_path = os.path.join(args.extra_dir, "{}2{}_deftrafo_setup.pkl".format(args.source, args.target))
    else:
        deformation_transfer_path = os.path.join(args.extra_dir, "{}2{}_def_transfer.pkl".format(args.source, args.target))

    if args.target == "smplx":
        model_params = {"betas":{"num":10}, "expression":{"num": 10}}

        mask_ids_fname = os.path.join(args.extra_dir, "smplx_mask_ids.npy")
        if os.path.exists(mask_ids_fname):
            mask_ids = np.load(mask_ids_fname)
            mask_ids = torch.from_numpy(mask_ids).to(device=device)
        else:
            print(f'Mask ids fname not found: {mask_ids_fname}')    

    elif args.target == "smplh" or args.target == "smpl":
        model_params = {"betas":{"num":10}}

        mask_ids_fname = ""   
        mask_ids = None     

    body_model_conf = {
        "ext":"npz",
        "model_type": args.target,
        "folder": args.model_path,
        "use_compressed": False,
        args.target:model_params
    }

    if args.target == "smplx" or args.target == "smpl":
        body_model_conf["use_face_contour"] = True


    for root, dirs, files in os.walk(args.source_path):
        for name in files:
            curr_file = os.path.join(root, name)

            new_root = os.path.join(args.target_path , "/".join(root.split("/")[:-2:-1]))
            os.makedirs(new_root, exist_ok=True)
            curr_target = os.path.join(new_root, name.replace(".npz", ".npy"))
            if os.path.exists(curr_target):
                print("%s has been competed"%(curr_target))
                continue

            if name.split(".")[-1] == "npz":
                curr = load_npz(curr_file)
                body_pose = None
            elif name.split(".")[-1] == "pkl":
                curr = load_pickle(curr_file)
                body_pose = None
            elif name.split(".")[-1] == "npy":
                curr = np.load(curr_file)
                body_pose = curr
            else:
                continue  
            
            if body_pose is None:
                try:
                    body_pose = curr["poses"]
                except:
                    print("Not Pose Data")
                    continue

                gender = str(curr["gender"])
                body_model_conf["gender"] = gender
            else:
                gender = "neutral"
                body_model_conf["gender"] = gender

        
            cid = name.split(".")[0]
            save_folder1 = os.path.join("temp", "objs")
            save_folder2 = os.path.join(new_root, str(time.time()))
            os.makedirs(save_folder1, exist_ok=True)
            os.makedirs(save_folder2, exist_ok=True)
        
            write_obj(args.model_path, curr_file, save_folder1, args.source, gender, 10, 10, True, device)
        
            body_model = build_layer(args.model_path, **body_model_conf)
            body_model = body_model.to(device=device)

            def_matrix = read_deformation_transfer(deformation_transfer_path, device=device)

            datasets = {
                "mesh_folder":{"data_folder":save_folder1},
                "batch_size":args.batch_size
            }

            data_obj_dict = build_dataloader(datasets)

            dataloader = data_obj_dict['dataloader']

            for ii, batch in enumerate(tqdm(dataloader)):
                for key in batch:
                    if torch.is_tensor(batch[key]):
                        batch[key] = batch[key].to(device=device)
                
                var_dict = run_fitting(batch, body_model, def_matrix, mask_ids)
                paths = batch['paths']

                for ii, path in enumerate(paths):
                    _, fname = os.path.split(path)

                    output_path = os.path.join(
                        save_folder2, f'{os.path.splitext(fname)[0]}.pkl')
                    
                    save_dict = {}
                    for key in var_dict.keys():
                        try:
                            save_dict[key] = var_dict[key][ii:ii+1]
                        except:
                            save_dict[key] = var_dict[key][ii:ii+1]
                    
                    with open(output_path, "wb") as f:
                        pickle.dump(save_dict, f)

            results = merge(save_folder2, gender)
            np.save(curr_target, results)

            cmd = "rm -r {}".format(save_folder1)
            subprocess.call(cmd, shell=platform.system() != 'Windows')
            cmd = "rm -r {}".format(save_folder2)
            subprocess.call(cmd, shell=platform.system() != 'Windows')

            del body_model
            del dataloader
            del data_obj_dict 
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
