import sys
import os
sys.path.append(os.path.dirname("./"))
from src.modules.fbx2smpl_helper import auto_fbx2smpl
import json

if __name__ == '__main__':
    with open("temp.json", "r") as f:
        json_dict = json.load(f)
    
    source = json_dict["source_dir"]
    target = json_dict["pose"]
    temp = json_dict["bvh_dir"]
    auto_fbx2smpl(src_dir=source, work_dir=temp, dst_root=target)