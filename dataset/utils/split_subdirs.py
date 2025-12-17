import os
import shutil
import numpy as np
from tqdm import tqdm

def split_files_to_subdirs(input_dir):
    files = os.listdir(input_dir)
    files = sorted(files)
    number = len(files)
    dirs = int(np.ceil(number / 1000))
    for i in range(dirs):
        os.makedirs(os.path.join(input_dir, "%02d"%(i)), exist_ok=True)
    
    for i in tqdm(range(len(files))):
        curr_dir = i // 1000
        curr_path = os.path.join(input_dir, files[i])
        target_path = os.path.join(input_dir, "%02d"%curr_dir, files[i])
        shutil.move(curr_path, target_path)

    return len(files) // 1000 + 1