import os
import numpy as np
from tqdm import tqdm
def split_files(root_path, final_path, text_path, smr_rep=False):
    files = os.listdir(final_path)
    dataset = root_path.split("/")[-1]
    files = sorted(files)
    trainval_dict = {}
    trainval_text = {}

    for i in tqdm(range(len(files))):
        curr_name = files[i].split(".")[0]
        curr_path = os.path.join(final_path, files[i])
        curr_text_path = os.path.join(text_path, files[i].replace("npy", "txt"))

        save_key = "{}_{}".format(dataset, curr_name)

        motion = np.load(curr_path)

        if motion.shape[0] < 40:
            continue

        with open(curr_text_path, "r") as f:
            texts = f.readlines()
            texts = [texts[i].strip() for i in range(len(texts))]

        trainval_dict[save_key] = motion
        trainval_text[save_key] = texts

    if smr_rep:
        extension = "-smr"
    else:
        extension = ""

    np.savez(os.path.join(root_path, "train-motion{}.npz".format(extension)), **trainval_dict)
    np.savez(os.path.join(root_path, "train-text{}.npz".format(extension)), **trainval_text)
