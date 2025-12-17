import numpy as np
from os.path import join as pjoin
from tqdm import tqdm


def mean_variance(train_file, save_dir, joints_num):
    with open(train_file, "r") as f:
        lines = f.readlines()
        lines = [line.strip() for line in lines]

    base_motion = 0
    base_std = 0
    count = 0
    for line in lines:
        line = line.split("#")[0]
        curr_matrix = np.load(line, allow_pickle=True)
        keys = curr_matrix.files
        for j in tqdm(range(len(keys))):
            key = keys[j]
            motion = curr_matrix[key]
            if np.isnan(motion).any():
                print("nan value in ==>", key)
                continue

            count += motion.shape[0]
            base_motion += motion.sum(axis=0)

    Mean = base_motion / count
    count = 0

    for line in lines:
        line = line.split("#")[0]
        curr_matrix = np.load(line, allow_pickle=True)
        keys = curr_matrix.files
        for j in tqdm(range(len(keys))):
            key = keys[j]
            motion = curr_matrix[key]
            if np.isnan(motion).any():
                print("nan value in ==>", key)
                continue

            count += motion.shape[0]
            base_std += ((motion - Mean) ** 2).sum(axis=0)

    print("total frames is:%d"%(count))
    Std = np.sqrt(base_std / count)

    Std[0:1] = Std[0:1].mean() / 1.0
    Std[1:3] = Std[1:3].mean() / 1.0
    Std[3:4] = Std[3:4].mean() / 1.0
    Std[4: 4+(joints_num - 1) * 3] = Std[4: 4+(joints_num - 1) * 3].mean() / 1.0
    Std[4+(joints_num - 1) * 3: 10+(joints_num - 1) * 9] = Std[4+(joints_num - 1) * 3: 10+(joints_num - 1) * 9].mean() / 1.0
    Std[10+(joints_num - 1) * 9: 10+(joints_num - 1) * 9 + joints_num*3] = Std[10+(joints_num - 1) * 9: 10+(joints_num - 1) * 9 + joints_num*3].mean() / 1.0
    Std[1+ joints_num * 12:5 + joints_num * 12] = Std[1+ joints_num * 12:5 + joints_num * 12].mean() / 1.0

    assert 5 + joints_num * 12 == Std.shape[-1]

    np.save(pjoin(save_dir, 'Mean-smr.npy'), Mean)
    np.save(pjoin(save_dir, 'Std-smr.npy'), Std)     

    return Mean, Std
