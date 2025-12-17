import numpy as np

map_hm36_to_smpl = {
    0:0,
    1:"6 7 0.78",
    2:"1 2 0.78",
    3:"0 12 0.58",
    4:7,
    5:2,
    6:"0 12 0.13",
    7:8,
    8:3,
    9:12,
    10:10,
    11:5,
    12:13,
    15:14,
    16:17,
    17:25,
    18:18,
    19:26,
    20:19,
    21:27,
}

map_part_smpl_to_smpl = {
    13:"9 12 16 0.5 0.5",
    14:"9 12 17 0.5 0.5",
}

def full_smpl(joints, njoints=22):
    new_motions = joints.copy()
    for i in range(njoints):
        if i not in map_part_smpl_to_smpl:
            continue
        target_key = map_part_smpl_to_smpl[i]
        splits = target_key.split()
        if len(splits) == 3:
            f, s, r = splits
            f = int(f)
            s = int(s)
            r = float(r)
            target = joints[:, f, :] * r + joints[:, s, :] * (1 - r)
        elif len(splits) == 5:
            f, s, t, r1, r2 = splits
            f = int(f)
            s = int(s)
            t = int(t)
            r1 = float(r1)
            r2 = float(r2)
            target = (joints[:, f, :] * r1 + joints[:, s, :] * (1 - r1)) * r2 + joints[:, t, :] * (1 - r2)
        elif len(splits) == 7:
            f, s, t, fo, r1, r2, r3 = splits
            f = int(f)
            s = int(s)
            t = int(t)
            fo = int(fo)
            r1 = float(r1)
            r2 = float(r2)
            r3 = float(r3)
            target = (joints[:, f, :] * r1 + joints[:, s, :] * (1 - r1)) * r3 + (joints[:, t, :] * r2 + joints[:, fo, :] * (1 - r2)) * (1 - r3)  
        new_motions[:, i, :] = target
        new_motions[:, 15, :2] = new_motions[:, 15, :2] * 0.5  + new_motions[:, 12, :2] * 0.5
        new_motions[:, 15, 2] = new_motions[:, 15, 2] * 0.4  + new_motions[:, 12, 2] * 0.6
         
    return new_motions

def convert_hm36_to_smpl(hm36, njoints=22):
    '''
    [nframes, 32, 3]
    '''

    nframes = hm36.shape[0]
    new_motions = np.zeros([nframes, njoints, 3])
    for i in range(njoints):
        if i not in map_hm36_to_smpl:
            continue
        target_key = map_hm36_to_smpl[i]
        if isinstance(target_key, str):
            first, second, ratio = target_key.split()
            first = int(first)
            second = int(second)
            ratio = float(ratio)
            target = hm36[:, first, :] * ratio + hm36[:, second, :] * (1 - ratio)
        else:
            target = hm36[:, target_key, :]     #### [nframes, 3]

        new_motions[:, i, :] = target

    new_motions = full_smpl(new_motions, njoints)
    return new_motions