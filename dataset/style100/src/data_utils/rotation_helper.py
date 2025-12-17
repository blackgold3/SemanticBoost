import numpy as np

def mat2aa(rot_mat_batch):
    r11 = rot_mat_batch[:, 0, 0]
    r22 = rot_mat_batch[:, 1, 1]
    r33 = rot_mat_batch[:, 2, 2]

    r32 = rot_mat_batch[:, 2, 1]
    r23 = rot_mat_batch[:, 1, 2]
    r13 = rot_mat_batch[:, 0, 2]
    r31 = rot_mat_batch[:, 2, 0]
    r21 = rot_mat_batch[:, 1, 0]
    r12 = rot_mat_batch[:, 0, 1]

    theta = np.expand_dims(np.arccos((r11 + r22 + r33 - 1) / 2), axis=1)

    axis_x = np.expand_dims(r32 - r23, axis=1)
    axis_y = np.expand_dims(r13 - r31, axis=1)
    axis_z = np.expand_dims(r21 - r12, axis=1)
    axis = np.concatenate((axis_x, axis_y, axis_z), axis=1) / np.sin(theta) / 2

    return theta * axis
