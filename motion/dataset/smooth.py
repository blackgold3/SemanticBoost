import numpy as np

def smooth_motion_joint(motion, beta=0.9):
    new_joints = [motion[0]]
    curr = motion[0]
    for i in range(1, motion.shape[0]):
        curr = curr * beta + motion[i] * (1 - beta)
        new_joints.append(curr.copy())
    new_joints = np.stack(new_joints, axis=0)
    return new_joints

def smooth_motion_translation(motion, beta=0.9):
    new_x = [motion[0]]
    curr = motion[0]
    for i in range(1, motion.shape[0]):
        curr[0] = curr[0] * beta + motion[i, 0] * (1 - beta)
        curr[1:] = motion[i, 1:]
        new_x.append(curr.copy())
    new_x = np.stack(new_x, axis=0)
    return new_x 

def comb(n, k):
    from math import factorial
    return factorial(n) // (factorial(k) * factorial(n-k))

def get_bezier_curve(points):
    n = len(points) - 1
    return lambda t: sum(comb(n, i)*t**i * (1-t)**(n-i)*points[i] for i in range(n+1))

def evaluate_bezier(points, total):
    bezier = get_bezier_curve(points)
    new_points = np.array([bezier(t) for t in np.linspace(0, 1, total)])
    return new_points

def bezier_smooth(motion, rank=50):
    nframes = motion.shape[0]
    if len(motion.shape) == 3:
        motion = motion.reshape(nframes, -1)
        new_motion = evaluate_bezier(motion, rank)
        new_motion = new_motion(nframes, -1, 3)
    else:
        new_motion = evaluate_bezier(motion, rank)
    return new_motion