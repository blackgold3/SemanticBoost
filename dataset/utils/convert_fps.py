import torch
import numpy as np
import torch.nn.functional as F
import math
def convert_to_target_fps(motion, source_fps, target_fps):
    if source_fps > target_fps:
        if source_fps // target_fps == (source_fps / target_fps):
            interp = int(source_fps // target_fps)
            return motion[::interp]
        else:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            motion = torch.from_numpy(motion).permute(1, 0).unsqueeze(0).unsqueeze(2).to(device)    ### [1, C, 1, Frames]
            interp = source_fps / target_fps
            target_frames = int(motion.shape[-1] / interp)
            new_interp = math.ceil(interp)
            interpolate_res = new_interp * target_frames
            new_motion = F.interpolate(motion, size=(1, interpolate_res), mode="bilinear")        
            new_motion = new_motion.squeeze().permute(1, 0)
            new_motion = new_motion.cpu().numpy()
            new_motion = new_motion[::new_interp]
            return new_motion
    elif source_fps < target_fps:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        motion = torch.from_numpy(motion).permute(1, 0).unsqueeze(0).unsqueeze(2).to(device)    ### [1, C, 1, Frames]
        interp = target_fps / source_fps
        target_frames = int(motion.shape[-1] * interp)
        new_motion = F.interpolate(motion, size=(1, target_frames), mode="bicubic")        
        new_motion = new_motion.squeeze().permute(1, 0)
        new_motion = new_motion.cpu().numpy()
        return new_motion  
    else:
        return motion

