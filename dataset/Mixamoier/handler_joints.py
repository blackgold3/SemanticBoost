import numpy as np
import os
from dataset.utils.reverse import swap_text
from dataset.utils.dual_with_text import action2text

def extracted_pose_handler(pose_dir, text_dir, new_pose_dir):
    os.makedirs(new_pose_dir, exist_ok=True)
    os.makedirs(text_dir, exist_ok=True)
    index = 0
    for root, dirs, files in os.walk(pose_dir):
        for path in files:
            if not path.endswith(".npy"):
                continue
            else:
                name = path
                action = name.split(".")[0]
                action = action.split("(")[0]
                action = action.replace("_180", "t180").replace("_1H", "t1h").replace("_2H", "t2h").replace("_360", "t360").replace("_45", "t45").replace("_1990", "t1990")
                action = action.replace("_1", "").replace("_2", "").replace("_3", "").replace("_4", "").replace("_5", "").replace("01", "").replace("02", "").replace("03", "").replace("04", "").replace("05", "")
                action = action.replace("t180","_180").replace("t360","_360").replace("t45","_45").replace("t2h","_2H").replace("t1h","_1H").replace("t1990","_1990")
                action = action.split("_")
                action = " ".join(action)
                action = action.lower()
                reverse_action = swap_text(action)

                curr_path = os.path.join(root, path)
                curr_pose = np.load(curr_path)
                speed = curr_pose[1::] - curr_pose[:-1:]
                speed = speed.sum(axis=1)
                speed = speed[::-1]
                mask = speed < 1e-4
                length = 0
                for j in range(mask.shape[0]):
                    if not mask[j]:
                        length = j
                        break
                length = max(curr_pose.shape[0] - length, 41)
                curr_pose = curr_pose[:length:]

                pose_save_path = os.path.join(new_pose_dir, "%06d.npy"%(index))
                text_save_path = os.path.join(text_dir, "%06d.txt"%(index))
                reverse_text_save_path = os.path.join(text_dir, "M%06d.txt"%(index))

                np.save(pose_save_path, curr_pose)

                with open(text_save_path, "w") as f:
                    f.writelines([action2text(action) + " \n"])
                
                with open(reverse_text_save_path, "w") as f:
                    f.writelines([action2text(reverse_action) + " \n"])

                index += 1

