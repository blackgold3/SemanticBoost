import os
import torch
import numpy as np
from tqdm import tqdm
os.environ['PYOPENGL_PLATFORM'] = 'egl'
import pandas as pd
from os.path import join as pjoin
from dataset.utils.dual_with_text import action2text
from dataset.utils.reverse import swap_text
from dataset.utils.convert_fps import convert_to_target_fps

def get_lower_case_name(text):
    lst = []
    for index, char in enumerate(text):
        if char.isupper() and index != 0:
            lst.append(" ")

        char = char.lower()
        lst.append(char)
    lst = "".join(lst)

    return lst

def amass_to_pose(src_path, save_path, target_fps=20, keep_betas=False):
    ex_fps = target_fps
    try:
        bdata = np.load(src_path, allow_pickle=True)
    except:
        print("skip -> ", src_path)
        return 0
    
    if "humanact" in src_path:
        np.save(save_path, bdata)
        return 20

    if "mocap_framerate" in bdata:
        fps = bdata['mocap_framerate']
    elif "mocap_frame_rate" in bdata:
        fps = bdata["mocap_frame_rate"]
    else:
        fps = 0
        return fps

    poses = bdata['poses']
    gender = str(bdata['gender'])
    trans = bdata["trans"]

    if isinstance(gender, bytes):
        gender = gender.decode('utf-8')
    if gender.startswith("b'"):
        gender = gender.replace("b'", "")
        gender = gender.replace("'", "")
    if gender == 'male':
        gender = np.ones([1, 1])
    elif gender == 'female':
        gender = np.zeros([1, 1])
    elif gender == 'neutral':
        gender = np.ones([1, 1]) * 2
    else:
        print(gender)


    poses = convert_to_target_fps(poses, fps, ex_fps)
    trans = convert_to_target_fps(trans, fps, ex_fps)

    pose_seq = np.concatenate([poses, trans], axis=1)
    gender = np.repeat(gender, poses.shape[0], axis=0)
    pose_seq = np.concatenate([pose_seq, gender], axis=1)

    if keep_betas:
        curr_b = bdata["betas"][np.newaxis, :10]
        curr_b = np.repeat(curr_b, poses.shape[0], axis=0)
        pose_seq = np.concatenate([pose_seq, curr_b], axis=1)
    
    np.save(save_path, pose_seq)

    return fps

def extract_amass_to_smpls(inputs_path, middle_root, save_dir, target_fps, keep_betas=True, index_path=None, index=0, texts_dir=None):
    '''
    index_path means HumanML3D labeled index_path
    '''
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(middle_root, exist_ok=True)

    if index_path is not None:  ########### HumanML3D
        humanml3d = ["ACCAD", "MPI_HDM05", "SFU", "BMLmovi", "CMU", "MPI_mosh", "EKUT", "KIT", "Eyes_Japan_Dataset", "BMLhandball", "Transitions_mocap",
                    "MPI_Limits", "HumanEva", "SSM_synced", "DFaust_67", "TotalCapture", "BioMotionLab_NTroje", "humanact12"]
        humanml3d = set(humanml3d)
        extra_paths = None
    else:
        humanml3d = ["TCD_handMocap", "CNRS", "DanceDB", "GRAB", "HUMAN4D", "SOMA", "WEIZMANN"]
        humanml3d = set(humanml3d)     
        extra_paths = []

    paths = []
    folders = []
    dataset_names = []
    for root, dirs, files in os.walk(inputs_path):
        dataset_name = root.split('/')[-2]      #### .../amass_smplh/dataset/subxxxx/
        if dataset_name not in humanml3d:
            continue

        folders.append(root)
        for name in files:
            if dataset_name not in dataset_names:
                dataset_names.append(dataset_name)
            paths.append(os.path.join(root, name))
            
    save_folders = [folder.replace(inputs_path, middle_root) for folder in folders]
    for folder in save_folders:
        os.makedirs(folder, exist_ok=True)
    group_path = [[path for path in paths if name in path] for name in dataset_names]
    all_count = sum([len(paths) for paths in group_path])
    cur_count = 0      
    real_count = 0       #### 13424
    
    for paths in group_path:
        dataset_name = paths[0].split('/')[-3]
        pbar = tqdm(paths)
        pbar.set_description('Processing: %s'%dataset_name)
        fps = 0
        for path in pbar:
            save_path = path.replace(inputs_path, middle_root)
            save_path = save_path[:-3] + 'npy'
            fps = amass_to_pose(path, save_path, target_fps, keep_betas)
            if fps != 0:
                real_count += 1

                if extra_paths is not None:
                    extra_paths.append(save_path)
                    
        cur_count += len(paths)
        print('Processed / All (fps %d): %d/%d'% (fps, cur_count, all_count) )
        print("real_count is %d"%(real_count)) 
    
    if index_path is not None:      ###### humanml3d
        index_file = pd.read_csv(index_path)
        total_amount = index_file.shape[0]
        fps = target_fps
            
        num_frames = 0
        for i in tqdm(range(total_amount)):
            curr_line = index_file.loc[i]
            source_path = curr_line['source_path']
            source_path = source_path.replace("./pose_data", middle_root)

            source_paths = []
            source_paths.append(source_path)

            if "humanact12" in source_path:
                source_paths.append(source_path.replace("/humanact12/humanact12", "/humanact12"))

            paths = source_path.split("/")
            path = paths[-1].split()
            path = "_".join(path)
            paths[-1] = path
            source_path = "/".join(paths)
            source_paths.append(source_path)

            source_path_bak = source_path.replace("_poses.", "_stageii.")
            new_name = curr_line['new_name']

            data = None
            source_paths.append(source_path_bak)

            max_num = 3
            for j in range(max_num):   ####### The amass-smplx names are different from amass-smplh, need to try different names
                replaced = "-" + "_" * (j + 2)
                source_path_bak_bak = source_path_bak.replace("-_", replaced)
                source_Path_bak_replace_bak = source_path_bak_bak.replace("_(", "__(")
                
                source_paths.append(source_path_bak_bak)
                source_paths.append(source_Path_bak_replace_bak)

            for source in source_paths:
                try:
                    data = np.load(source)
                    break
                except Exception as e:
                    print(e)
            
            assert data is not None, source_paths

            start_frame = curr_line['start_frame']
            end_frame = curr_line['end_frame']
            num_frames += (end_frame - start_frame)

            if 'humanact12' not in source_path:
                if 'Eyes_Japan_Dataset' in source_path:
                    data = data[3*fps:]
                if 'MPI_HDM05' in source_path:
                    data = data[3*fps:]
                if 'TotalCapture' in source_path:
                    data = data[1*fps:]
                if 'MPI_Limits' in source_path:
                    data = data[1*fps:]
                if 'Transitions_mocap' in source_path:
                    data = data[int(0.5*fps):]
                    
                data = data[start_frame:end_frame]
            else:
                data = data.reshape(data.shape[0], -1) 

            np.save(pjoin(save_dir, new_name), data)
    else:   #### new_dataset
        curr_index = index
        # extra_paths = sorted(extra_paths)
        total_amount = len(extra_paths)
        num_frames = 0
        for i in tqdm(range(len(extra_paths))):
            curr_path = extra_paths[i].replace(inputs_path, middle_root)
            if "GRAB" in curr_path:
                curr_text = curr_path.split("/")[-1].split("_")
                texts = []
                for j in range(len(curr_text)):
                    if curr_text[j].isnumeric():
                        break

                    texts.append(curr_text[j])
                    
                texts = " ".join(texts)
                texts= texts.replace("stageii", "")
                texts= texts.replace(".npz", "")
                texts= texts.replace(".npy", "")

                addition_desc = ""
                key_words1 = ["binoculars see", "bowl drink", "camera takepicture", "cup drink", "fryingpan cook", "mug drink", 
                            "waterbottle drink", "wineglass drink"]
                for key in key_words1:
                    if key in texts:
                        addition_desc = "with"
                    
                texts = texts.split()
                motion_target = texts[0]
                motion = texts[1::]
                motion = " ".join(motion)

                if addition_desc != "":
                    final_text = [motion, addition_desc, motion_target]
                else:
                    final_text = [motion, motion_target]
                action = " ".join(final_text)
                action = action.replace("takepicture", "take picture").replace("fryingpan", "frying pan").replace("waterbottle", "water bottle").replace("wineglass", "wine glass").replace("gamecontroller", "game controller").replace("piggybank", "piggy bank").replace("teapot", "tea pot").replace("stanfordbunny", "stanford bunny").replace("alarmclock", "alarm clock")
                action = action.replace("cylindermedium", "medium cylinder").replace("torusmedium", "medium torus").replace("cubesmall", "small cube").replace("spheremedium", "medium sphere").replace("pyramidlarge", "large pyramid").replace("spherelarge", "large sphere").replace("torussmall", "small torus").replace("lightbulb", "light bulb")
                action = action.replace("cubemedium", "medium cube").replace("cylindersmall", "small cylinder").replace("toruslarge", "large torus").replace("cylinderlarge", "large cylinder").replace("pyramidmedium", "medium pyramid").replace("cubelarge", "large cube").replace("spheresmall", "small sphere").replace("pyramidsmall", "small pyramid")

            elif "TCD_handMocap" in curr_path:
                curr_text = curr_path.split("/")[-1].split("_")[0].lower()
                if "bottle" in curr_text:
                    texts = "unscrew bottle while sitting"
                elif "count" in curr_text:
                    texts = "count while sitting"
                elif "direction" in curr_text:
                    texts = "give direction while sitting"
                elif "fingertp" in curr_text or "tposefinger" in curr_text:
                    texts = "curl the fingers sequentially while sitting"
                elif "flute" in curr_text:
                    texts = "play the flute while sitting"
                elif "grasp" in curr_text:
                    texts = "grasp something while sitting"
                elif "objecta" in curr_text:
                    texts = "tighten the bottle while sitting"
                elif "objectb" in curr_text:
                    texts = "juggle while sitting"
                elif "ok" in curr_text:
                    texts = "making the OK gesture while sitting"
                elif "point" in curr_text:
                    texts = "point somethings while sitting"
                elif "run" in curr_text:
                    texts = "run"
                elif "sign" in curr_text or curr_text == "v":
                    texts = "make some gesture while sitting"
                elif "talk" in curr_text:
                    texts = "talk while sitting"
                elif "type" in curr_text:
                    texts = "type while sitting"
                elif "walk" in curr_text:
                    texts = "walk"
                elif "wav" in curr_text:
                    texts = "wave hand while sitting"
                elif "writ" in curr_text:
                    texts = "write while sitting"

                action = texts
            
            elif "HUMAN4D" in curr_path:
                curr_text = curr_path.split("/")[-1].split("_")[1]
                texts = get_lower_case_name(curr_text)

                if "talking walking" in texts:
                    texts = "talking while walking"
                elif "punching kicking" in texts:
                    texts = "punching and kicking"
                elif "sitting standing" in texts:
                    texts = "sitting down and standing up repeatedly"

                texts = texts.split()

                final_text = []
                if texts[0] == "basketball":
                    final_text.append("playing")
                elif texts[0] == "physical":
                    final_text.append("doing")
                
                final_text += texts
                action = " ".join(final_text)
                
            elif "SOMA" in curr_path:
                curr_text = curr_path.split("/")[-1].split("_")[0]

                if curr_text == "random":
                    curr_text = "do motions randomly"

                action = curr_text

            elif "CNRS" in curr_path:
                action = "walk along a straight line"

            elif "WEIZMANN" in curr_path:
                speed = curr_path.split("/")[-1].split("_")[0]
                shape = curr_path.split("/")[-1].split("_")[1]
                shape = shape.split("(")[0]
                shape = shape.replace("CCW", "")
                shape = shape.replace("CW", "")

                texts = ["walk"]
                if shape.lower() in ["circle", "ellipse"]:
                    texts.append("counterclockwise")  
                texts.append("along")

                if shape == "AFig8":
                    texts.append("a figure 8 shape")
                elif shape.lower() in ["circle", "ellipse"]:
                    texts.append("a")
                    texts.append(shape.lower())
                elif shape == "SShapeLR":
                    texts.append("a S shape")
                elif shape == "StraightLong":
                    texts.append("a long straight line")
                else:
                    texts.append("a straight line")
                
                if speed.lower() == "fast":
                    texts.append("fast")
                elif speed.lower() == "slow":
                    texts.append("slowly")
                elif speed.lower() == "normal":
                    texts.append("with a normal speed")
                elif speed.lower() == "mixed":
                    if shape.lower() == "fastslow":
                        texts.append("first quickly and then slowlyz")
                    elif shape.lower() == "slowfast":
                        texts.append("first slowly and then fast")      
                
                action = " ".join(texts)

            elif "DanceDB" in curr_path:
                emotions = set(["angry", "curiosity", "happy", "nervous", "sad", "scary",
                                "annoyed", "bored", "excited", "miserable", "mix", "pleased",
                                "relaxed", "satisfied", "tired", "afraid", "neutral"])
                
                texts = curr_path.split("/")[-1].replace("_stageii.npz", "")
                texts = texts.replace("_poses.npz", "")
                texts = texts.replace("_stageii.npy", "")
                texts = texts.replace("_poses.npy", "")
                texts = texts.replace("_C3D", "")
                texts = texts.replace("_CY", "")
                texts = texts.replace("_v1", "")
                texts = texts.replace("_v2", "")
                texts = texts.replace("_v3", "")
                texts = texts.replace("_v4", "")
                texts = texts.replace("_01", "")
                texts = texts.replace("_1os", "")
                texts = texts.replace("_2os", "")
                texts = texts.replace("_3os", "")
                texts = texts.replace("_1", "")
                texts = texts.split("_")
                dtype = None
                emotion = None
            
                if texts[1] in ["Aristeidou", "Aristidou"]:
                    dtype = texts[2:]
                elif texts[1].lower() in emotions:
                    emotion = texts[1:]
                elif texts[1].lower() in ["theodoros"]:
                    dtype = texts[0:1]
                else:
                    dtype = texts[1:]

                texts = []
                if emotion is not None:
                    emotion = " ".join(emotion)
                    texts.append("dance")
                    if emotion.lower() in ["angry", "annoyed", "excited", "afraid"]:
                        texts.append("an")
                    else:
                        texts.append("a")
                    texts.append(emotion.lower())
                    texts.append("dance")
                elif dtype is not None:
                    dtype = " ".join(dtype)
                    texts.append("dance")
                    texts.append(dtype.lower())
                else:
                    texts.append("dance") 

                action = " ".join(texts)  

            else:
                action = ""

            curr_smpl = np.load(curr_path)
            curr_frames = curr_smpl.shape[0]

            if curr_frames > 250:
                curr_smpl = torch.from_numpy(curr_smpl)
                curr_smpl = torch.split(curr_smpl, 200, dim=0)
            else:
                curr_smpl = torch.from_numpy(curr_smpl)
                curr_smpl = [curr_smpl]

            for smpl in curr_smpl:
                if smpl.shape[0] < 40:
                    continue
                else:
                    num_frames += smpl.shape[0]
                    
                text_save_path = os.path.join(texts_dir, "%06d.txt"%(curr_index))
                text_save_pathM = os.path.join(texts_dir, "M%06d.txt"%(curr_index))

                save_text = action2text(action)
                save_textM = action2text(swap_text(action))

                with open(text_save_path, "w") as f:
                    f.writelines([save_text + " \n"])

                with open(text_save_pathM, "w") as f:
                    f.writelines([save_textM + " \n"])                 

                np.save(pjoin(save_dir, "%06d.npy"%(curr_index)), smpl.numpy())
                curr_index += 1
            

    return num_frames   

if __name__ == "__main__":
    '''
    read from humanml3d datasets
    '''
    inputs_path = "/data/TTA/data/amass_smplh"
    index_path = 'T2M/index.csv'
    save_dir = '/data/TTA/data/joints'
    save_root = '/data/TTA/data/pose_data'
    target_fps = 20
    keep_betas = True

    num_frames1 = extract_amass_to_smpls(inputs_path, save_root, save_dir, target_fps, keep_betas, index_path)
    num_frames2 = extract_amass_to_smpls(inputs_path, save_root, save_dir, target_fps, keep_betas)

    print("Origin data time is:%.3f min"%((num_frames1) / target_fps / 60))
    print("New data time is:%.3f min"%((num_frames2) / target_fps / 60))
    print("total time is:%.3f min"%((num_frames1 + num_frames2) / target_fps / 60))
