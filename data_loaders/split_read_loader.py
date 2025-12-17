import torch
from torch.utils import data
import numpy as np
from os.path import join as pjoin
import random
from tqdm import tqdm
from data_loaders.control_mask import get_control_mask
from eval.word_vectorizer import WordVectorizer
from dataset.utils.rotation_conversions import *
'''For use of training text-2-motion generative model'''
class Text2MotionDataset(data.Dataset):
    def __init__(self, args, train=True):
        self.data_root = args.data_root
        self.rep = args.rep
        self.train = train
        self.max_text_len = 77
        self.control_signal = args.model_mode  
        self.max_motion_length = args.nframes
        self.meta_dir = args.evaluate_base_dir
        self.min_motion_len = 40
        if train:
            self.w_vectorizer = None
        else:
            word_path = args.w_vectorizer_path
            w_vectorizer = WordVectorizer(word_path, "our_vab")
            self.w_vectorizer = w_vectorizer

        fps = 20
        self.joints_num = 22
        feature_length = args.njoints

        if self.rep == "t2m":
            extension = ""
        elif self.rep == "smr":
            extension = "-smr"
        elif self.rep == "JP":
            extension = "-JP"

        self.mean = np.load(pjoin(self.data_root, 'Mean{}.npy'.format(extension)))
        self.std = np.load(pjoin(self.data_root, 'Std{}.npy'.format(extension)))
        self.mean_eval = np.load(pjoin(self.meta_dir, 'mean.npy'))
        self.std_eval = np.load(pjoin(self.meta_dir, 'std.npy'))
        self.feature_length = feature_length

        if train:
            split_file = pjoin(self.data_root, 'train{}.txt'.format(extension))
        else:
            split_file = pjoin(self.data_root, 'test{}.txt'.format(extension))

        self.total_samples = 0

        name_list = []
        data_dict = {}
        with open(split_file, "r") as f:
            lines = f.readlines()
            lines = [line.strip() for line in lines]
            for line in lines:
                files = line.split("#")
                motion_file = files[0]
                text_file = files[1]
                motions = np.load(motion_file, allow_pickle=True)
                texts = np.load(text_file, allow_pickle=True)

                # 将加载的数据转换为字典
                motions_npz = motions.items()
                texts_npz = texts.items()

                motions = {}
                texts = {}
                for name, array in tqdm(motions_npz):
                    motions[name] = array
                for name, array in tqdm(texts_npz):
                    texts[name] = array

                names = list(motions.keys())
                for i in tqdm(range(len(names)), desc=line):
                    curr_name = names[i]
                    curr_motion = motions[curr_name]
                    curr_text = texts[curr_name].tolist()
                  
                    if curr_motion.shape[0] < self.min_motion_len:
                        continue
                    elif np.isnan(curr_motion).any():
                        continue
                    else:   
                        caption, tokens, f_tag, to_tag = curr_text[0].split("#")
                        f_tag = float(f_tag)
                        to_tag = float(to_tag)
                        f_tag = 0.0 if np.isnan(f_tag) else f_tag
                        to_tag = 0.0 if np.isnan(to_tag) else to_tag
                        if to_tag > 0 and f_tag < to_tag:           ###### HumanML3D 有一定的人工操作，标注了起点和终点
                            n_motion = curr_motion[int(f_tag*fps) : int(to_tag*fps)]
                            if (n_motion.shape[0]) < self.min_motion_len:
                                continue     
                            else:
                                curr_motion = n_motion

                        curr_data = {
                            "motion":curr_motion,
                            "text":curr_text,
                        }
                        name_list.append(curr_name)
                        data_dict[curr_name] = curr_data
                        self.total_samples += 1

        print("total_samples ==================================== %06d"%(self.total_samples))
        self.data_dict = data_dict
        self.name_list = name_list

    def inv_transform(self, data):
        return data * self.std + self.mean

    def forward_transform(self, data):
        return (data - self.mean) / self.std

    def eval_inv_transform(self, data):
        data = self.inv_transform(data)
        data = (data - self.mean_eval) / self.std_eval
        return data
        
    def __len__(self):
        return self.total_samples

    def __getitem__(self, idx):
        name = self.name_list[idx]
        data = self.data_dict[name]

        motion = data["motion"] 

        '''
        随机旋转，数据增强
        '''
        angel = random.randint(0, 359)
        rotation_angel = np.deg2rad(angel)
        R_y = np.array([
            [np.cos(rotation_angel), 0, np.sin(rotation_angel)],
            [0, 1, 0],
            [-np.sin(rotation_angel), 0, np.cos(rotation_angel)]
        ]) 
        motion[:, :3] = np.matmul(R_y, motion[:, :3].T).T
        joints =  motion[:, 3:66].reshape(motion.shape[0], -1, 3).transpose(0, 2, 1)    ### [N, 3, 21]
        joints = np.matmul(R_y, joints).transpose(0, 2, 1).reshape(motion.shape[0], -1)  ### [N ,21, 3]
        motion[:, 3:66] = joints
        root = torch.from_numpy(motion[:, 66:72])
        root = rotation_6d_to_matrix(root)  ### [N, 3, 3]
        root = torch.from_numpy(R_y).float() @ root 
        root = matrix_to_rotation_6d(root)
        root = root.numpy()
        motion[:, 66:72] = root

        '''
        文本处理
        '''
        text = random.choice(data["text"])
        caption, tokens, _, _ = text.split("#")
        
        if not self.train:
            tokens = tokens.split()
            if len(tokens) < self.max_text_len:
                # pad with "unk"
                tokens = ['sos/OTHER'] + tokens + ['eos/OTHER']
                sent_len = len(tokens)
                tokens = tokens + ['unk/OTHER'] * (self.max_text_len + 2 - sent_len)
            else:
                # crop
                tokens = tokens[:self.max_text_len]
                tokens = ['sos/OTHER'] + tokens + ['eos/OTHER']
                sent_len = len(tokens)
            pos_one_hots = []
            word_embeddings = []
            for token in tokens:
                word_emb, pos_oh = self.w_vectorizer[token]
                pos_one_hots.append(pos_oh[None, :])
                word_embeddings.append(word_emb[None, :])
            pos_one_hots = np.concatenate(pos_one_hots, axis=0)
            word_embeddings = np.concatenate(word_embeddings, axis=0)
        else:
            pos_one_hots = np.zeros([self.max_text_len + 2, 15])
            word_embeddings = np.zeros([self.max_text_len + 2, 300])
            sent_len = 1      
        

        '''
        后处理，对齐长度
        '''
        m_length = motion.shape[0]
        in_painting_mask = get_control_mask(self.control_signal, [m_length, self.feature_length])

        if m_length > self.max_motion_length:
            offset = random.randint(0, m_length - self.max_motion_length)
            motion = motion[offset:offset + self.max_motion_length]
            in_painting_mask = in_painting_mask[offset:offset + self.max_motion_length]
            m_length = self.max_motion_length

        "Z Normalization"
        motion = (motion - self.mean) / self.std

        if m_length < self.max_motion_length:
            motion = np.concatenate([motion,
                                     np.zeros((self.max_motion_length - m_length, *motion.shape[1::]))
                                     ], axis=0)
            in_painting_mask = np.concatenate([in_painting_mask,
                                        np.zeros((self.max_motion_length - m_length, *in_painting_mask.shape[1::]))
                                        ], axis=0)      
  
        
        mask = np.zeros([1, 1, self.max_motion_length])
        mask[:, :, :m_length] = 1
        
        return word_embeddings, pos_one_hots, caption, sent_len, motion.astype(np.float32), m_length, mask, in_painting_mask


def DATALoader(args, batch_size, num_workers = 8, train=True) : 
    
    val_loader = torch.utils.data.DataLoader(Text2MotionDataset(args, train=train),
                                              batch_size,
                                              shuffle = True,
                                              num_workers=num_workers,
                                              drop_last = False)
    return val_loader


def cycle(iterable):
    while True:
        for x in iterable:
            yield x