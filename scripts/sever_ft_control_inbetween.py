import numpy as np
import subprocess
import platform
import os

class sever_motion_concat(object):
    def __init__(self, checkpoints, begin_part, end_part, motion_clip_size, target_dir):
        '''
        checkpoints: 模型的参数和相关配置信息
        begin_part: 第一个文件用来做衔接任务的帧数，例如选20，就会截取第一个文件的最后 20帧做衔接任务
        end_part: 第二个文件用来做衔接任务的帧数
        motion_clip_size: 为了凸显动作的连接效果，截断第一个动作尾部和第二个动作头部的长度，因为动作头尾经常是静止的
        target_dir: 本次任务的保存路径
        '''
        self.checkpoints = checkpoints
        self.begin_part = begin_part
        self.end_part = end_part
        self.motion_clip_size = motion_clip_size
        self.target_dir = target_dir
        os.makedirs(self.target_dir, exist_ok=True)
    
    def __call__(self, file1, file2, concat_length, prompt=None, visualize=True):
        '''
        file1: 第一个文件的路径
        file2: 第二个文件的路径
        concat_length: 衔接区域的长度，总体上衔接生成的动作组成是 begin_part + blend1 + 衔接区域 + blend2 + end_part 
        prompt: 决定 nocond 或者 text 生成
        '''

        ########## 文件处理 ##############
        file1 = np.load(file1) ### 动作1截掉尾部的若干帧
        file2 = np.load(file2)  ### 动作2截掉头部的若干帧
        if self.motion_clip_size != 0:
            file1 = file1[:-self.motion_clip_size]
            file2 = file2[self.motion_clip_size:]
        file1_path = os.path.join(self.target_dir, "file1.npy")
        file2_path = os.path.join(self.target_dir, "file2.npy")
        np.save(file1_path, file1)
        np.save(file2_path, file2)

        if visualize:
            cmd = "python -m sample.visual_file --source {} --mode smr --target_dir {} --target_name file1 --render_mode pyrender --out_mode video".format(file1_path, self.target_dir)
            subprocess.call(cmd, shell=platform.system() != 'Windows')
            cmd = "python -m sample.visual_file --source {} --mode smr --target_dir {} --target_name file2 --render_mode pyrender --out_mode video".format(file2_path, self.target_dir)
            subprocess.call(cmd, shell=platform.system() != 'Windows')

        synthesize_first_part = file1[-self.begin_part:]
        synthesize_last_part = file2[:self.end_part]

        ############ 构造 help_file ##########
        nframes = self.begin_part + concat_length + self.end_part
        help_file = np.zeros([nframes, 269])
        help_file[:self.begin_part] = synthesize_first_part
        help_file[-self.end_part:] = synthesize_last_part
        help_file_path = os.path.join(self.target_dir, "help_file.npy")
        np.save(help_file_path, help_file)

        ############# 生成命令生成 #########
        if prompt is None:
            mode = "nocond"
            prompt = "null"
        else:
            mode == "text"
        
        cmd = "python -m sample.visual_t2m --save --target_dir {} --target_name {} ".format(self.target_dir, "domain")
        cmd += "--checkpoints {} --help_file {} ".format(self.checkpoints, help_file_path)
        cmd += "--mode {} --prompt {} --nframes {} ".format(mode, prompt, nframes)

        if not visualize:
            cmd += " --novis "
        subprocess.call(cmd, shell=platform.system() != 'Windows')
        target_path = os.path.join(self.target_dir, "domain.npy")
        file_mid = np.load(target_path)
        
        ############## 拼接 ###################
        if visualize:
            files = np.concatenate([file1, file_mid[self.begin_part:nframes-self.end_part], file2], axis=0)
            print("===========", files.shape)
            concat_path = os.path.join(self.target_dir, "concat.npy")
            np.save(concat_path, files)
            cmd = "python -m sample.visual_file --source {} --mode smr --target_dir {} --target_name concat --render_mode pyrender --out_mode video".format(concat_path, self.target_dir)
            subprocess.call(cmd, shell=platform.system() != 'Windows')
        return file_mid[self.begin_part:nframes-self.end_part]

if __name__ == "__main__":
    checkpoints = "/apdcephfs_cq11/share_1567347/share_info/kleinhe/checkpoints/text2motion/train_ft_control_inbetween_20_20_lowlr/S040000_F0.2014_T0.5115.pth"
    target_dir = "results/inbetween"
    file1 = "/apdcephfs_cq11/share_1567347/share_info/kleinhe/motionDataset/HumanML3D/smr_rep/000021.npy"
    file2 = "/apdcephfs_cq11/share_1567347/share_info/kleinhe/motionDataset/HumanML3D/smr_rep/000000.npy"
    generator = sever_motion_concat(checkpoints, 20, 20, 20, target_dir)
    generator(file1, file2, 40, None)

    #### length = 0 + 20 + (20) + 20 + 0, inbetween 的头尾 20帧是固定的，这个是 ft_control_mask 决定的
    #### 所以 __call__ 里的 content_length 是真实的生成长度，取消了 blend_size 以后，同样的长度，可以生成更长的内容
