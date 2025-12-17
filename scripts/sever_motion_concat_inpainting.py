import numpy as np
import subprocess
import platform
import os

class sever_motion_concat(object):
    def __init__(self, rep, checkpoints, blend_size, begin_part, end_part, motion_clip_size, target_dir):
        '''
        checkpoints: 模型的参数和相关配置信息
        blend_size: 混合过渡区域大小, 越大过渡越平滑，但太大也容易出现比较长时间原地不动的情况
        begin_part: 第一个文件用来做衔接任务的帧数，例如选20，就会截取第一个文件的最后 20帧 + blend_size 做衔接任务
        end_part: 第二个文件用来做衔接任务的帧数
        motion_clip_size: 为了凸显动作的连接效果，截断第一个动作尾部和第二个动作头部的长度，因为动作头尾经常是静止的
        target_dir: 本次任务的保存路径
        '''
        self.checkpoints = checkpoints
        self.blend_size = blend_size
        self.begin_part = blend_size + begin_part
        self.end_part = blend_size + end_part
        self.motion_clip_size = motion_clip_size
        self.target_dir = target_dir
        os.makedirs(self.target_dir, exist_ok=True)
        if rep == "smr":
            self.njoints = 269
        elif rep == "JP":
            self.njoints = 198
        self.rep = rep
    
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
            cmd = "python -m sample.visual_file --source {} --mode {} --target_dir {} --target_name file1 --render_mode pyrender --out_mode video".format(file1_path, self.rep, self.target_dir)
            subprocess.call(cmd, shell=platform.system() != 'Windows')
            cmd = "python -m sample.visual_file --source {} --mode {} --target_dir {} --target_name file2 --render_mode pyrender --out_mode video".format(file2_path, self.rep, self.target_dir)
            subprocess.call(cmd, shell=platform.system() != 'Windows')

        synthesize_first_part = file1[-self.begin_part:]
        synthesize_last_part = file2[:self.end_part]

        ############ 构造 help_file ##########
        nframes = self.begin_part + concat_length + self.end_part
        help_file = np.zeros([nframes, self.njoints])
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
        
        cmd = "python -m sample.visual_inpainting --save --target_dir {} --target_name {} ".format(self.target_dir, "domain")
        cmd += "--checkpoints {} --help_file {} ".format(self.checkpoints, help_file_path)
        cmd += "--mode {} --prompt {} --nframes {} ".format(mode, prompt, nframes)
        cmd += "--capture_begin {} {} --capture_end {} {} ".format(0, nframes - self.end_part + self.blend_size, self.begin_part - self.blend_size, nframes)
        cmd += "--target_begin {} {} --target_end {} {} ".format(0, nframes - self.end_part + self.blend_size, self.begin_part - self.blend_size, nframes)
        cmd += "--blend_size {}".format(self.blend_size)

        if not visualize:
            cmd += " --novis "

        subprocess.call(cmd, shell=platform.system() != 'Windows')

        ############## 拼接 ###################
        target_path = os.path.join(self.target_dir, "domain.npy")
        file_mid = np.load(target_path)
        
        if visualize:
            files = np.concatenate([file1[:-self.blend_size], file_mid[self.begin_part-self.blend_size:nframes-self.end_part+self.blend_size], file2[self.blend_size:]], axis=0)
            print("===========", files.shape)
            concat_path = os.path.join(self.target_dir, "concat.npy")
            np.save(concat_path, files)
            cmd = "python -m sample.visual_file --source {} --mode {} --target_dir {} --target_name concat --render_mode pyrender --out_mode video".format(concat_path, self.rep, self.target_dir)
            subprocess.call(cmd, shell=platform.system() != 'Windows')
        return file_mid[self.begin_part-self.blend_size:nframes-self.end_part+self.blend_size]

if __name__ == "__main__":
    checkpoints = "/apdcephfs_cq11/share_1567347/share_info/kleinhe/checkpoints/text2motion/train_amass_noencode11_nospeed/model_300000.pth"
    target_dir = "results/generate"
    file1 = "/data/TTA/motionWE/results/nospeed/long-pose.npy"
    file2 = "results/nospeeed/backflip-pose.npy"
    generator = sever_motion_concat("JP", checkpoints, 10, 20, 20, 20, target_dir)
    generator(file1, file2, 10, None)

    #### length = 10 + 20 + 10 + 20 + 10
