from argparse import Namespace
import os
import numpy as np
import torch
from src_t2m.model_util import create_model_and_diffusion, load_model_wo_clip
from model.cfg_sampler import ClassifierFreeSampleModel
import yaml
from data_loaders.control_mask import get_control_mask
from sample.double_take import double_take
class ModelInfer(object):
    def __init__(self, checkpoints, device="cuda"):
        checkpoints_dir = checkpoints.split("/")[:-1]
        checkpoints_dir = "/".join(checkpoints_dir)
        hyper_path = os.path.join(checkpoints_dir, "motionMonitor", "version_0", "hparams.yaml")
        with open(hyper_path, 'r') as f:
            lines = f.readlines()
            lines = lines[1:]
        with open("temp.yaml", "w") as f:
            f.writelines(lines)
        with open("temp.yaml", "r") as f:
            temp_dict = yaml.safe_load(f)
        os.remove("temp.yaml")

        args = Namespace()
        for key, value in temp_dict.items():
            setattr(args, key, value)
        self.args = args
        self.rep = args.rep
        self.device = device
        self.model_mode = args.model_mode

        ############# 均值方差 ###########
        data_root = self.args.data_root
        if self.rep == "smr":
            extension = "-smr"
        elif self.rep == "JP":
            extension = "-JP"
        else:
            extension = ""
        self.mean = torch.from_numpy(np.load(os.path.join(data_root, 'Mean{}.npy'.format(extension)))).to(self.device)
        self.std = torch.from_numpy(np.load(os.path.join(data_root, 'Std{}.npy'.format(extension)))).to(self.device)

        ########## 模型加载 ##############
        model, diffusion = create_model_and_diffusion(self.args, self.model_mode)
        try:
            static_dict = torch.load(checkpoints, map_location="cpu")
            load_model_wo_clip(model, static_dict["ema"])
        except:
            print("================ Not load checkpoints ================")
        self.model = ClassifierFreeSampleModel(model)
        self.model.eval()
        self.model = self.model.to(device)
        self.diffusion = diffusion

    def inpainting_mask(self, help_file, nframes, capture_begin, capture_end, target_begin, target_end, blend_size=10):
        '''
        基于纯粹的 inpainting 方案把 help_file 中的一部分保留下来, 开始帧和结尾帧参数 capture_begin 和 capture_end
        这部分将会出现在生成结果的指定部分 target_begin 到 target_end 之间
        help_file [nframes, 269]
        blend_size 是混合区域，保证过度平滑的帧数, 需要提供 help_file 这一部分的源数据用于过度, 如果是头尾区域，就不需要
        '''
        if help_file is not None and os.path.exists(help_file):
            help_file = np.load(help_file)
        elif help_file is not None and isinstance(help_file, np.ndarray):
            pass
        else:
            raise ValueError("Not a correct help file input")
        
        if target_end - target_begin != capture_end - capture_begin:
            raise ValueError("Not a matched domain")
        
        '''
        同时兼容 first 保留生成, last 保留生成 和 mid 保留生成
        如果需要做 inbetween, 多次调用本函数, inpainted_motion 和 inpainting_mask 累加
        '''

        if target_begin == 0 and target_end == nframes: #### 复现生成动作
            blend_begin = 0
            blend_end = 0
        elif target_begin == 0:   ##### 把一段内容放在生成目标的起点
            blend_begin = 0
            blend_end = blend_size
        elif target_end == nframes:   ##### 把一段内容放在生成目标的终点
            blend_begin = blend_size
            blend_end = 0
        else:
            blend_begin = blend_size
            blend_end = blend_size
        
        if capture_end + blend_end > help_file.shape[0] or capture_begin < blend_begin:
            raise ValueError("Over the max capture length")
        
        if target_end + blend_end > nframes or target_begin < blend_begin:
            raise ValueError("Over the max target length")

        help_file = torch.from_numpy(help_file).to(self.device)     ### [nframes, 269]
        help_file = (help_file - self.mean) / self.std
        help_file = help_file[capture_begin-blend_begin:capture_end+blend_end].permute(1, 0).unsqueeze(0).unsqueeze(2)   ### [1, 269, 1, nframes]
        inpainted_motion = torch.zeros([1, self.model.njoints, self.model.nfeats, nframes]).to(self.device)
        inpainted_motion[:, :, :, target_begin-blend_begin:target_end+blend_end] = help_file
        inpainting_mask = torch.zeros([1, self.model.njoints, self.model.nfeats, nframes], dtype=torch.float,device=self.device)  # True means use gt motion
        inpainting_mask[:, 3:, :, target_begin:target_end] = 1

        ########### 混合区域线性插值生成，前三维是 translation 相关部分，不适合利用 help_file 的数据，只插值动作关节点和旋转相关信息 ########
        if blend_begin != 0:
            ratio_list = torch.arange(0.0, 0.85, 0.85 / int(blend_begin))
            inpainting_mask[:, 3:, :, target_begin-blend_begin:target_begin] = ratio_list.unsqueeze(0).unsqueeze(0).unsqueeze(0).repeat(1, self.model.njoints - 3, self.model.nfeats, 1)
        if blend_end != 0:
            reverse_ratio_list = torch.arange(0.85, 0, -0.85 / int(blend_end))
            inpainting_mask[:, 3:, :, target_end:target_end+blend_end] = reverse_ratio_list.unsqueeze(0).unsqueeze(0).unsqueeze(0).repeat(1, self.model.njoints - 3, self.model.nfeats, 1)
        
        return inpainted_motion, inpainting_mask

    def ft_control_model_kwargs(self, model_kwargs, help_file, nframes):
        '''
        help_file 的长度应该和 nframes 相同
        用于 ft_control 提取 help_file 中的信息
        部分 ft_control 功能，长度大小不一致不一定会生效
        '''

        if help_file is None:
            print(">>>>>>>>>>>>>>>> Do not input any helpfile <<<<<<<<<<<<")
            help_file = np.zeros([nframes, self.model.njoints * self.model.nfeats], dtype=np.float32)
        elif os.path.exists(help_file):
            help_file = np.load(help_file)
        
        if help_file.shape[1] != self.model.njoints * self.model.nfeats:
            raise ValueError("Help file is not the same rep as model")
        
        if help_file.shape[0] != nframes:
            print(">>>>>>>>>>>>>>>> Help file is not align with nframes [%d | %d] <<<<<<<<<<<<"%(help_file.shape[0], nframes))
            if help_file.shape[0] < nframes:
                help_file = np.concatenate([help_file, np.zeros([nframes - help_file.shape[0], help_file.shape[1]])], axis=0)
            else:
                help_file = help_file[:nframes]
        
        help_file = torch.from_numpy(help_file).to(self.device)     ### [nframes, 269]
        help_file = (help_file - self.mean) / self.std
        help_file = help_file.permute(1, 0).unsqueeze(0).unsqueeze(2)   ### [1, 269, 1, nframes]      
        model_kwargs['y']['inpainted_motion'] = help_file

        mask = get_control_mask(self.model_mode, (nframes, self.model.njoints * self.model.nfeats))     ### [nframes, nfeats]
        mask = torch.from_numpy(mask).to(self.device)     ### [nframes, 269]
        mask = mask.permute(1, 0).unsqueeze(0).unsqueeze(2)   ### [1, 269, 1, nframes]      
        model_kwargs['y']['inpainting_mask'] = mask
        return model_kwargs

    def doubleTake(self, prompts, nframes, handshake_size, blend_size):
        print("sample mode - double_take long motion")
        if len(prompts) != len(nframes):
            raise ValueError("The number of prompts is not aligned with the number of nframes")
        
        batch_size = len(prompts)
        default_frames = max(nframes)
        model_kwargs = {
            "y":{
                "mask":torch.ones((batch_size, 1, 1, default_frames), dtype=torch.float32).to(self.device),
                "lengths":torch.LongTensor(nframes).to(self.device),
                "text":prompts,
                "scale":torch.ones([batch_size], dtype=torch.float32).to(self.device) * self.args.guidance_param
            }
        }
        sample, step_sizes = double_take(model_kwargs, self.model, self.diffusion, handshake_size, blend_size, self.device)
        sample = sample.permute(0, 2, 3, 1).float() 
        sample = sample * self.std + self.mean   
        all_motions = sample[0][0]   ### [nframes, njoints]
        return all_motions

    def __call__(self, nframes, model_kwargs):
        sample_fn = self.diffusion.p_sample_loop
        if "inpainting_mask" in model_kwargs["y"] and "inpainted_motion" in model_kwargs["y"] and not self.model_mode.startswith("ft_control"):      ###### 用来对初始 noise 进行一定的赋值，让约束强度更高，非必要项
            eval_mask = 1 - model_kwargs['y']['inpainting_mask']
            init_image = model_kwargs['y']['inpainted_motion']
        else:
            eval_mask = None
            init_image = None

        sample = sample_fn(
            self.model,
            (1, self.model.njoints, self.model.nfeats, nframes),
            clip_denoised=False,
            model_kwargs=model_kwargs,
            skip_timesteps=0,  # 0 is the default value - i.e. don't skip any step
            init_image=init_image,
            progress=True,
            dump_steps=None,
            noise=None,
            const_noise=False,
            eval_mask=eval_mask,
            ddim=False
        )
        sample = sample["output"]
        sample = sample.permute(0, 2, 3, 1).float()  ##### [bs, 1, nframes, njoints]
        sample = sample * self.std + self.mean   

        all_motions = sample[0][0]   ### [nframes, njoints]
        return all_motions