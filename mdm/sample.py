from argparse import Namespace
import torch
from mdm.dataset.recover_joints import recover_from_ric
from mdm.model.cfg_sampler import ClassifierFreeSampleModel
from mdm.model_util import create_model_and_diffusion, load_model_wo_clip, create_trt_model
import os
import numpy as np
from mdm.dataset.recover_smr import *
import json
from mdm.double_take import double_take

class Predictor(object):
    def __init__(self, **kargs):
        self.path = kargs["path"]
        self.handshake_size = 20
        self.blend_size = 10
        self.speedup = kargs.get("speedup", 1)

        args = Namespace()
        with open(self.path["config"], 'r') as f:
            params1 = json.load(f)
        for key, value in params1.items():
            setattr(args, key, value)

        args.quantization = False
        mode = kargs.get("mode", "camd")
        
        if mode != "mdm" and (not os.path.exists(self.path[f"{mode}1"]) or not os.path.exists(self.path[f"{mode}2"])):
            self.speedup = 0

        if mode == "camd":
            args.arch = "llama_decoder_static"
            args.encode_full = 2
            args.txt_tokens = 1
            args.model_path = self.path["camd"]
            args.rep = "smr"
            args.conv_bias = False
            args.conv_norm = "rmsnorm"
            args.conv_activate = "silu"
            args.trans_activate = "swiglu"
            args.quantization = self.speedup == 1

        elif mode == "camd-augment":
            args.arch = "llama_decoder_static"
            args.encode_full = 2
            args.txt_tokens = 1
            args.model_path = self.path["camd-augment"]
            args.rep = "smr"
            args.conv_bias = False
            args.conv_norm = "rmsnorm"
            args.conv_activate = "silu"
            args.trans_activate = "swiglu"
            args.quantization = self.speedup == 1

        elif mode == "mdm":
            args.arch = "trans_enc"
            args.encode_full = 0
            args.txt_tokens = 0
            args.model_path = self.path["mdm"]
            args.rep = "t2m"

        elif mode == "ncamd":
            args.arch = "llama_decoder_rope"
            args.encode_full = 2
            args.txt_tokens = 2
            args.model_path = self.path["ncamd"]
            args.rep = "smr"
            args.conv_bias = True
            args.conv_norm = "layernorm"
            args.conv_activate = "relu"
            args.trans_activate = "swiglu"
            args.quantization = self.speedup == 1
        
        elif mode == "ncamd-augment":
            args.arch = "llama_decoder_rope"
            args.encode_full = 2
            args.txt_tokens = 2
            args.model_path = self.path["ncamd-augment"]
            args.rep = "smr"
            args.conv_bias = True
            args.conv_norm = "layernorm"
            args.conv_activate = "relu"
            args.trans_activate = "swiglu"     
            args.quantization = self.speedup == 1  

        self.skip_steps = kargs.get("skip_steps", 0)
        self.device = kargs.get("device", "cpu")
        self.length = kargs.get("length", "120")
        self.args = args
        self.rep = args.rep
        self.num_frames = args.num_frames
        self.condition = kargs.get("condition", "text")
        if self.condition == "uncond":
            self.args.guidance_param = 0

        if self.rep == "t2m":
            extension = ""
        elif self.rep == "smr":
            extension = "_smr"

        self.mean = torch.from_numpy(np.load(os.path.join(self.path["dataset_dir"], 'Mean{}.npy'.format(extension)))).to(self.device)
        self.std = torch.from_numpy(np.load(os.path.join(self.path["dataset_dir"], 'Std{}.npy'.format(extension)))).to(self.device)
        
        if not args.quantization:
            print(f"Loading checkpoints from...")
            self.model, self.diffusion = create_model_and_diffusion(args, args.control_signal, self.path)
            state_dict = torch.load(self.args.model_path, map_location='cpu')
            if mode == "mdm":
                load_model_wo_clip(self.model, state_dict)
            else:
                load_model_wo_clip(self.model, state_dict["ema"])
            if self.args.guidance_param != 1 and not self.args.unconstrained:
                self.model = ClassifierFreeSampleModel(self.model)   # wrapping model with the classifier-free sampler
            self.model.to(self.device)
            self.model.eval()  # disable random masking
        else:
            self.model, self.diffusion = create_trt_model(args, mode, args.control_signal, self.path, self.device)

    def predict(self,prompt, num_repetitions=1, path=None):
        double_split = prompt.split("|")
        if len(double_split) > 1:
            print("sample mode - double_take long motion")
            sample, step_sizes = double_take(prompt, path, num_repetitions, self.model, self.diffusion, self.handshake_size, 
                                    self.blend_size, self.num_frames, self.args.guidance_param, self.length, self.device)
            
            sample = sample.permute(0, 2, 3, 1).float() 
            sample = sample * self.std + self.mean   
            if self.rep == "t2m":
                sample = recover_from_ric(sample, 22)    
                sample = sample.view(-1, *sample.shape[2:]).permute(0, 2, 3, 1) 
            elif self.rep == "smr":
                sample = sample.permute(0, 2, 3, 1)       
        else:
            try:
                nframes = int(self.length)
            except:
                nframes = self.num_frames
            model_kwargs = {'y':{'text': str(prompt), 'lengths':nframes}}
            if self.args.guidance_param != 1:
                model_kwargs['y']['scale'] = torch.ones(num_repetitions, device=self.device) * self.args.guidance_param

            sample_fn = self.diffusion.p_sample_loop
            sample = sample_fn(
                self.model,
                (num_repetitions, self.model.njoints, self.model.nfeats, nframes),
                clip_denoised=False,
                model_kwargs=model_kwargs,
                skip_timesteps=self.skip_steps,  # 0 is the default value - i.e. don't skip any step
                init_image=None,
                progress=True,
                dump_steps=None,
                noise=None,
                const_noise=False
            )
            sample = sample["output"]
            sample = sample.permute(0, 2, 3, 1).float() 
            sample = sample * self.std + self.mean            

            if self.rep == "t2m":
                sample = recover_from_ric(sample, 22)    
                sample = sample.view(-1, *sample.shape[2:]).permute(0, 2, 3, 1) 
            elif self.rep == "smr":
                sample = sample.permute(0, 2, 3, 1) 

        all_motions = sample.permute(0, 3, 1, 2)
        return all_motions