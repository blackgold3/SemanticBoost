import numpy as np
import torch
import torch.nn as nn
from copy import deepcopy

# A wrapper model for Classifier-free guidance **SAMPLING** only
# https://arxiv.org/abs/2207.12598
class ClassifierFreeSampleModel(nn.Module):

    def __init__(self, model):
        super().__init__()
        self.model = model  # model is the actual model to run

        # assert self.model.cond_mask_prob > 0, 'Cannot run a guided diffusion on a model that has not been trained with no conditions'

        # pointers to inner model
        self.njoints = self.model.njoints
        self.nfeats = self.model.nfeats
        self.cond_mode = self.model.cond_mode

    def forward(self, x, timesteps, y=None):
        cond_mode = self.model.cond_mode
        assert cond_mode in ['text', 'action', "motion", "text-motion"]
        y_uncond = deepcopy(y)
        y_uncond['uncond'] = True

        out = self.model(x, timesteps, y)                   ###### 全部条件生成

        if "predict_length" in out.keys():
            y_uncond["predict_mask"] = out["predict_length"]

        out_uncond = self.model(x, timesteps, y_uncond)     ####### 全部无条件

        output = {}

        y['scale'] = y['scale'].to(out_uncond["output"].device)

        output["output"] = out_uncond["output"] + (y['scale'].view(-1, 1, 1, 1) * (out["output"] - out_uncond["output"]))
        
        return output       ##### 这里并不是生成 \epsilon，而是特征

