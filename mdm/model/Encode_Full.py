from torch import nn
import torch
import torch.nn.functional as F
from mdm.model.layer_norm_fp16 import RMSNorm, LayerNorm

class ResConv1DBlock(nn.Module):
    def __init__(self, n_in, n_state, bias, norm_type, activate_type):
        super().__init__()

        if activate_type.lower() == "silu":
            activate = nn.SiLU()
        elif activate_type.lower() == "relu":
            activate = nn.ReLU()
        elif activate_type.lower() == "gelu":
            activate = nn.GELU()
        elif activate_type.lower() == "mish":
            activate = nn.Mish()
        
        if norm_type.lower() == "rmsnorm":
            norm = RMSNorm
        elif norm_type.lower() == "layernorm":
            norm = LayerNorm

        self.norm1 = norm(n_state)
        self.norm2 = norm(n_in)
        self.relu1 = activate
        self.relu2 = activate
        self.conv1 = nn.Conv1d(n_in, n_state, 3, 1, 1, bias=bias)
        self.conv2 = nn.Conv1d(n_state, n_in, 1, 1, 0, bias=bias)     

    def forward(self, x):
        x_orig = x
        x = self.conv1(x)
        x = self.norm1(x.transpose(-2, -1))
        x = self.relu1(x.transpose(-2, -1))

        x = self.conv2(x)
        x = self.norm2(x.transpose(-2, -1))
        x = self.relu2(x.transpose(-2, -1))
        
        x = x + x_orig
        return x

class Encoder_Block(nn.Module):
    def __init__(self, begin_channel=263, latent_dim=512, num_layers=6, TN=1, bias=True, norm_type="layernorm", activate_type="relu"):
        super(Encoder_Block, self).__init__()
        self.layers = []

        begin_channel = begin_channel
        target_channel = latent_dim

        if activate_type.lower() == "silu":
            activate = nn.SiLU()
        elif activate_type.lower() == "relu":
            activate = nn.ReLU()
        elif activate_type.lower() == "gelu":
            activate = nn.GELU()
        elif activate_type.lower() == "mish":
            activate = nn.Mish()
        
        self.layers.append(nn.Conv1d(begin_channel, target_channel, 3, 2, 1, bias=bias))
        self.layers.append(activate)

        for _ in range(num_layers):      ### 196 -> 98 -> 49 -> 24 -> 12 -> 6 -> 3
            self.layers.append(nn.Conv1d(target_channel, target_channel, 3, 2, 1, bias=bias))
            self.layers.append(activate)
            self.layers.append(ResConv1DBlock(target_channel, target_channel, bias, norm_type, activate_type))

        self.layers = nn.Sequential(*self.layers)
        self.maxpool = nn.AdaptiveMaxPool1d(TN)

    def forward(self, x): 
        bs, njoints, nfeats, nframes = x.shape
        reshaped_x = x.reshape(bs, njoints * nfeats, nframes)      ### [bs, 263, seq]

        res1 = self.layers(reshaped_x)      #### [bs, 512, 1]
        res2 = self.maxpool(res1)

        res3 = res2.permute(2, 0, 1)
        return res3