from torch import nn
import torch
import torch.nn.functional as F
from model.layer_norm_fp16 import RMSNorm, LayerNorm

class ResConv1DBlock(nn.Module):
    def __init__(self, n_in, n_state, norm_type, activate_type):
        super().__init__()

        if activate_type.lower() == "silu":
            activate = nn.SiLU()
        elif activate_type.lower() == "relu":
            activate = nn.ReLU()
        elif activate_type.lower() == "gelu":
            activate = nn.GELU()
        
        if norm_type.lower() == "rmsnorm":
            norm = RMSNorm
        elif norm_type.lower() == "layernorm":
            norm = LayerNorm

        self.norm1 = norm(n_state)
        self.norm2 = norm(n_in)
        self.relu1 = activate
        self.relu2 = activate

        self.conv1 = nn.Conv1d(n_in, n_state, 3, 1, 1)
        self.conv2 = nn.Conv1d(n_state, n_in, 1, 1, 0)     

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
    def __init__(self, latent_dim=512, num_layers=6, norm_type="layernorm", activate_type="relu"):
        super(Encoder_Block, self).__init__()
        self.layers = []

        if activate_type.lower() == "silu":
            activate = nn.SiLU()
        elif activate_type.lower() == "relu":
            activate = nn.ReLU()
        elif activate_type.lower() == "gelu":
            activate = nn.GELU()
        
        for _ in range(num_layers):      ### 196 -> 98 -> 49 -> 24 -> 12 -> 6
            self.layers.append(nn.Conv1d(latent_dim, latent_dim, 3, 2, 1))
            self.layers.append(activate)
            self.layers.append(ResConv1DBlock(latent_dim, latent_dim, norm_type, activate_type))

        self.layers = nn.Sequential(*self.layers)
        self.linear = nn.Sequential(nn.Linear(latent_dim * 2, latent_dim), activate)
        self.maxpool = nn.AdaptiveMaxPool1d(1)

    def forward(self, x, emb): 
        '''
        x: seqlen, bs, latent_dim
        emb: 1, bs, latent_dim
        '''
        seqlen, bs, latent_dim = x.shape
        x = x.permute(1, 0, 2)  #### [bs, seqlen, 512]
        emb = emb.permute(1, 0, 2)  #### [bs, 1, 512]
        emb = emb.repeat(1, seqlen, 1)
        concat = torch.cat([x, emb], dim=2)
        res = self.linear(concat)   ### [bs, seqlen, 512]
        res = res.permute(0, 2, 1)
        res = self.layers(res)
        res = self.maxpool(res)     #### [bs, 512, 1]
        res = res.permute(2, 0, 1)
        return res