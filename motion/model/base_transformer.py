import torch
from torch import nn
import torch.nn.functional as F
import copy
from torch.nn import MultiheadAttention
from motion.model.layer_norm_fp16 import LayerNorm, RMSNorm
import numpy as np
import math

class SwiGLU(nn.Module):
    '''
    follow the structure of llama
    '''
    def __init__(self, dim, hidden_dim, multiple_of = 256):
        super().__init__()
        hidden_dim = int(2 * hidden_dim / 3)
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)

        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias= False)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))

def _get_activation_fn(activation: str):
    if activation.lower() == "relu":
        return F.relu
    elif activation.lower() == "gelu":
        return F.gelu

    raise RuntimeError("activation should be relu/gelu, not {}".format(activation))

def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

class RefinedLayer(nn.Module):
    __constants__ = ['batch_first', 'norm_first']

    def __init__(self, d_model, nhead, dim_feedforward = 2048, dropout = 0.1,
                 activation = F.relu, layer_norm_eps = 1e-5, device=None, dtype=None, max_seq_len=196, position_type="static", word_tokens=False, norm_type="rmsnorm", attention_type="torch"):
        factory_kwargs = {'device': device, 'dtype': dtype, "bias":False}
        super().__init__()
        if norm_type.lower() == "rmsnorm":
            Norm = RMSNorm
        elif norm_type.lower() == "layer":
            Norm = LayerNorm

        self.attention_type = attention_type
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=False, **factory_kwargs)

        if word_tokens:
            self.cross_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=False, **factory_kwargs)
            self.norm3 = Norm(d_model, layer_norm_eps)
            self.dropout3 = nn.Dropout(dropout)
        self.word_tokens = word_tokens
        # Implementation of Feedforward model

        self.norm1 = Norm(d_model, layer_norm_eps)
        self.norm2 = Norm(d_model, layer_norm_eps)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        # Legacy string support for activation function.
        if isinstance(activation, str) and activation.lower() != "swiglu":
            activation = _get_activation_fn(activation)
            self.linear1 = nn.Linear(d_model, dim_feedforward, **factory_kwargs)
            self.dropout = nn.Dropout(dropout)
            self.linear2 = nn.Linear(dim_feedforward, d_model, **factory_kwargs)      
            self.ffn = self._ff_block 
        elif activation.lower() == "swiglu":
            self.ffn = SwiGLU(d_model, dim_feedforward)
        
        self.activation = activation

    def forward(
            self,
            src,
            word_tokens = None,
            src_mask = None,
            src_key_padding_mask = None):
        x = src
        x = x + self._sa_block(self.norm1(x), src_mask, src_key_padding_mask)   
        if self.word_tokens:
            x = x + self._csa_block(self.norm3(x), word_tokens)   
        x = x + self.dropout2(self.ffn(self.norm2(x)))
        return x

    # encoder block
    def _sa_block(self, x, attn_mask, key_padding_mask):
        x = self.self_attn(x, x, x,
                        attn_mask=attn_mask,
                        key_padding_mask=key_padding_mask,
                        need_weights=False)[0]


        return self.dropout1(x)

    # multihead attention block
    def _csa_block(self, x, mem, attn_mask=None, key_padding_mask=None):
        x = self.cross_attn(x, mem, mem,
                                attn_mask=attn_mask,
                                key_padding_mask=key_padding_mask,
                                need_weights=False)[0]


        return self.dropout3(x)

    # feed forward block
    def _ff_block(self, x):
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return x

class Refined_Transformer(nn.Module):
    def __init__(self, refined_layer, num_layers):
        super().__init__()
        self.layers = _get_clones(refined_layer, num_layers)
        self.num_layers = num_layers

    def forward(
            self,
            src,
            word_tokens=None,
            src_mask=None,
            src_key_padding_mask = None):
        output = src
        src_key_padding_mask_for_layers = src_key_padding_mask
        for mod in self.layers:
            output = mod(output, word_tokens=word_tokens, src_mask=src_mask, src_key_padding_mask=src_key_padding_mask_for_layers)
        return output
