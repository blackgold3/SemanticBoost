import torch
from torch import nn
import torch.nn.functional as F
import copy
from motion.model.layer_norm_fp16 import LayerNorm, RMSNorm
import numpy as np
import math

def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device)  # type: ignore
    freqs = torch.outer(t, freqs).float()  # type: ignore
    ones = torch.ones_like(freqs)

    real = ones * torch.cos(freqs)
    comp = ones * torch.sin(freqs)

    freqs_cis = torch.stack([real, comp], dim=2)
    return freqs_cis

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
                 activation = F.relu, layer_norm_eps = 1e-5, device=None, dtype=None, max_seq_len=196, position_type="static", word_tokens=False, norm_type="rmsnorm"):
        factory_kwargs = {'device': device, 'dtype': dtype, "bias":False}
        super().__init__()
        if norm_type.lower() == "rmsnorm":
            Norm = RMSNorm
        elif norm_type.lower() == "layer":
            Norm = LayerNorm

        self.self_attn = Attention(d_model, nhead, bias=False)

        if position_type.lower() == "rope":
            self.freqs_cis = precompute_freqs_cis(d_model // nhead, max_seq_len * 2)
            self.freqs_cis = nn.parameter.Parameter(self.freqs_cis, requires_grad=False)
        else:
            self.freqs_cis = None

        if word_tokens:
            self.cross_attn = Attention(d_model, nhead, bias=False)

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
            src_mask = None):
        x = src
        x = x + self._sa_block(self.norm1(x), src_mask)   
        if self.word_tokens:
            x = x + self._csa_block(self.norm3(x), word_tokens)   
        x = x + self.dropout2(self.ffn(self.norm2(x)))
        return x

    # encoder block
    def _sa_block(self, x, attn_mask):
        x = self.self_attn(x, x, x, self.freqs_cis, mask=attn_mask)

        return self.dropout1(x)

    # multihead attention block
    def _csa_block(self, x, mem, attn_mask=None):
        x = self.cross_attn(x, mem, mem, self.freqs_cis, mask=None)  
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
            src_mask=None):
        output = src
        for mod in self.layers:
            output = mod(output, word_tokens=word_tokens, src_mask=src_mask)
        return output


'''
llama2 model
'''

def reshape_for_broadcast(freqs_cis, x):
    freqs_cis = freqs_cis[:x.shape[0], :]
    ndim = x.ndim
    assert 0 <= 1 < ndim
    assert freqs_cis.shape == (x.shape[0], x.shape[-2], x.shape[-1])
    shape = [d if i == 0 or i == ndim - 2 or i == ndim -1 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(*shape)


def apply_rotary_emb(xq, xk, freqs_cis):
    xq_ = xq.float().reshape(*xq.shape[:-1], -1, 2)
    xk_ = xk.float().reshape(*xk.shape[:-1], -1, 2)

    freqs_cisq = reshape_for_broadcast(freqs_cis, xq_)
    freqs_cisk = reshape_for_broadcast(freqs_cis, xk_)
    
    xq_out = torch.zeros_like(xq_)
    xq_out[..., 0] = xq_[..., 0] * freqs_cisq[..., 0] - xq_[..., 1] * freqs_cisq[..., 1]
    xq_out[..., 1] = xq_[..., 1] * freqs_cisq[..., 0] + xq_[..., 0] * freqs_cisq[..., 1]
    xq_out = xq_out.flatten(3)

    xk_out = torch.zeros_like(xk_)
    xk_out[..., 0] = xk_[..., 0] * freqs_cisk[..., 0] - xk_[..., 1] * freqs_cisk[..., 1]
    xk_out[..., 1] = xk_[..., 1] * freqs_cisk[..., 0] + xk_[..., 0] * freqs_cisk[..., 1]
    xk_out = xk_out.flatten(3)

    return xq_out.type_as(xq), xk_out.type_as(xk)

class Attention(nn.Module):
    """Multi-head attention module."""
    def __init__(self, d_model, nhead, bias=False):
        super().__init__()
        self.nheads = nhead
        self.head_dim = d_model // self.nheads
        self.wq = nn.Linear(d_model, self.nheads * self.head_dim, bias=bias)
        self.wk = nn.Linear(d_model, self.nheads * self.head_dim, bias=bias)
        self.wv = nn.Linear(d_model, self.nheads * self.head_dim, bias=bias)
        self.wo = nn.Linear(self.nheads * self.head_dim, d_model, bias=bias)
    
    def forward(self, q, k, v, freqs_cis=None, mask=None):
        seqlen, bs, latent_dim = q.shape
        cond_len, bs, latent_dim = k.shape

        xq = self.wq(q)
        xk = self.wk(k)
        xv = self.wv(v)

        xq = xq.view(seqlen, bs, self.nheads, self.head_dim)
        xk = xk.view(cond_len, bs, self.nheads, self.head_dim)
        xv = xv.view(cond_len, bs, self.nheads, self.head_dim)

        if freqs_cis is not None:
            xq, xk = apply_rotary_emb(xq, xk, freqs_cis=freqs_cis)
            
        xq = xq.permute(1, 2, 0, 3)
        xk = xk.permute(1, 2, 0, 3)
        xv = xv.permute(1, 2, 0, 3) # (bs, nheads, seqlen, head_dim) @ (bs, nheads, head_dim, cond_len) -> [bs, nheads, seqlen, cond_len] @ (bs, nheads, cond_len, head_dim) -> [bs, nheads, seqlen, head_dim]

        scores = torch.matmul(xq, xk.transpose(2, 3)) / math.sqrt(self.head_dim)
        if mask is not None:
            scores = scores + mask  # (bs, n_local_heads, seqlen, cache_len + seqlen)
        scores = F.softmax(scores.float(), dim=-1).type_as(xq)
        output = torch.matmul(scores, xv)  # (bs, nheads, seqlen, head_dim)
        output = output.permute(2, 0, 1, 3).contiguous().view(seqlen, bs, -1)
        return self.wo(output)
