from torch import nn
import torch
import torch.nn.functional as F

class LayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps = 1e-5, elementwise_affine = True,
                 device=None, dtype=None):
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        if isinstance(normalized_shape, int):
            normalized_shape = [normalized_shape]
        self.normalized_shape = normalized_shape  # type: ignore[arg-type]
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        if self.elementwise_affine:
            self.weight = nn.parameter.Parameter(torch.ones(self.normalized_shape, **factory_kwargs))
            self.bias = nn.parameter.Parameter(torch.zeros(self.normalized_shape, **factory_kwargs))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)
        
    def forward(self, input):
        orig_type = input.dtype
        ret = F.layer_norm(input.type(torch.float32), self.normalized_shape, self.weight.type(torch.float32), self.bias.type(torch.float32), self.eps)
        return ret.type(orig_type)

class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight