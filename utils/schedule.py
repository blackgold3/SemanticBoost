import torch
import math
from torch.optim.lr_scheduler import _LRScheduler

class CustomScheduler(_LRScheduler):
    def __init__(self, optimizer, warmup_epochs, total_epochs, eta_min=0, verbose=False):
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.eta_min = eta_min
        super(CustomScheduler, self).__init__(optimizer, verbose=verbose)

    def get_lr(self):
        if self.last_epoch < self.warmup_epochs:
            # Warmup 阶段，线性增加学习率
            return [base_lr * (self.last_epoch + 1) / self.warmup_epochs for base_lr in self.base_lrs]
        else:
            # 余弦退火阶段
            progress = self.last_epoch / self.total_epochs 
            return [self.eta_min + (base_lr - self.eta_min) * 0.5 * (1 + math.cos(math.pi * progress)) for base_lr in self.base_lrs]