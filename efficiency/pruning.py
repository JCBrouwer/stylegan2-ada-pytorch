from abc import abstractmethod

import numpy as np
import torch
from torch_utils import training_stats

CLOSE_TO_ZERO = 1e-3


class Pruning:
    @abstractmethod
    def forward_hook(mod, inputs, outputs):
        raise NotImplementedError

    @abstractmethod
    def before_minibatch(self):
        pass

    @abstractmethod
    def after_minibatch(self):
        pass

    @abstractmethod
    def before_backward(self):
        pass


class L1(Pruning):
    """
    Based on 'Learning Efficient Convolutional Networks through Network Slimming': https://arxiv.org/abs/1708.06519
    """

    def __init__(self, parent, lambda_l1=0.005, conv_only=False):
        self.parent = parent
        self.lambda_l1 = lambda_l1

        self.masks = []
        for name, mod in list(self.parent.G_mapping.named_modules()) + list(self.parent.G_synthesis.named_modules()):
            if "fc" in name or "affine" in name:
                if not conv_only:
                    mod.register_forward_hook(self.forward_hook)
                    mod.mask = torch.nn.Parameter(torch.ones((1, mod.weight.shape[0]), device=self.parent.device))
                    self.masks.append(mod.mask)
            elif "conv" in name:
                mod.register_forward_hook(self.forward_hook)
                mod.mask = torch.nn.Parameter(torch.ones((1, mod.weight.shape[0], 1, 1), device=self.parent.device))
                self.masks.append(mod.mask)

    def forward_hook(self, mod, inputs, outputs):
        return mod.mask.to(outputs.dtype) * outputs

    def after_minibatch(self):
        l1_penalty = 0
        for mask in self.masks:
            l1_penalty += torch.norm(mask, p=1)
        l1_penalty.mul(self.lambda_l1).backward()

        sparsities = [(p <= CLOSE_TO_ZERO).sum().detach().cpu() / p.numel() for p in self.masks]
        print("sparsity", np.mean(sparsities), "\t l1", l1_penalty.item())
        training_stats.report("Pruning/l1", l1_penalty)
        training_stats.report("Pruning/sparsity", sparsities)


def soft_threshold(w, thresh):
    with torch.no_grad():
        temp = torch.abs(w) - thresh
        return torch.sign(w) * torch.nn.functional.relu(temp)


class Proximal(Pruning):
    """
    Based on 'GAN Slimming: All-in-One GAN Compression by A Unified Optimization Framework': https://github.com/VITA-Group/GAN-Slimming
    """

    def __init__(
        self, parent, rho=0.01, l1_lr=0.1, momentum=0.5, anneal_epochs=20, batches_per_epoch=256, conv_only=False
    ):
        self.parent = parent
        self.rho = rho

        self.masks = []
        for name, mod in list(self.parent.G_mapping.named_modules()) + list(self.parent.G_synthesis.named_modules()):
            if "fc" in name or "affine" in name:
                if not conv_only:
                    mod.mask = torch.ones((1, mod.weight.shape[0]), device=self.parent.device)
                    self.masks.append(mod.mask)
                    mod.register_forward_hook(self.forward_hook)
            elif "conv" in name:
                mod.mask = torch.ones((1, mod.weight.shape[0], 1, 1), device=self.parent.device)
                self.masks.append(mod.mask)
                mod.register_forward_hook(self.forward_hook)

        self.mask_optimizer = torch.optim.SGD(
            [torch.nn.Parameter(mask) for mask in self.masks], lr=l1_lr, momentum=momentum
        )
        self.mask_schedule = torch.optim.lr_scheduler.CosineAnnealingLR(self.mask_optimizer, T_max=anneal_epochs)
        self.batch = 0
        self.batches_per_epoch = batches_per_epoch

    def forward_hook(self, mod, inputs, outputs):
        return mod.mask.to(outputs.dtype) * outputs

    def before_minibatch(self):
        self.mask_optimizer.zero_grad()

    def after_minibatch(self):
        self.mask_optimizer.step()

        lr = self.mask_schedule.get_last_lr()[0]
        sparsities = []
        for mask in self.masks:
            mask.data = soft_threshold(mask.data, thresh=self.rho * lr)
            sparsities.append((mask == 0.0).sum().detach().cpu() / mask.numel())

        training_stats.report("Pruning/thresh", self.rho * lr)
        training_stats.report("Pruning/sparsity", sparsities)
        print("sparsity", np.mean(sparsities), "\t thresh", self.rho * lr)

        if self.batch != 0 and self.batch % self.batches_per_epoch == 0:
            self.mask_schedule.step()
        self.batch += 1
