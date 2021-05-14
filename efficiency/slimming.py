import copy
from abc import abstractmethod

import numpy as np
import torch
from torch_utils import misc, training_stats
from torch_utils.ops import conv2d_gradfix
from training.loss import StyleGAN2Loss

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

    def __init__(self, parent, lambda_l1=0.0001):
        self.parent = parent
        self.lambda_l1 = lambda_l1

        self.masks = []
        for name, mod in list(self.parent.G_mapping.named_modules()) + list(self.parent.G_synthesis.named_modules()):
            if "fc" in name or "affine" in name:
                pass
                # mod.register_forward_hook(self.forward_hook)
                # mod.mask = torch.nn.Parameter(torch.ones((1, mod.weight.shape[0]), device=self.parent.device))
                # self.masks.append(mod.mask)
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

        training_stats.report("Pruning/l1", l1_penalty)
        sparsities = [(p <= CLOSE_TO_ZERO).sum().detach().cpu() / p.numel() for p in self.masks]
        print("sparsity", np.mean(sparsities), "\t l1", l1_penalty.item())


def soft_threshold(w, thresh):
    with torch.no_grad():
        temp = torch.abs(w) - thresh
        return torch.sign(w) * torch.nn.functional.relu(temp)


class Proximal(Pruning):
    """
    Based on 'GAN Slimming: All-in-One GAN Compression by A Unified Optimization Framework': https://github.com/VITA-Group/GAN-Slimming
    """

    def __init__(self, parent, rho=0.01, l1_lr=0.1, momentum=0.5, anneal_epochs=20, batches_per_epoch=256):
        self.parent = parent
        self.rho = rho

        self.masks = []
        for name, mod in list(self.parent.G_mapping.named_modules()) + list(self.parent.G_synthesis.named_modules()):
            if "fc" in name or "affine" in name:
                pass
                # mod.mask = torch.ones((1, mod.weight.shape[0]), device=self.parent.device)
                # self.masks.append(mod.mask)
                # mod.register_forward_hook(self.forward_hook)
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


class SlimmingLoss(StyleGAN2Loss):
    def __init__(self, device, G_mapping, G_synthesis, D, pruning="prox", **kwargs):
        super().__init__(device, G_mapping, G_synthesis, D, **kwargs)
        if pruning == "prox":
            self.pruner = Proximal(self)
        elif pruning == "l1":
            self.pruner = L1(self)

    def accumulate_gradients(self, phase, real_img, real_c, gen_z, gen_c, sync, gain):
        assert phase in ["Gmain", "Greg", "Gboth", "Dmain", "Dreg", "Dboth"]
        do_Gmain = phase in ["Gmain", "Gboth"]
        do_Dmain = phase in ["Dmain", "Dboth"]
        do_Gpl = (phase in ["Greg", "Gboth"]) and (self.pl_weight != 0)
        do_Dr1 = (phase in ["Dreg", "Dboth"]) and (self.r1_gamma != 0)

        #
        # Generator losses
        #

        if do_Gmain:
            self.pruner.before_minibatch()
            gen_img, _gen_ws = self.run_G(gen_z, gen_c, sync=(sync and not do_Gpl))  # May get synced by Gpl.
            gen_logits = self.run_D(gen_img, gen_c, sync=False)
            training_stats.report("Loss/scores/fake", gen_logits)
            training_stats.report("Loss/signs/fake", gen_logits.sign())
            loss_Gmain = torch.nn.functional.softplus(-gen_logits)  # -log(sigmoid(gen_logits))
            training_stats.report("Loss/G/loss", loss_Gmain)
            loss = loss_Gmain.mean().mul(gain)
            self.pruner.before_backward()
            loss.backward()
            self.pruner.after_minibatch()

        # Path length regularization
        if do_Gpl:
            batch_size = gen_z.shape[0] // self.pl_batch_shrink
            gen_img, gen_ws = self.run_G(gen_z[:batch_size], gen_c[:batch_size], sync=sync)
            pl_noise = torch.randn_like(gen_img) / np.sqrt(gen_img.shape[2] * gen_img.shape[3])
            with conv2d_gradfix.no_weight_gradients():
                pl_grads = torch.autograd.grad(
                    outputs=[(gen_img * pl_noise).sum()], inputs=[gen_ws], create_graph=True, only_inputs=True
                )[0]
            pl_lengths = pl_grads.square().sum(2).mean(1).sqrt()
            pl_mean = self.pl_mean.lerp(pl_lengths.mean(), self.pl_decay)
            self.pl_mean.copy_(pl_mean.detach())
            pl_penalty = (pl_lengths - pl_mean).square()
            training_stats.report("Loss/pl_penalty", pl_penalty)
            loss_Gpl = pl_penalty * self.pl_weight
            training_stats.report("Loss/G/reg", loss_Gpl)
            (gen_img[:, 0, 0, 0] * 0 + loss_Gpl).mean().mul(gain).backward()

        #
        # Discriminator losses
        #

        loss_Dgen = 0
        if do_Dmain:
            gen_img, _gen_ws = self.run_G(gen_z, gen_c, sync=False)
            gen_logits = self.run_D(gen_img, gen_c, sync=False)  # Gets synced by loss_Dreal.
            training_stats.report("Loss/scores/fake", gen_logits)
            training_stats.report("Loss/signs/fake", gen_logits.sign())
            loss_Dgen = torch.nn.functional.softplus(gen_logits)  # -log(1 - sigmoid(gen_logits))
            loss_Dgen.mean().mul(gain).backward()

        if do_Dmain or do_Dr1:
            real_img_tmp = real_img.detach().requires_grad_(do_Dr1)
            real_logits = self.run_D(real_img_tmp, real_c, sync=sync)
            training_stats.report("Loss/scores/real", real_logits)
            training_stats.report("Loss/signs/real", real_logits.sign())

            loss_Dreal = 0
            if do_Dmain:
                loss_Dreal = torch.nn.functional.softplus(-real_logits)  # -log(sigmoid(real_logits))
                training_stats.report("Loss/D/loss", loss_Dgen + loss_Dreal)

            loss_Dr1 = 0
            if do_Dr1:
                with conv2d_gradfix.no_weight_gradients():
                    r1_grads = torch.autograd.grad(
                        outputs=[real_logits.sum()], inputs=[real_img_tmp], create_graph=True, only_inputs=True
                    )[0]
                r1_penalty = r1_grads.square().sum([1, 2, 3])
                loss_Dr1 = r1_penalty * (self.r1_gamma / 2)
                training_stats.report("Loss/r1_penalty", r1_penalty)
                training_stats.report("Loss/D/reg", loss_Dr1)
            (real_logits * 0 + loss_Dreal + loss_Dr1).mean().mul(gain).backward()
