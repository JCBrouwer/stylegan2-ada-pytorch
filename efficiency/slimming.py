import numpy as np
import torch
from torch_utils import training_stats
from torch_utils.ops import conv2d_gradfix
from training.loss import StyleGAN2Loss

from .pruning import *


class SlimmingLoss(StyleGAN2Loss):
    def __init__(self, device, G_mapping, G_synthesis, D, pruning="l1-out", **kwargs):
        lambda_l1 = kwargs["lambda_l1"]
        del kwargs["lambda_l1"]
        super().__init__(device, G_mapping, G_synthesis, D, **kwargs)
        if pruning == "prox":
            self.pruner = Proximal(self)
        elif pruning == "mask":
            self.pruner = L1Mask(self, lambda_l1)
        elif "l1" in pruning:
            # in -> 1, out -> 0, in-out -> (0, 1)
            dims = ((0, 1) if "in" in pruning else 0) if "out" in pruning else 1
            self.pruner = L1Weight(self, lambda_l1, dims=dims)

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
