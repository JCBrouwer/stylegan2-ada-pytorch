import numpy as np
import torch
from torch_utils import training_stats
from torch_utils.ops import conv2d_gradfix
from training.loss import StyleGAN2Loss

from .distillation import *
from .pruning import *
from .quantization import *


class SlimmingLoss(StyleGAN2Loss):
    def __init__(
        self,
        device,
        G_mapping,
        G_synthesis,
        D,
        batch_size,
        lambda_l1=5e-3,
        pruning="mask",
        distill="lpips",
        teacher_path="",
        lpips_net="vgg",
        quantization="linear",
        input_signed=False,
        nbits=8,
        input_max=4,
        quantize_mapping=True,
        prune_torgb=True,
        lambda_pixel=10,
        **kwargs
    ):
        super().__init__(device, G_mapping, G_synthesis, D, **kwargs)

        # quantization must be initialized before pruning! (otherwise the L1Weight will try to prune the wrong weights)
        if quantization == "linear":
            self.quantizer = Linear(
                self, input_signed=input_signed, quantize_mapping=quantize_mapping, nbits=nbits, input_max=input_max
            )
        elif quantization == "qgan":
            self.quantizer = QGAN(
                self, input_signed=True, quantize_mapping=quantize_mapping, nbits=nbits, input_max=input_max
            )
        elif quantization == "fp16":
            self.G_mapping = force_fp16(self.G_mapping)
            self.G_synthesis = force_fp16(self.G_synthesis)
            misc.print_module_summary(self.G_mapping, [torch.randn(8, 512), None])
            misc.print_module_summary(self.Synthesis, [torch.randn(8, 18, 512)])
        else:
            self.quantizer = Quantization()  # does nothing

        if pruning == "prox":
            self.pruner = Proximal(self)  # doesn't really work :(
        elif pruning == "mask":
            self.pruner = L1Mask(self, lambda_l1, prune_torgb)
        elif "l1" in pruning:
            dims = ((0, 1) if "in" in pruning else 0) if "out" in pruning else 1  # in -> 1, out -> 0, in-out -> (0, 1)
            self.pruner = L1Weight(self, lambda_l1, dims=dims)
        else:
            self.pruner = Pruning()  # does nothing

        if distill == "basic":
            self.distiller = Basic(self, teacher_path, batch_size, lambda_pixel=lambda_pixel)
        elif distill == "lpips":
            self.distiller = LPIPS(self, teacher_path, batch_size, lambda_pixel=lambda_pixel, lpips_net=lpips_net)
        elif distill == "self-supervised":
            self.distiller = SelfSupervised(
                self, teacher_path, batch_size, lambda_pixel=lambda_pixel, lpips_net=lpips_net
            )
        else:
            self.distiller = Distillation()  # does nothing

    def accumulate_gradients(self, phase, real_img, real_c, gen_z, gen_c, sync, gain):
        assert phase in ["Gmain", "Greg", "Gboth"]
        do_Gmain = phase in ["Gmain", "Gboth"]
        do_Gpl = (phase in ["Greg", "Gboth"]) and (self.pl_weight != 0)

        #
        # Generator losses
        #

        if do_Gmain:
            gen_z = self.distiller.get_latents().to(self.device)
            gen_img, _ = self.run_G(gen_z, None, sync=False)
            self.pruner.loss_backward()
            self.distiller.loss_backward(gen_img)

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
