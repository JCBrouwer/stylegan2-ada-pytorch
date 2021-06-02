from time import time

import numpy as np
import torch
import torch.fx
from torch_utils import training_stats
from torch_utils.ops import conv2d_gradfix
from training.loss import StyleGAN2Loss

from .distillation import *
from .pruning import *
from .quantization import *

torch.fx.wrap("len")


def save_torchscript_model(model, model_dir, model_filename):
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    model_filepath = os.path.join(model_dir, model_filename)
    torch.jit.save(torch.jit.script(model), model_filepath)


def load_torchscript_model(model_filepath, device):
    model = torch.jit.load(model_filepath, map_location=device)
    return model


def measure_inference_latency(G_map, G_synth, device, input_size=(16, 512), num_samples=100, num_warmups=10):

    G_map.to(device)
    G_map.eval()
    G_synth.to(device)
    G_synth.eval()

    x = torch.randn(size=input_size).to(device)

    with torch.no_grad():
        for _ in range(num_warmups):
            ws = G_map(x, None)
            _ = G_synth(ws, force_fp32=device == "cpu")
    torch.cuda.synchronize()

    with torch.no_grad():
        start_time = time()
        for _ in range(num_samples):
            ws = G_map(x, None)
            _ = G_synth(ws, force_fp32=device == "cpu")
            torch.cuda.synchronize()
        end_time = time()
    elapsed_time = end_time - start_time
    elapsed_time_ave = elapsed_time / num_samples

    print(elapsed_time_ave)


class SlimmingLoss(StyleGAN2Loss):
    def __init__(
        self,
        device,
        G_mapping,
        G_synthesis,
        D,
        batch_size,
        lambda_l1=0.0001,
        pruning="l1-in-out",
        distill="lpips",
        teacher_path="",
        lpips_net="vgg",
        quantization="qgan",
        input_signed=False,
        nbits=8,
        input_max=4,
        quantize_mapping=True,
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
        elif quantization == "torch":
            self.G_mapping = torch_quantize(self.G_mapping)
            self.G_synthesis = torch_quantize(self.G_synthesis)
        else:
            self.quantizer = Quantization()  # does nothing

        if pruning == "prox":
            self.pruner = Proximal(self)  # doesn't really work :(
        elif pruning == "mask":
            self.pruner = L1Mask(self, lambda_l1)
        elif "l1" in pruning:
            dims = ((0, 1) if "in" in pruning else 0) if "out" in pruning else 1  # in -> 1, out -> 0, in-out -> (0, 1)
            self.pruner = L1Weight(self, lambda_l1, dims=dims)
        else:
            self.pruner = Pruning()  # does nothing

        if distill == "basic":
            self.distiller = Basic(self, teacher_path, batch_size)
        elif distill == "lpips":
            self.distiller = LPIPS(self, teacher_path, batch_size, lpips_net=lpips_net)
        else:
            self.distiller = Distillation()  # does nothing

    def accumulate_gradients(self, phase, real_img, real_c, gen_z, gen_c, sync, gain):
        assert phase in ["Gmain", "Greg", "Gboth", "Dmain", "Dreg", "Dboth"]
        do_Gmain = phase in ["Gmain", "Gboth"]
        do_Dmain = phase in ["Dmain", "Dboth"]
        do_Gpl = (phase in ["Greg", "Gboth"]) and (self.pl_weight != 0)
        do_Dr1 = (phase in ["Dreg", "Dboth"]) and (self.r1_gamma != 0)

        #
        # Generator losses
        #

        # if do_Gmain:
        for _ in range(32):
            gen_z = self.distiller.get_latents().to(self.device)
            self.pruner.before_minibatch()
            gen_img, _ = self.run_G(gen_z, None, sync=False)
            self.pruner.after_minibatch()
            self.distiller.loss_backward(gen_img)

            # gen_img, _gen_ws = self.run_G(gen_z, gen_c, sync=(sync and not do_Gpl))  # May get synced by Gpl.
            # gen_logits = self.run_D(gen_img, gen_c, sync=False)
            # training_stats.report("Loss/scores/fake", gen_logits)
            # training_stats.report("Loss/signs/fake", gen_logits.sign())
            # loss_Gmain = torch.nn.functional.softplus(-gen_logits)  # -log(sigmoid(gen_logits))
            # training_stats.report("Loss/G/loss", loss_Gmain)
            # loss = loss_Gmain.mean().mul(gain)
            # self.pruner.before_backward()
            # loss.backward()

        qGmap = torch.quantization.convert(copy.deepcopy(self.G_mapping).to("cpu"), inplace=True)
        qGsynth = torch.quantization.convert(copy.deepcopy(self.G_synthesis).to("cpu"), inplace=True)

        # Print quantized model.
        print(qGmap)
        print(qGsynth)

        measure_inference_latency(self.G_mapping, self.G_synthesis, "cpu")
        measure_inference_latency(qGmap, qGsynth, "cpu")

        measure_inference_latency(self.G_mapping, self.G_synthesis, "cuda")
        measure_inference_latency(qGmap, qGsynth, "cuda")

        exit(0)

        return  # only use pruning and distillation for now

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
