# -*- coding: utf-8 -*-

# import dnnlib
import torch
import numpy as np
from typing import List, Optional
import PIL.Image

import os
import re
from typing import List, Optional

import click

import dnnlib
import time

import sys
import io

from generate import *

from training.training_loop import *
from training.networks import *
from torch_utils.misc import *
from training.legacy import *


# def generate_images(
#     ctx: click.Context,
#     network_pkl: str,
#     seeds: Optional[List[int]],
#     truncation_psi: float,
#     noise_mode: str,
#     outdir: str,
#     class_idx: Optional[int],
#     projected_w: Optional[str],
# ):

# class Generator(torch.nn.Module):
#     def __init__(
#         self,
#         z_dim,  # Input latent (Z) dimensionality.
#         c_dim,  # Conditioning label (C) dimensionality.
#         w_dim,  # Intermediate latent (W) dimensionality.
#         img_resolution,  # Output resolution.
#         img_channels,  # Number of output color channels.
#         mapping_kwargs={},  # Arguments for MappingNetwork.
#         synthesis_kwargs={},  # Arguments for SynthesisNetwork.
#     ):


def speedtest(
    seeds: Optional[List[int]],
    truncation_psi: float,
    noise_mode: str,
    outdir: str,
    z_dim: int,
    c_dim: int,
    w_dim: int,
    img_resolution: int,
    img_channels: int,
    synthesis_kwargs: dict,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    G = Generator(
        z_dim=z_dim,
        c_dim=c_dim,
        w_dim=w_dim,
        img_resolution=img_resolution,
        img_channels=img_channels,
        synthesis_kwargs=synthesis_kwargs,
    )
    G = G.to(device)

    old_stdout = sys.stdout  # Memorize the default stdout stream
    sys.stdout = buffer = io.StringIO()
    z = torch.from_numpy(np.random.RandomState(0).randn(1, G.z_dim)).to(device)
    summary = print_module_summary(G, [z, c_dim])
    sys.stdout = old_stdout
    # sys.stdout = sys.__stdout__
    whatWasPrinted = buffer.getvalue()

    label = torch.zeros([1, G.c_dim], device=device)

    print(G.z_dim, G.c_dim, G.w_dim, G.img_resolution, G.img_channels)

    for seed_idx, seed in enumerate(seeds):
        print("Generating image for seed %d (%d/%d) ..." % (seed, seed_idx, len(seeds)))
        z = torch.from_numpy(np.random.RandomState(seed).randn(1, G.z_dim)).to(device)
        img = G(z, label, truncation_psi=truncation_psi, noise_mode=noise_mode)
        img = (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
        PIL.Image.fromarray(img[0].cpu().numpy(), "RGB").save(f"{outdir}/seed{seed:04d}.png")

    return whatWasPrinted


if __name__ == "__main__":
    torch.cuda.empty_cache()
    seeds = [i for i in range(10)]
    trunc = 1
    outdir = "out"
    noise_mode = "const"
    z_dim = 512
    c_dim = 0
    w_dim = 512
    img_resolution = 1024
    img_channels = 3
    channel_base = 32768
    channel_max = 512
    synthesis_kwargs = {"channel_base": channel_base, "channel_max": channel_max}

    # Put into a loop, changing some of the synthesis_kwargs per round
    # Create variable/dict to store parameter:average time
    averagetime = {}
    fullsummary = {}
    n = 2
    channel_base_arr = np.linspace(32768 / n, 32768, n, dtype=int)
    channel_max_arr = np.linspace(512 / n, 512, n, dtype=int)
    # Start loop
    for i in range(n):
        # Change channel_base and channel_max
        channel_base = channel_base_arr[i]
        channel_max = channel_max_arr[n - 1]
        # Start timer
        timestart = time.time()
        summary = speedtest(
            seeds=seeds,
            truncation_psi=trunc,
            noise_mode=noise_mode,
            outdir=outdir,
            z_dim=z_dim,
            c_dim=c_dim,
            w_dim=w_dim,
            img_resolution=img_resolution,
            img_channels=img_channels,
            synthesis_kwargs=synthesis_kwargs,
        )  # pylint: disable=no-value-for-parameter
        # End timer
        timeend = time.time()
        torch.cuda.empty_cache()

        # Store time and kwargs in dict
        Parameters = int(summary[-52:-44])
        Buffers = int(summary[-40:-33])
        timeavg = (timeend - timestart) / len(seeds)
        averagetime[str(i)] = (Parameters, timeavg)
        fullsummary[str(i)] = summary
        # End loop
    torch.cuda.empty_cache()
    # Print dict

    print("test")
    # print(fullsummary)
    print(averagetime)
    # Parameters = int(summary[-52:-44])
    # Buffers = int(summary[-40:-33])

