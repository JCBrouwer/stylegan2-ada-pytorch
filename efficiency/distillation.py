import os
from abc import abstractmethod
from pathlib import Path

import lpips
import numpy as np
import torch
from torch_utils import misc, training_stats
from training.dataset import ImageFolderDataset


CLOSE_TO_ZERO = 1e-3


class DistillationDataset(ImageFolderDataset):
    def __getitem__(self, idx):
        raw_idx = self._raw_idx[idx]
        image = self._load_raw_image(raw_idx)
        seed = Path(self._image_fnames[raw_idx]).stem
        return image.copy(), int(seed)


class Distillation:
    @abstractmethod
    def get_latents(self):
        pass

    @abstractmethod
    def loss_backward(self):
        pass


class Basic(Distillation):
    def __init__(self, parent, path, batch_size, lambda_pixel=10):
        assert path != "", "Must specify --teacher-path when distilling!"
        self.parent = parent
        self.z_dim = parent.G_mapping.z_dim
        self.device = parent.device
        dataset = DistillationDataset(path)
        sampler = misc.InfiniteSampler(dataset)
        self.iterloader = iter(torch.utils.data.DataLoader(dataset=dataset, sampler=sampler, batch_size=batch_size))
        self.lambda_pixel = lambda_pixel

    def __getstate__(self):
        return {k: v for k, v in self.__dict__.items() if not k == "iterloader"}  # dataloader iterator can't be pickled

    def get_latents(self):
        self.teacher_imgs, seeds = next(self.iterloader)
        self.teacher_imgs = self.teacher_imgs.to(self.device, non_blocking=True).to(torch.float32) / 127.5 - 1
        latents = torch.tensor([np.random.RandomState(seed).randn(self.z_dim) for seed in seeds])
        return latents.float()

    def loss_backward(self, student_imgs, retain_graph=False):
        loss = self.lambda_pixel * torch.nn.functional.l1_loss(self.teacher_imgs, student_imgs)
        training_stats.report("Distillation/loss", loss)
        loss.backward(retain_graph=retain_graph)


class LPIPS(Basic):
    def __init__(self, parent, path, batch_size, lambda_pixel=10, lpips_net="alex"):
        super().__init__(parent, path, batch_size, lambda_pixel)
        self.perceptual_loss = lpips.LPIPS(net=lpips_net).to(self.device)

    def loss_backward(self, student_imgs, retain_graph=False):
        super().loss_backward(student_imgs, retain_graph=True)
        perceptual = self.perceptual_loss(self.teacher_imgs, student_imgs).sum()
        training_stats.report("Distillation/perceptual", perceptual)
        perceptual.backward(retain_graph=retain_graph)


class SelfSupervised(LPIPS):
    def loss_backward(self, student_imgs):
        super().loss_backward(student_imgs, retain_graph=True)
        gen_logits_o = self.parent.run_D(self.teacher_imgs, None, sync=False)
        gen_logits_c = self.parent.run_D(student_imgs, None, sync=False)
        ldo = torch.nn.functional.softplus(gen_logits_o)
        ldc = torch.nn.functional.softplus(gen_logits_c)
        loss = torch.nn.functional.l1_loss(ldo, ldc)
        training_stats.report("Distillation/self-supervised", loss)
        loss.backward()
