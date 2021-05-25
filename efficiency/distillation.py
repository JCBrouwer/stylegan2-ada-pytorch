import os
from abc import abstractmethod
from pathlib import Path

import lpips
import numpy as np
import torch
from skimage import io, transform
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
    def __init__(self, parent, path, batch_size):
        assert path != "", "Must specify --teacher-path when distilling!"
        self.parent = parent
        self.z_dim = parent.G_mapping.z_dim
        self.device = parent.device
        dataset = DistillationDataset(path)
        sampler = misc.InfiniteSampler(dataset)
        self.dataloader = iter(torch.utils.data.DataLoader(dataset=dataset, sampler=sampler, batch_size=batch_size))

    def get_latents(self):
        self.teacher_imgs, seeds = next(self.dataloader)
        self.teacher_imgs = self.teacher_imgs.to(self.device, non_blocking=True).to(torch.float32) / 127.5 - 1
        latents = torch.tensor([np.random.RandomState(seed).randn(self.z_dim) for seed in seeds])
        return latents.float()

    def loss_backward(self, student_imgs):
        loss = torch.nn.functional.l1_loss(self.teacher_imgs, student_imgs)
        training_stats.report("Distillation/loss", loss)
        loss.backward()


class LPIPS(Basic):
    def __init__(self, parent, path, batch_size, lpips_net="alex"):
        super().__init__(parent, path, batch_size)
        self.perceptual_loss = lpips.LPIPS(net=lpips_net).to(self.device)

    def loss_backward(self, student_imgs):
        loss = torch.nn.functional.l1_loss(self.teacher_imgs, student_imgs)
        perceptual = self.perceptual_loss(self.teacher_imgs, student_imgs).sum()
        training_stats.report("Distillation/loss", loss)
        training_stats.report("Distillation/perceptual", perceptual)
        (loss + perceptual).backward()


class Advanced(Distillation):
    """
    
    """

    def __init__(self):
        return

    def before_backward(self):
        return
