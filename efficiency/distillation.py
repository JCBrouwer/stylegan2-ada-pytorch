import os
from abc import abstractmethod
from pathlib import Path

import numpy as np
import torch
from skimage import io, transform
from torch_utils import training_stats, misc
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
        self.parent = parent
        self.z_dim = parent.G_mapping.z_dim
        dataset = DistillationDataset(path)
        sampler = misc.InfiniteSampler(dataset)
        self.dataloader = iter(torch.utils.data.DataLoader(dataset=dataset, sampler=sampler, batch_size=batch_size))

    def get_latents(self):
        self.teacher_imgs, seeds = next(self.dataloader)
        self.teacher_imgs = self.teacher_imgs.to(self.parent.device, non_blocking=True).to(torch.float32) / 127.5 - 1
        latents = torch.tensor([np.random.RandomState(seed).randn(self.z_dim) for seed in seeds])
        return latents.float()

    def loss_backward(self, student_imgs):
        loss = torch.nn.functional.l1_loss(self.teacher_imgs, student_imgs)
        training_stats.report("Distillation/loss", loss)
        loss.backward()


class Advanced(Distillation):
    """
    
    """

    def __init__(self):
        return

    def before_backward(self):
        return
