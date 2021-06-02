import os
from abc import abstractmethod
from pathlib import Path

import lpips
import numpy as np
import torch
from torch_utils import misc, training_stats
from training.dataset import ImageFolderDataset

import dnnlib
import pickle

CLOSE_TO_ZERO = 1e-3


class SelfSupervisedDataset(ImageFolderDataset):
    def __getitem__(self, idx):
        raw_idx = self._raw_idx[idx]
        image = self._load_raw_image(raw_idx)
        seed = Path(self._image_fnames[raw_idx]).stem
        return image.copy(), int(seed)


class SelfSupervised:
    @abstractmethod
    def get_latents(self):
        pass

    @abstractmethod
    def loss_backward(self):
        pass


class Direct(SelfSupervised):
    def __init__(self, parent, path, batch_size):
        assert path != "", "Must specify --teacher-path when distilling!"
        self.parent = parent
        self.z_dim = parent.G_mapping.z_dim
        self.device = parent.device
        dataset = SelfSupervisedDataset(path)
        sampler = misc.InfiniteSampler(dataset)
        self.iterloader = iter(torch.utils.data.DataLoader(dataset=dataset, sampler=sampler, batch_size=batch_size))
        
        
        #Load discriminator from original network
        device = torch.device("cuda")
        # "ffhq256": "https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/transfer-learning-source-nets/ffhq-res256-mirror-paper256-noaug.pkl"
        network_pkl = "https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/transfer-learning-source-nets/ffhq-res256-mirror-paper256-noaug.pkl"
        with dnnlib.util.open_url(network_pkl) as f:
            _G, self.Disc, _Gs = pickle.load(f)
        
    def __getstate__(self):
        return {k: v for k, v in self.__dict__.items() if not k == "iterloader"}  # dataloader iterator can't be pickled

#Discriminator
    def run_D(self, img, c, sync):
        if self.augment_pipe is not None: #Not implemented here
            img = self.augment_pipe(img)
        with misc.ddp_sync(self.Disc, sync):
            logits = self.Disc(img, c)
        return logits
    
    def get_latents(self):
        self.teacher_imgs, seeds = next(self.iterloader)
        self.teacher_imgs = self.teacher_imgs.to(self.device, non_blocking=True).to(torch.float32) / 127.5 - 1
        latents = torch.tensor([np.random.RandomState(seed).randn(self.z_dim) for seed in seeds])
        return latents.float()

    def distance(self, ldo, ldc): #literal  difference, should possibly be changed to euclidean normalized distance
        dist = abs(ldo-ldc)
        return dist
    
    def loss_backward(self, student_imgs):
        # loss_Dgen = torch.nn.functional.softplus(gen_logits)
        gen_logits_o = self.run_D(self.teacher_imgs, None, sync=False)
        gen_logits_c = self.run_D(student_imgs, None, sync=False)
        ldo = torch.nn.functional.softplus(gen_logits_o)
        ldc = torch.nn.functional.softplus(gen_logits_c)
        loss = self.distance(ldo, ldc)
        training_stats.report("SelfSupervised/loss", loss)
        loss.backward()