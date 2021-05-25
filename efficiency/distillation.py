from abc import abstractmethod

import numpy as np
import torch
from torch_utils import training_stats

import os 
import torch.utils.data as TD
from skimage import io, transform
# from PIL import Image

import torchvision.transforms.functional as TF
import torch.utils.data.Dataset as Dataset

CLOSE_TO_ZERO = 1e-3


class DistillationDataset(Dataset):   
    
    def __init__(self, directory):
        self.directory = directory
        self.image_ids = os.listdir(directory)
    
    def __len__(self):
        return len(self.image_ids)
        
    def __getitem__(self, index):
        if type(index) == int or type(index) == float:
            index = int(index)
            if len(str(index)) < 4: #Append zeros before seed
                seednew = ''
                for i in range(4-len(str(index))):
                    seednew += '0'
                seednew += str(index)
                index = seednew
            elif len(str(index)) > 4: #Only 4 digit seeds
                return False 
        index = 'seed' + str(index)

        path = os.path.join(self.directory, index)
        image = io.imread(path)
        
        return image
    
    
class Distillation:
    
    # @abstractmethod
    # def forward_hook(mod, inputs, outputs):
    #     raise NotImplementedError

    # @abstractmethod
    # def before_minibatch(self):
    #     pass

    # @abstractmethod
    # def after_minibatch(self):
    #     pass

    @abstractmethod
    def before_backward(self):
        pass
   
    
class RMSE(Distillation):
    
    def __init__(self):
        self.dataset = DistillationDataset(directory = 'C:/Users/Levi/Desktop/CS4245/stylegan2-ada-pytorch/out/')
        return

    def set_seed(seed):
        # random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    def before_backward(self, real_img, gen_img, seed):
        self.set_seed(seed)
        self.dataset.__getitem__(seed)
        
        RMSE = 0
        

    """
    TD.DataLoader()
    """


class Advanced(Distillation):
    """
    
    """

    def __init__(self):
        return

    def before_backward(self):
        return
