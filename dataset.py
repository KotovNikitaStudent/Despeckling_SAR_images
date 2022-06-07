import os
import numpy as np
import torch
from torch.utils import data
from glob import glob
from skimage import io


def load_image(path_to_image):
    image = io.imread(path_to_image)
    image = image / (2 ** 8 - 1)
    return image


class DespeckleDataset(data.Dataset):
    def __init__(self, images_path, mode='train'):
        self.clean_images_path = sorted(glob(os.path.join(images_path, mode, "clean", "*.tif"), recursive=True))
        self.noise_images_path = sorted(glob(os.path.join(images_path, mode, "noise", "*.tif"), recursive=True))
        self.len = len(self.clean_images_path)
        self.images_clean = [load_image(i) for i in self.clean_images_path]
        self.images_noise = [load_image(i) for i in self.noise_images_path]

    def __len__(self):
        return self.len
    
    def __getitem__(self, idx):
        image_clean = self.images_clean[idx]
        image_noise = self.images_noise[idx]
                
        if image_clean.ndim == 2 and image_noise.ndim == 2:        
            image_clean = np.expand_dims(image_clean, axis=-1)
            image_noise = np.expand_dims(image_noise, axis=-1)
        
        image_clean = np.transpose(image_clean, (2, 0, 1))
        image_noise = np.transpose(image_noise, (2, 0, 1))
        
        image_clean = torch.from_numpy(image_clean)
        image_noise = torch.from_numpy(image_noise)
        
        return image_clean, image_noise 
