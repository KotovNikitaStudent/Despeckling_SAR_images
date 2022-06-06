import os
import numpy as np
import torch
from torch.utils import data
from glob import glob
from skimage import io


def load_image(path_to_image):
    image = io.imread(path_to_image)
    return image


class DespeckleDataset(data.Dataset):
    def __init__(self, image_path):
        self.path_images = sorted(glob(os.path.join(image_path, "*"), recursive=True))
        self.len = len(self.data)
        self.images = [load_image(i) for i in self.path_images]

    def __len__(self):
        return self.len
    
    def __getitem__(self, idx):
        image = self.images[idx]
        image = np.expand_dims(image, axis=-1)
        image = torch.transpose(image, (2, 0, 1))
        image = torch.from_numpy(image)
        return image
