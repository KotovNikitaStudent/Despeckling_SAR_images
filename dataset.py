import os
import numpy as np
from torch.utils.data import Dataset
from glob import glob
from skimage import io
import torchvision.transforms as transforms


def load_image(path_to_image):
    image = io.imread(path_to_image)
    if image.dtype == np.uint8:
        image = image.astype(np.float32) / 255.0
    elif image.dtype == np.uint16:
        image = image.astype(np.float32) / 65535.0
    else:
        image = image.astype(np.float32)
    return image


class DespeckleDataset(Dataset):
    def __init__(self, images_path, mode="train", transform=None):
        self.transform = transform or transforms.ToTensor()

        self.clean_paths = sorted(
            glob(os.path.join(images_path, mode, "clean", "*.tif"))
        )
        self.noise_paths = sorted(
            glob(os.path.join(images_path, mode, "noise", "*.tif"))
        )

        assert len(self.clean_paths) == len(self.noise_paths), (
            f"Количество clean и noise файлов не совпадает: {len(self.clean_paths)} vs {len(self.noise_paths)}"
        )

    def __len__(self):
        return len(self.clean_paths)

    def __getitem__(self, idx):
        clean_path = self.clean_paths[idx]
        noise_path = self.noise_paths[idx]

        clean_img = load_image(clean_path)
        noise_img = load_image(noise_path)

        if clean_img.ndim == 2:
            clean_img = np.expand_dims(clean_img, axis=-1)
        if noise_img.ndim == 2:
            noise_img = np.expand_dims(noise_img, axis=-1)

        if self.transform:
            clean_img = self.transform(clean_img)
            noise_img = self.transform(noise_img)

        return noise_img, clean_img


if __name__ == "__main__":
    dataset = DespeckleDataset("data/", mode="train")
    noisy, clean = dataset[0]
    print("Noisy shape:", noisy.shape)
    print("Clean shape:", clean.shape)
