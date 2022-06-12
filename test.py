import os
import numpy as np
from glob import glob
from tqdm import tqdm
import torch
from utils import create_dir
from model import DespeckleFilter
from skimage import io


DATA_ROOT = "/Users/nikita/Downloads"

args = {
    "channels": 1,
    "device": 0,
}

def main():
    create_dir("results")
    checkpoint_path = "weight/despeckle_best.pth"

    test_image = os.path.join(DATA_ROOT, 'sentinel-radar_germany_test_45_1024_1408.tif')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = DespeckleFilter(args['channels'])
    model = model.to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()

    name = test_image.split("/")[-1].split(".")[0]
    image = io.imread(test_image)
    origin_image = image.copy()
    image = np.expand_dims(image, axis=0)
    image = np.expand_dims(image, axis=0)
    image = image / (2 ** 8 - 1)
    image = image.astype(np.float32)
    image = torch.from_numpy(image)
    image = image.to(device)
    
    with torch.no_grad():
        pred_y = model(image)
        pred_y = pred_y[0].cpu().numpy()
        pred_y = np.squeeze(pred_y, axis=0)
        pred_y = np.array(pred_y * (2 ** 8 - 1), dtype=np.uint8)

    size = origin_image.shape
    line = np.zeros((size[1], 10))
    cat_images = np.concatenate([origin_image, line, pred_y], axis=1)
    io.imsave(f"results/{name}.png", cat_images)


if __name__ == "__main__":
    main()
