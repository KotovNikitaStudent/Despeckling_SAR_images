import os
import numpy as np
from glob import glob
from tqdm import tqdm
import torch
from Despeckling_SAR_images.utils import create_dir
from model import DespeckleFilter
from skimage import io


DATA_ROOT = "/Users/nikita/Downloads/despeckle_test" # TODO: here a full path to test the noising image

args = {
    "image_path": DATA_ROOT,
    "channels": 1,
    "device": 0,
}


def main():
    create_dir("results")
    checkpoint_path = "files/checkpoint.pth"

    test_images = glob(os.path.join(DATA_ROOT, '*'), recursive=True)
    
    device = torch.cuda.set_device(args["device"])
    device = torch.device(f"cuda:{args['device']}")
    
    model = DespeckleFilter(args['channels'])
    model = model.to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()

    for i, x in tqdm(enumerate(test_images), total=len(test_images)):
        name = x.split("/")[-1].split(".")[0]
        
        image = io.imread(x)
        x = np.transpose(image, (2, 0, 1))      ## (3, 512, 512)
        x = x/255.0
        x = np.expand_dims(x, axis=0)           ## (1, 3, 512, 512)
        x = x.astype(np.float32)
        x = torch.from_numpy(x)
        x = x.to(device)

        with torch.no_grad():
            pred_y = model(x)
            pred_y = pred_y[0].cpu().numpy()        ## (1, 512, 512)
            pred_y = np.squeeze(pred_y, axis=0)     ## (512, 512)
            pred_y = pred_y > 0.5
            pred_y = np.array(pred_y, dtype=np.uint8)

        io.imsave(f"results/{name}.tif", pred_y * 255)


if __name__ == "main":
    main()
