import os
from pyexpat import model
from statistics import mode
from torch.utils.data import DataLoader
from loss import TotalVariationLoss
from model import DespeckleFilter
from dataset import DespeckleDataset
from torch import optim


DATA_ROOT = "/"

args = {
    "image_path": DATA_ROOT,
    "channels": 1,
    "batch_size": 16,
    "gpu": 0,
    "lr": 1e-3,
    "epoch": 100,   
}


def main():
    os.environ['CUDA_VISIBLE_DEVICE'] = args["gpu"]
    
    model = DespeckleFilter()
    model = model.cuda()
    
    dataset_train = DespeckleDataset(args["image_path"])
    dataloader = DataLoader(dataset_train, batch_size=args["batch_size"], num_workers=4, shuffle=True)
    loss = TotalVariationLoss()
    loss = loss.cuda()
    optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=args['lr'], weight_decay=5e-4, momentum=0.9, nesterov=True)


def train():
    pass


if __name__ == "__main__":
    main()