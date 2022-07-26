from torch.utils.data import DataLoader
from utils import create_dir
from loss import TotalVariationLoss
from model import DespeckleFilter
from dataset import DespeckleDataset
import torch
from logger import logger

DATA_ROOT = "/raid/n.kotov1/sar_data/despeckle_dataset" # dataset contains 1-channel images

args = {
    "image_path": DATA_ROOT,
    "channels": 1,
    "batch_size": 16,
    "device": 3,
    "lr": 0.0002,
    "epochs": 500,   
}


def main():
    create_dir("weight")
    device = torch.cuda.set_device(args["device"])
    device = torch.device(f"cuda:{args['device']}")
    
    model = DespeckleFilter(args['channels'])
    model = model.to(device)
    
    dataset_train = DespeckleDataset(args["image_path"], mode='train')
    dataset_val = DespeckleDataset(args["image_path"], mode='val')
    
    dataloader_train = DataLoader(dataset_train, batch_size=args["batch_size"], num_workers=4, shuffle=True)
    dataloader_val = DataLoader(dataset_val, batch_size=args["batch_size"], num_workers=4, shuffle=True)

    loss_fn = TotalLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args['lr'])
        
    best_valid_loss = float("inf")

    for epoch in range(args["epochs"]+1):

        train_loss = train(model, dataloader_train, optimizer, loss_fn, device)
        valid_loss = evaluate(model, dataloader_val, loss_fn, device)

        if valid_loss < best_valid_loss:
            data_str = f"Valid loss improved from {best_valid_loss:2.5f} to {valid_loss:2.5f}. Saving checkpoint."
            # print(data_str)
            logger.info(data_str)

            best_valid_loss = valid_loss
            torch.save(model.state_dict(), "weight/despeckle_best.pth")

        data_str = f'Epoch: {epoch+1:02}\n'
        data_str += f'\tTrain Loss: {train_loss:.5f}\n'
        data_str += f'\tValid Loss: {valid_loss:.5f}\n'
        
        logger.info(data_str)
        # print(data_str)
        
    print('Training finished')
    logger.info('Training finished')
    

def train(model, loader, optimizer, loss_fn, device):
    epoch_loss = 0.0

    model.train()
    
    for x, y in loader:
        x = x.to(device, dtype=torch.float32)
        y = y.to(device, dtype=torch.float32)

        optimizer.zero_grad()
        y_pred = model(x)
        total_loss = loss_fn(y_pred, y)
        total_loss.backward()
        optimizer.step()
        epoch_loss += total_loss.item()

    epoch_loss = epoch_loss/len(loader)

    return epoch_loss


def evaluate(model, loader, loss_fn, device):
    epoch_loss = 0.0

    model.eval()
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device, dtype=torch.float32)
            y = y.to(device, dtype=torch.float32)

            y_pred = model(x)
            total_loss = loss_fn(y_pred, y)
            epoch_loss += total_loss.item()

        epoch_loss = epoch_loss/len(loader)

    return epoch_loss


if __name__ == "__main__":
    main()
