from torch.utils.data import DataLoader
from loss import TotalVariationLoss
from model import DespeckleFilter
from dataset import DespeckleDataset
from torch import optim
import torch
from tqdm import tqdm


DATA_ROOT = "/Users/nikita/Downloads/despeckle_dataset/s2" # dataset contains 1-channel images

args = {
    "image_path": DATA_ROOT,
    "channels": 1,
    "batch_size": 16,
    "device": 0,
    "lr": 1e-4,
    "epochs": 500,   
}


def main():
    device = torch.cuda.set_device(args["device"])
    device = torch.device(f"cuda:{args['device']}")
    
    model = DespeckleFilter(args['channels'])
    model = model.to(device)
    
    dataset_train = DespeckleDataset(args["image_path"])
    dataloader = DataLoader(dataset_train, batch_size=args["batch_size"], num_workers=4, shuffle=True)

    loss_fn = TotalVariationLoss()
    optimizer = optim.NAdam(model.parameters(), lr=args['lr'])
        
    best_train_loss = float("inf")

    for epoch in range(args["epochs"]+1):

        train_loss = train(model, dataloader, optimizer, loss_fn, device)

        if train_loss < best_train_loss:
            data_str = f"Train loss improved from {best_train_loss:2.5f} to {train_loss:2.5f}. Saving checkpoint."
            print(data_str)

            best_train_loss = train_loss
            torch.save(model.state_dict(), "despeckle_best.pth")

        data_str = f'Epoch: {epoch+1:02}\n'
        data_str += f'\tTrain Loss: {train_loss:.5f}\n'
        
        print(data_str)
        
    print('Training finished')
    

def train(model, loader, optimizer, loss_fn, device):
    epoch_loss = 0.0

    model.train()
    
    for x, y in tqdm(loader, total=len(loader)):
        x = x.to(device, dtype=torch.float32)
        y = y.to(device, dtype=torch.float32)

        optimizer.zero_grad()
        y_pred = model(x)
        total_loss, _, _ = loss_fn(y_pred, y)
        total_loss.backward()
        optimizer.step()
        epoch_loss += total_loss.item()

    epoch_loss = epoch_loss/len(loader)

    return epoch_loss


if __name__ == "__main__":
    main()