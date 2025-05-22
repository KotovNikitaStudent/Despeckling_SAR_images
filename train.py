import os
from torch.utils.data import DataLoader
from utils import create_dir

from models import DespeckleNet
from models import DespeckleNetPlusPlus
from models import CBAMDilatedNet
from models import MultiScaleReconstructionNet

from dataset import DespeckleDataset
from loss import CharbonnierLoss
import torch
import numpy as np
from torch.cuda.amp import autocast, GradScaler
from logger import logger
import argparse
from torchvision import transforms

from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim


MODEL_MAP = {
    "DespeckleNet": DespeckleNet,
    "DespeckleNetPlusPlus": DespeckleNetPlusPlus,
    "CBAMDilatedNet": CBAMDilatedNet,
    "MultiScaleReconstructionNet": MultiScaleReconstructionNet,
}


def main(args):
    create_dir(args.weight_dir)

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    logger.info(f"Используется устройство: {device}")

    transform_train = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomRotation(90),
        ]
    )

    transform_val = transforms.ToTensor()

    model_class = MODEL_MAP[args.model]
    model = model_class(in_channels=args.channels)
    model = model.to(device)

    dataset_train = DespeckleDataset(
        args.data_path, mode="train", transform=transform_train
    )
    dataset_val = DespeckleDataset(args.data_path, mode="val", transform=transform_val)

    dataloader_train = DataLoader(
        dataset_train, batch_size=args.batch_size, num_workers=4, shuffle=True
    )
    dataloader_val = DataLoader(dataset_val, batch_size=args.batch_size, num_workers=4)

    loss_fn = CharbonnierLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)

    scaler = GradScaler()

    best_valid_loss = float("inf")

    for epoch in range(args.epochs):
        train_loss = train(model, dataloader_train, optimizer, loss_fn, device, scaler)
        valid_loss, avg_psnr, avg_ssim = evaluate_with_metrics(
            model, dataloader_val, loss_fn, device
        )

        if valid_loss < best_valid_loss:
            data_str = f"Valid loss improved from {best_valid_loss:.5f} to {valid_loss:.5f}. Saving checkpoint."
            logger.info(data_str)

            best_valid_loss = valid_loss
            torch.save(
                model.state_dict(),
                os.path.join(args.weight_dir, f"{args.model}_despeckle_best.pth"),
            )

        data_str = f"Epoch: {epoch + 1:02}\n"
        data_str += f"\tTrain Loss: {train_loss:.5f}\n"
        data_str += f"\tValid Loss: {valid_loss:.5f}\n"
        data_str += f"\tPSNR: {avg_psnr:.2f} dB\n"
        data_str += f"\tSSIM: {avg_ssim:.4f}\n"

        logger.info(data_str)

    logger.info("Training finished")


def train(model, loader, optimizer, loss_fn, device, scaler):
    model.train()
    epoch_loss = 0.0

    for x, y in loader:
        x = x.to(device, dtype=torch.float32)
        y = y.to(device, dtype=torch.float32)

        optimizer.zero_grad()

        with autocast():
            y_pred = model(x)
            loss = loss_fn(y_pred, y)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        epoch_loss += loss.item()

    return epoch_loss / len(loader)


def evaluate_with_metrics(model, loader, loss_fn, device):
    model.eval()
    epoch_loss = 0.0
    total_psnr = 0.0
    total_ssim = 0.0
    count = 0

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device, dtype=torch.float32)
            y = y.to(device, dtype=torch.float32)

            y_pred = model(x)
            loss = loss_fn(y_pred, y)
            epoch_loss += loss.item()

            y_pred_np = y_pred.cpu().numpy()
            y_true_np = y.cpu().numpy()

            for i in range(len(y_pred_np)):
                pred = np.clip(y_pred_np[i, 0], 0, 1)
                true = y_true_np[i, 0]

                pred_uint8 = (pred * 255).astype(np.uint8)
                true_uint8 = (true * 255).astype(np.uint8)

                total_psnr += psnr(pred_uint8, true_uint8)
                total_ssim += ssim(pred_uint8, true_uint8, channel_axis=None)
                count += 1

    avg_loss = epoch_loss / len(loader)
    avg_psnr = total_psnr / count if count > 0 else float("nan")
    avg_ssim = total_ssim / count if count > 0 else float("nan")

    return avg_loss, avg_psnr, avg_ssim


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Обучение модели деспеклинга SAR-изображений"
    )

    parser.add_argument(
        "--data-path",
        type=str,
        required=True,
        help="Путь к директории с данными (должна содержать train/val)",
    )
    parser.add_argument(
        "--weight-dir",
        type=str,
        default="weights",
        help="Папка для сохранения весов модели",
    )
    parser.add_argument(
        "--channels",
        type=int,
        default=1,
        help="Число каналов изображения (обычно 1 для SAR)",
    )
    parser.add_argument("--batch-size", type=int, default=16, help="Размер батча")
    parser.add_argument("--gpu", type=int, default=0, help="Номер GPU для обучения")
    parser.add_argument("--lr", type=float, default=0.0002, help="Learning rate")
    parser.add_argument("--epochs", type=int, default=500, help="Число эпох обучения")

    parser.add_argument(
        "--model",
        type=str,
        default="DespeckleNet",
        choices=list(MODEL_MAP.keys()),
        help="Модель для обучения",
    )

    args = parser.parse_args()

    logger.info(f"Выбрана модель: {args.model}")
    main(args)
