import os
import numpy as np
import torch
from skimage import io
from torchvision import transforms
import argparse
from glob import glob

from models import DespeckleNet
from models import DespeckleNetPlusPlus
from models import CBAMDilatedNet
from models import MultiScaleReconstructionNet


MODEL_MAP = {
    "DespeckleNet": DespeckleNet,
    "DespeckleNetPlusPlus": DespeckleNetPlusPlus,
    "CBAMDilatedNet": CBAMDilatedNet,
    "MultiScaleReconstructionNet": MultiScaleReconstructionNet,
}


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    os.makedirs(args.output_dir, exist_ok=True)

    model_class = MODEL_MAP[args.model]
    model = model_class(in_channels=1)

    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.to(device)
    model.eval()

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
        ]
    )

    image_paths = sorted(glob(os.path.join(args.noisy_dir, "*.*")))

    print(f"Начинаем обработку {len(image_paths)} изображений...")

    for img_path in image_paths:
        name = os.path.splitext(os.path.basename(img_path))[0]

        try:
            noisy_image = io.imread(img_path)
        except Exception as e:
            print(f"Ошибка загрузки файла {img_path}: {e}")
            continue

        if len(noisy_image.shape) > 2:
            noisy_image = noisy_image[:, :, 0]

        input_tensor = transform(noisy_image).unsqueeze(0).to(device)

        with torch.no_grad():
            output_tensor = model(input_tensor).cpu().squeeze().numpy()

        output_tensor = np.clip(output_tensor, 0, 1)
        denoised_image = (output_tensor * 255).astype(np.uint8)

        result_path = os.path.join(args.output_dir, f"{name}_denoised.png")
        io.imsave(result_path, denoised_image)

        print(f"{name} обработано")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Инференс модели деспеклинга")

    parser.add_argument(
        "--noisy_dir",
        type=str,
        required=True,
        help="Путь к папке с зашумлёнными изображениями",
    )
    parser.add_argument(
        "--checkpoint", type=str, required=True, help="Путь к весам модели (.pth)"
    )
    parser.add_argument(
        "--output_dir", type=str, required=True, help="Папка для сохранения результатов"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="DespeckleNet",
        choices=list(MODEL_MAP.keys()),
        help="Модель для инференса",
    )

    args = parser.parse_args()
    main(args)
