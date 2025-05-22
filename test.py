import os
import numpy as np
import torch
from skimage import io
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from torchvision import transforms
import argparse
from glob import glob
from logger import logger

from models import (
    DespeckleNet,
    DespeckleNetPlusPlus,
    CBAMDilatedNet,
    MultiScaleReconstructionNet,
)

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

    logger.info("Аргументы командной строки:")
    for k, v in vars(args).items():
        logger.info(f"\t{k}: {v}")

    transform = transforms.Compose([transforms.ToTensor()])

    image_paths = sorted(glob(os.path.join(args.noisy_dir, "*.*")))

    total_psnr = 0.0
    total_ssim = 0.0
    count = 0

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

        # Преобразование входного изображения
        input_tensor = transform(noisy_image).unsqueeze(0).to(device)

        with torch.no_grad():
            output_tensor = model(input_tensor).cpu().squeeze().numpy()

        # Обрезка и преобразование к uint8
        output_tensor = np.clip(output_tensor, 0, 1)
        denoised_image = (output_tensor * 255).astype(np.uint8)

        result_path = os.path.join(args.output_dir, f"{name}_denoised.png")
        io.imsave(result_path, denoised_image)

        if args.clean_dir:
            clean_path = os.path.join(args.clean_dir, os.path.basename(img_path))
            if os.path.exists(clean_path):
                clean_image = io.imread(clean_path)
                if len(clean_image.shape) > 2:
                    clean_image = clean_image[:, :, 0]
                clean_image = np.array(clean_image, dtype=np.float32)
                denoised_resized = np.array(denoised_image, dtype=np.float32)

                if clean_image.shape != denoised_resized.shape:
                    print(f"Размеры не совпадают для {name}. Пропуск метрик.")
                    continue

                # <<< ОСНОВНАЯ ПОПРАВКА ТУТ >>>
                psnr_val = psnr(clean_image, denoised_resized, data_range=255)
                ssim_val = ssim(
                    clean_image,
                    denoised_resized,
                    data_range=255,
                    channel_axis=None,
                    win_size=7,  # SSIM лучше работает с окном 7x7
                )

                total_psnr += psnr_val
                total_ssim += ssim_val
                count += 1

                print(f"{name} | PSNR: {psnr_val:.2f} dB | SSIM: {ssim_val:.4f}")

        else:
            print(f"{name} обработано")

    if count > 0:
        avg_psnr = total_psnr / count
        avg_ssim = total_ssim / count
        print(f"\nСреднее PSNR: {avg_psnr:.2f} dB | Средний SSIM: {avg_ssim:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Инференс модели деспеклинга")

    parser.add_argument(
        "--noisy_dir",
        type=str,
        required=True,
        help="Путь к папке с зашумлёнными изображениями",
    )
    parser.add_argument(
        "--clean_dir",
        type=str,
        default="",
        help="Путь к папке с чистыми изображениями (если есть)",
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
