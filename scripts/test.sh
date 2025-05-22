python test.py \
  --model DespeckleNet \
  --noisy_dir /data/n.kotov1/despeckle_dataset/test/noise \
  --clean_dir /data/n.kotov1/despeckle_dataset/test/clean \
  --checkpoint checkpoints/DespeckleNet_despeckle_best.pth \
  --output_dir /data/n.kotov1/despeckle_dataset/inference
  