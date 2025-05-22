python train.py \
  --data-path /data/n.kotov1/despeckle_dataset \
  --model DespeckleNet \
  --weight-dir checkpoints \
  --channels 1 \
  --batch-size 16 \
  --gpu 1 \
  --lr 0.001 \
  --epochs 50 \
  --patience 10
