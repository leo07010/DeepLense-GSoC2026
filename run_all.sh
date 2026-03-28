#!/bin/bash
#SBATCH --job-name=DeepLense
#SBATCH --partition=dev
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:3
#SBATCH --time=02:00:00
#SBATCH --account=GOV114009
#SBATCH --output=deeplense_%j.log

module load pytorch/2.8
export PYTHONPATH=/home/leo07010/.local/lib/python3.9/site-packages:$PYTHONPATH

cd /home/leo07010/GoogleSC/DeepLense

# GPU 0: QCNN e2e — 8x8 (SPSA, 4 patches)
echo "=== Launching QCNN 8x8 and 14x14 ==="
CUDA_VISIBLE_DEVICES=0 python -u train_qcnn.py \
    --mode e2e \
    --data-dir data \
    --epochs 30 \
    --batch-size 8 \
    --lr 0.01 \
    --downsample 8 \
    --train-subset 3000 \
    --num-workers 0 \
    --target nvidia \
    --save-prefix qcnn_8x8 \
    > qcnn_8x8.log 2>&1 &

# GPU 1: QCNN e2e — 14x14 (SPSA, 9 patches)
CUDA_VISIBLE_DEVICES=1 python -u train_qcnn.py \
    --mode e2e \
    --data-dir data \
    --epochs 30 \
    --batch-size 8 \
    --lr 0.01 \
    --downsample 14 \
    --train-subset 3000 \
    --num-workers 0 \
    --target nvidia \
    --save-prefix qcnn_14x14 \
    > qcnn_14x14.log 2>&1 &

wait
echo "=== All done ==="
echo "--- 8x8 results ---"
tail -5 qcnn_8x8.log
echo "--- 14x14 results ---"
tail -5 qcnn_14x14.log
