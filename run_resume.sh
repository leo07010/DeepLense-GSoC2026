#!/bin/bash
#SBATCH --job-name=Resume
#SBATCH --partition=dev
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:3
#SBATCH --time=02:00:00
#SBATCH --account=GOV114009
#SBATCH --output=resume_%j.log

module load pytorch/2.8
export PYTHONPATH=/home/leo07010/.local/lib/python3.9/site-packages:$PYTHONPATH

cd /home/leo07010/GoogleSC/DeepLense

# GPU 0: CNN+VQC (resume from epoch 44)
echo "=== CNN+VQC resume ==="
CUDA_VISIBLE_DEVICES=0 python -u -m jupyter nbconvert \
    --to notebook --execute \
    --ExecutePreprocessor.timeout=7200 \
    --output Test_III_CNN_VQC_executed.ipynb \
    Test_III_CNN_VQC.ipynb > cnn_vqc_nb.log 2>&1 &

# GPU 1: VQC Tuned (resume from epoch 44)
echo "=== VQC Tuned resume ==="
CUDA_VISIBLE_DEVICES=1 python -u -m jupyter nbconvert \
    --to notebook --execute \
    --ExecutePreprocessor.timeout=7200 \
    --output Test_III_VQC_tuned_executed.ipynb \
    Test_III_VQC_tuned.ipynb > vqc_tuned_nb.log 2>&1 &

# GPU 2: QCNN 8x8 (1 epoch left) then QCNN 14x14 (17 epochs left)
echo "=== QCNN 8x8 resume ==="
CUDA_VISIBLE_DEVICES=2 python -u train_qcnn.py \
    --mode e2e --data-dir data --epochs 30 --batch-size 8 --lr 0.01 \
    --downsample 8 --train-subset 3000 --num-workers 0 \
    --target nvidia --save-prefix qcnn_8x8 \
    > qcnn_8x8.log 2>&1

echo "=== QCNN 14x14 resume ==="
CUDA_VISIBLE_DEVICES=2 python -u train_qcnn.py \
    --mode e2e --data-dir data --epochs 30 --batch-size 8 --lr 0.01 \
    --downsample 14 --train-subset 3000 --num-workers 0 \
    --target nvidia --save-prefix qcnn_14x14 \
    > qcnn_14x14.log 2>&1 &

wait
echo "=== All done ==="
