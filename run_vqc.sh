#!/bin/bash
#SBATCH --job-name=VQC
#SBATCH --partition=dev
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:3
#SBATCH --time=02:00:00
#SBATCH --account=GOV114009
#SBATCH --output=vqc_%j.log

module load pytorch/2.8
export PYTHONPATH=/home/leo07010/.local/lib/python3.9/site-packages:$PYTHONPATH

cd /home/leo07010/GoogleSC/DeepLense

# GPU 0: PCA+VQC tuned
echo "=== VQC Tuned ==="
CUDA_VISIBLE_DEVICES=0 python -u -m jupyter nbconvert \
    --to notebook --execute \
    --ExecutePreprocessor.timeout=7200 \
    --output Test_III_VQC_tuned_executed.ipynb \
    Test_III_VQC_tuned.ipynb > vqc_tuned_nb.log 2>&1 &

# GPU 1: CNN+VQC
echo "=== CNN+VQC ==="
CUDA_VISIBLE_DEVICES=1 python -u -m jupyter nbconvert \
    --to notebook --execute \
    --ExecutePreprocessor.timeout=7200 \
    --output Test_III_CNN_VQC_executed.ipynb \
    Test_III_CNN_VQC.ipynb > cnn_vqc_nb.log 2>&1 &

wait
echo "=== All done ==="
echo "--- VQC Tuned ---"
tail -3 vqc_tuned_nb.log
echo "--- CNN+VQC ---"
tail -3 cnn_vqc_nb.log
