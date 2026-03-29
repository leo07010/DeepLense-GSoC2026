#!/bin/bash
#SBATCH --job-name=Notebooks
#SBATCH --partition=dev
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --account=GOV114009
#SBATCH --output=notebooks_%j.log

module load pytorch/2.8
export PYTHONPATH=/home/leo07010/.local/lib/python3.9/site-packages:$PYTHONPATH

cd /home/leo07010/GoogleSC/DeepLense

# Execute notebooks with outputs embedded
echo "=== Test I: CNN ==="
python -u -m jupyter nbconvert \
    --to notebook --execute \
    --ExecutePreprocessor.timeout=7200 \
    --output Test_I_Classical_CNN_executed.ipynb \
    notebooks/Test_I_Classical_CNN.ipynb

echo "=== Test III: VQC PennyLane ==="
python -u -m jupyter nbconvert \
    --to notebook --execute \
    --ExecutePreprocessor.timeout=7200 \
    --output Test_III_VQC_PennyLane_executed.ipynb \
    notebooks/Test_III_VQC_PennyLane.ipynb

echo "=== Done ==="
