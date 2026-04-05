#!/bin/bash
#SBATCH --job-name=crisis-predictor
#SBATCH --partition=gpu-long
#SBATCH --gres=gpu:h100-96:1
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --nodelist=xgpi11
#SBATCH --output=logs/pipeline_%j.out
#SBATCH --error=logs/pipeline_%j.err

set -e
cd ~/community-crisis-predictor
mkdir -p logs

WORKDIR=/tmp/$USER-crisis-$SLURM_JOB_ID
python3 -m venv $WORKDIR/venv
source $WORKDIR/venv/bin/activate
export TMPDIR=$WORKDIR
export PYTHONUNBUFFERED=1

pip install --upgrade pip --no-cache-dir
pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cu121
pip install --no-cache-dir -e "$HOME/community-crisis-predictor[dev]"
pip install --no-cache-dir statsmodels

echo "=== Job started: $(date) ==="
echo "=== Node: $(hostname) ==="
python -c "import torch; print('CUDA available:', torch.cuda.is_available())"

python -u -m src.pipeline.run_all --config config/default.yaml --skip-topics --force

echo "=== Job finished: $(date) ==="
