#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
mode=${1:-smoke}
export MUJOCO_GL=egl
export PYTHONUNBUFFERED=1
mkdir -p run_records
python -c 'import torch; print(torch.__version__, torch.version.cuda); assert torch.cuda.is_available(), "CUDA unavailable"; print(torch.cuda.get_device_name(0))'
python -m pip freeze > run_records/packages.txt
nvidia-smi > run_records/nvidia-smi.txt
case "$mode" in
  smoke)
    python src/train.py task=walker-walk seed=1 exp_name=smoke train_steps=1000 seed_steps=500 eval_episodes=1 eval_freq=1000 save_model=true use_wandb=false 2>&1 | tee run_records/smoke.txt
    ;;
  full)
    for seed in 1 2 3 4 5; do
      python src/train.py task=walker-walk seed="$seed" exp_name=reproduction save_model=true use_wandb=false 2>&1 | tee "run_records/walker-walk-${seed}.txt"
    done
    ;;
  *) echo "Usage: bash run_reproduction.sh [smoke|full]" >&2; exit 2 ;;
esac
