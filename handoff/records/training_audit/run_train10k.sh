#!/usr/bin/env bash
set -euo pipefail
TASK_ROOT=/vepfs-mlp2/mlp-public/huyaoqing/so101_vla_20260918
exec 9>"$TASK_ROOT/train10k.lock"
flock -n 9 || exit 2
export HTTP_PROXY=http://127.0.0.1:27897 HTTPS_PROXY=http://127.0.0.1:27897 NO_PROXY=localhost,127.0.0.1
export HF_HOME="$TASK_ROOT/cache/huggingface" HF_LEROBOT_HOME="$TASK_ROOT/data" TMPDIR="$TASK_ROOT/tmp"
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 TOKENIZERS_PARALLELISM=false
export HF_HUB_DISABLE_XET=1 HF_HUB_DOWNLOAD_TIMEOUT=120 HF_HUB_ETAG_TIMEOUT=30
export CUDA_VISIBLE_DEVICES=0 BENCHMARK_REPORT="$TASK_ROOT/outputs/train10k_resources.json"
export ACCELERATE_MIXED_PRECISION=bf16
export PATH="$TASK_ROOT/venv/bin:$PATH"
cd "$TASK_ROOT"
test -f "$TASK_ROOT/outputs/smoke200/checkpoints/last/pretrained_model/train_config.json"
python -u train_limited.py \
 --config_path="$TASK_ROOT/outputs/smoke200/checkpoints/last/pretrained_model/train_config.json" \
 --resume=true --steps=10000 --save_freq=1000 --log_freq=100 \
 --eval_steps=1000 --max_eval_samples=128 \
 --output_dir="$TASK_ROOT/outputs/train10k" --job_name=so101_train10k
echo TRAIN10K_COMPLETE
