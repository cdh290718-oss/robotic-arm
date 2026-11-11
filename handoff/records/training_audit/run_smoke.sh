#!/usr/bin/env bash
set -euo pipefail
TASK_ROOT=/vepfs-mlp2/mlp-public/huyaoqing/so101_vla_20260918
export HTTP_PROXY=http://127.0.0.1:27897 HTTPS_PROXY=http://127.0.0.1:27897 NO_PROXY=localhost,127.0.0.1
export HF_HOME="$TASK_ROOT/cache/huggingface" HF_LEROBOT_HOME="$TASK_ROOT/data" TMPDIR="$TASK_ROOT/tmp"
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 TOKENIZERS_PARALLELISM=false
export HF_HUB_DISABLE_XET=1 HF_HUB_DOWNLOAD_TIMEOUT=120 HF_HUB_ETAG_TIMEOUT=30
export CUDA_VISIBLE_DEVICES=0 BENCHMARK_REPORT="$TASK_ROOT/outputs/smoke_resources.json"
export ACCELERATE_MIXED_PRECISION=bf16
export PATH="$TASK_ROOT/venv/bin:$PATH"
cd "$TASK_ROOT"
python -u train_limited.py \
 --policy.path=lerobot/smolvla_base \
 --dataset.repo_id=ChomCUI/yellow_cube_0918_v1 \
 --dataset.root="$TASK_ROOT/data/ChomCUI/yellow_cube_0918_v1" \
 --dataset.video_backend=pyav --dataset.eval_split=0.15 \
 --rename_map='{"observation.images.3":"observation.images.camera1","observation.images.1":"observation.images.camera2"}' \
 --policy.device=cuda --policy.use_amp=true --policy.push_to_hub=false \
 --batch_size=4 --num_workers=2 --prefetch_factor=1 --steps=200 \
 --log_freq=20 --save_freq=200 --env_eval_freq=0 --eval_steps=200 --max_eval_samples=32 \
 --seed=42 --wandb.enable=false --output_dir="$TASK_ROOT/outputs/smoke200" --job_name=so101_smoke200
echo SMOKE_COMPLETE
