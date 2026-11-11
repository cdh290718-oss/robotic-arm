#!/usr/bin/env bash
set -euo pipefail
TASK_ROOT=/vepfs-mlp2/mlp-public/huyaoqing/so101_vla_20260918
exec 9>"$TASK_ROOT/pipeline.lock"
flock -n 9 || exit 2
echo WAITING_FOR_SETUP
ready=0
for i in $(seq 1 120); do
 if grep -q SETUP_COMPLETE "$TASK_ROOT/logs/setup.log"; then ready=1; break; fi
 sleep 15
done
test "$ready" = 1 || { echo SETUP_TIMEOUT; exit 3; }
export HF_HOME="$TASK_ROOT/cache/huggingface" HF_LEROBOT_HOME="$TASK_ROOT/data" TMPDIR="$TASK_ROOT/tmp"
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 TOKENIZERS_PARALLELISM=false
export HTTP_PROXY=http://127.0.0.1:27897 HTTPS_PROXY=http://127.0.0.1:27897 NO_PROXY=localhost,127.0.0.1
export PATH="$TASK_ROOT/venv/bin:$PATH"
python -c 'import torch; torch.cuda.set_per_process_memory_fraction(0.2); x=torch.ones((32,32),device="cuda"); print("CUDA_OK",torch.__version__,(x@x).mean().item())'
python -u "$TASK_ROOT/prepare_remote_data.py" > "$TASK_ROOT/logs/prepare.log" 2>&1
bash "$TASK_ROOT/run_smoke.sh" > "$TASK_ROOT/logs/smoke.log" 2>&1
echo PIPELINE_SMOKE_COMPLETE
