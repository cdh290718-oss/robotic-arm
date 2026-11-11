#!/usr/bin/env bash
set -euo pipefail
TASK_ROOT=/vepfs-mlp2/mlp-public/huyaoqing/so101_vla_20260918
mkdir -p "$TASK_ROOT"/{cache,python,tmp,logs,outputs,data,source}
export HTTP_PROXY=http://127.0.0.1:27897 HTTPS_PROXY=http://127.0.0.1:27897
export NO_PROXY=localhost,127.0.0.1
export UV_CACHE_DIR="$TASK_ROOT/cache/uv" UV_PYTHON_INSTALL_DIR="$TASK_ROOT/python" TMPDIR="$TASK_ROOT/tmp"
export UV_CONCURRENT_DOWNLOADS=2 UV_CONCURRENT_BUILDS=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
/root/.local/bin/uv venv --python 3.12 "$TASK_ROOT/venv"
/root/.local/bin/uv pip install --python "$TASK_ROOT/venv/bin/python" 'torch==2.7.1' 'torchvision==0.22.1' --index-url https://download.pytorch.org/whl/cu118
/root/.local/bin/uv pip install --python "$TASK_ROOT/venv/bin/python" "$TASK_ROOT/source/lerobot-0.6.0[smolvla,training,core_scripts]" 'torch==2.7.1' 'torchvision==0.22.1' 'torchcodec==0.4.0'
/root/.local/bin/uv pip check --python "$TASK_ROOT/venv/bin/python"
echo SETUP_COMPLETE
