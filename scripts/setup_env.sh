#!/usr/bin/env bash
# Source this file in the selected Hygon environment after every new shell.
export SONGSHAN3_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
export WORK_HOME="$(dirname -- "$SONGSHAN3_ROOT")"
export SONGSHAN3_BASE_PYTHON="${SONGSHAN3_BASE_PYTHON:-/usr/local/bin/python}"
# DTK's activation script is not nounset-safe.
task_had_nounset=0
case "$-" in *u*) task_had_nounset=1; set +u ;; esac
source /opt/dtk/env.sh
if [ "$task_had_nounset" -eq 1 ]; then set -u; fi
unset task_had_nounset
export XMAKE_ROOT=y
export XMAKE_GLOBALDIR="$SONGSHAN3_ROOT/tooling/xmake-global"
export XMAKE_PKG_CACHEDIR="$SONGSHAN3_ROOT/cache/xmake-packages"
export INFINI_ROOT="$SONGSHAN3_ROOT/infini"
export PATH="$SONGSHAN3_ROOT/tooling/xmake-3.0.5/bin:/opt/dtk/cuda/cuda/bin:${PATH}"
export LD_LIBRARY_PATH="${INFINI_ROOT}/lib:/opt/dtk/lib:/opt/dtk/cuda/cuda/lib64:/usr/local/lib/python3.11/site-packages/torch/lib:/usr/local/lib/python3.11/site-packages/torch.libs:/opt/hyhal/lib:/opt/dtk/.hyhal/rocm_smi/lib:/opt/dtk/.hyhal/rocm_smi:${LD_LIBRARY_PATH:-}"
export TRITON_CACHE_DIR="$SONGSHAN3_ROOT/cache/triton"
export NINETOOTHED_CACHE_DIR="$SONGSHAN3_ROOT/cache/ninetoothed"
export PIP_NO_INDEX=1
export PIP_FIND_LINKS="$SONGSHAN3_ROOT/wheels"
export PIP_DISABLE_PIP_VERSION_CHECK=1
if [ -f "$SONGSHAN3_ROOT/venvs/h3/bin/activate" ]; then
    source "$SONGSHAN3_ROOT/venvs/h3/bin/activate"
fi
