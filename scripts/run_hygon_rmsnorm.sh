#!/usr/bin/env bash
set -euo pipefail
umask 022
task_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)
source "$task_root/scripts/setup_env.sh"
source "$task_root/scripts/experiment_common.sh"
songshan3_begin hygon-rmsnorm-native
python "$task_root/scripts/gpu_preflight.py" --output "$task_dir/preflight" > "$task_dir/preflight.log" 2>&1
python "$task_root/scripts/snapshot_environment.py" --output "$task_dir/environment.before.json" > "$task_dir/environment.before.log" 2>&1
cp "$task_root/development/core_ninetoothed_bridge.cpp" "$task_root/benchmarks/hygon_rmsnorm_probe.py" \
   "$task_root/benchmarks/hygon_rmsnorm_acceptance.py" "$task_root/benchmarks/hygon_native.py" \
   "$task_root/benchmarks/h3_operator_baseline.py" "$task_dir/task-source/"
export PYTHONPATH="$task_root/development/ninetoothed-hygon/src:$task_root/benchmarks"
export NINETOOTHED_CACHE_DIR="$task_dir/ninetoothed-cache"
export TRITON_CACHE_DIR="$task_dir/triton-cache"
g++ -shared -std=c++17 -O3 -fPIC -pthread "$task_root/development/core_ninetoothed_bridge.cpp" \
    -I "$INFINI_ROOT/include" -L "$INFINI_ROOT/lib" -Wl,-rpath,"$INFINI_ROOT/lib" \
    -Wl,-rpath-link,/usr/local/lib/python3.11/site-packages/torch.libs \
    -linfinicore_cpp_api -linfiniop -linfinirt -linfiniccl -ldl -Wl,-z,defs \
    -o "$task_dir/core-ninetoothed-bridge.so" > "$task_dir/build.log" 2>&1
set +e
timeout 1200 python -u "$task_root/benchmarks/hygon_rmsnorm_acceptance.py" \
    --bridge "$task_dir/core-ninetoothed-bridge.so" --output "$task_dir/acceptance" > "$task_dir/run.log" 2>&1
task_status=$?
set -e
unset PYTHONPATH
tail -30 "$task_dir/run.log"
exit "$task_status"
