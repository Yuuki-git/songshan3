#!/usr/bin/env bash
set -euo pipefail
umask 022
task_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)
source "$task_root/scripts/setup_env.sh"
source "$task_root/scripts/experiment_common.sh"
songshan3_begin hygon-rms3d-native
mkdir -p "$task_dir/task-source"
cp "$task_root/benchmarks/hygon_rmsnorm3d_probe.py" "$task_root/benchmarks/hygon_rmsnorm3d_acceptance.py" "$task_root/benchmarks/hygon_rmsnorm_probe.py" "$task_root/benchmarks/hygon_native.py" "$task_root/benchmarks/h3_operator_baseline.py" "$task_dir/task-source/"
cp "$task_root/development/core_ninetoothed_bridge.cpp" "$task_root/development/core_ninetoothed_bridge_v2.cpp" "$task_root/scripts/gpu_preflight.py" "$task_root/scripts/run_hygon_rmsnorm3d.sh" "$task_dir/task-source/"
set +e
python "$task_root/scripts/gpu_preflight.py" --output "$task_dir/preflight" > "$task_dir/preflight.log" 2>&1
task_preflight_status=$?
set -e
if [ "$task_preflight_status" -ne 0 ]; then
    cat "$task_dir/preflight.log"
    printf '%s\n' "$task_preflight_status" > "$task_dir/exit-code.txt"
    printf 'GPU preflight failed; no numerical acceptance or timing was run.\n' >&2
    exit "$task_preflight_status"
fi
python "$task_root/scripts/snapshot_environment.py" --output "$task_dir/environment.before.json" > "$task_dir/environment.before.log" 2>&1
export PYTHONPATH="$task_root/development/ninetoothed-hygon/src:$task_root/benchmarks"
export NINETOOTHED_CACHE_DIR="$task_dir/ninetoothed-cache"
export TRITON_CACHE_DIR="$task_dir/triton-cache"
g++ -shared -std=c++17 -O3 -fPIC -pthread "$task_root/development/core_ninetoothed_bridge_v2.cpp" \
    -I "$INFINI_ROOT/include" -L "$INFINI_ROOT/lib" -Wl,-rpath,"$INFINI_ROOT/lib" \
    -Wl,-rpath-link,/usr/local/lib/python3.11/site-packages/torch.libs \
    -linfinicore_cpp_api -linfiniop -linfinirt -linfiniccl -ldl -Wl,-z,defs \
    -o "$task_dir/core-ninetoothed-bridge-v2.so" > "$task_dir/bridge-build.log" 2>&1
set +e
timeout 900 python -u "$task_root/benchmarks/hygon_rmsnorm3d_probe.py" --output "$task_dir/probe" > "$task_dir/probe.log" 2>&1
task_status=$?
if [ "$task_status" -eq 0 ]; then
    task_vector=$(python -c 'import json,sys; print("--vector" if json.load(open(sys.argv[1]))["passed_candidates"][0] else "")' "$task_dir/probe/probe.json")
    timeout 1500 python -u "$task_root/benchmarks/hygon_rmsnorm3d_acceptance.py" \
        --bridge "$task_dir/core-ninetoothed-bridge-v2.so" --output "$task_dir/acceptance" $task_vector > "$task_dir/run.log" 2>&1
    task_status=$?
fi
set -e
printf '%s\n' "$task_status" > "$task_dir/exit-code.txt"
unset PYTHONPATH
python "$task_root/scripts/snapshot_environment.py" --output "$task_dir/environment.after.json" > "$task_dir/environment.after.log" 2>&1
if [ -f "$task_dir/run.log" ]; then tail -1 "$task_dir/run.log"; else tail -2 "$task_dir/probe.log"; fi
exit "$task_status"
