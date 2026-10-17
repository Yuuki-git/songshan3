#!/usr/bin/env bash
set -euo pipefail
umask 022
task_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)
source "$task_root/scripts/setup_env.sh"
source "$task_root/scripts/experiment_common.sh"
songshan3_begin hygon-fusion3d-core
python "$task_root/scripts/gpu_preflight.py" --output "$task_dir/preflight" > "$task_dir/preflight.log" 2>&1
python "$task_root/scripts/snapshot_environment.py" --output "$task_dir/environment.before.json" > "$task_dir/environment.before.log" 2>&1
python "$task_root/scripts/prepare_hygon_fusion_backend.py" --directory "$task_dir/ninetoothed" > "$task_dir/backend.log" 2>&1
export PYTHONPATH="$task_dir/ninetoothed/src:$task_root/benchmarks"
export NINETOOTHED_CACHE_DIR="$task_dir/ninetoothed-cache"
export TRITON_CACHE_DIR="$task_dir/triton-cache"
mkdir -p "$task_dir/task-source"
cp "$task_root/development/core_h3_fusion.cpp" "$task_root/development/core_h3_fusion.hpp" \
   "$task_root/benchmarks/hygon_fusion3d.py" "$task_root/benchmarks/hygon_fusion3d_host.py" \
   "$task_root/benchmarks/hygon_fusion3d_acceptance.py" "$task_root/benchmarks/hygon_native.py" \
   "$task_root/scripts/run_hygon_fusion3d_core.sh" "$task_root/scripts/prepare_hygon_fusion_backend.py" \
   "$task_dir/task-source/"
set +e
g++ -shared -std=c++17 -O3 -fPIC -pthread "$task_dir/task-source/core_h3_fusion.cpp" \
    -I "$INFINI_ROOT/include" -L "$INFINI_ROOT/lib" -Wl,-rpath,"$INFINI_ROOT/lib" \
    -Wl,-rpath-link,/usr/local/lib/python3.11/site-packages/torch.libs \
    -linfinicore_cpp_api -linfiniop -linfinirt -linfiniccl -ldl -Wl,-z,defs \
    -o "$task_dir/core-h3-fusion.so" > "$task_dir/build.log" 2>&1
task_status=$?
if [ "$task_status" -eq 0 ]; then
    ldd "$task_dir/core-h3-fusion.so" > "$task_dir/ldd.log"
    timeout 1200 python -u "$task_dir/task-source/hygon_fusion3d_acceptance.py" \
        --output "$task_dir/acceptance" --bridge "$task_dir/core-h3-fusion.so" > "$task_dir/run.log" 2>&1
    task_status=$?
fi
set -e
printf '%s\n' "$task_status" > "$task_dir/exit-code.txt"
unset PYTHONPATH
python "$task_root/scripts/snapshot_environment.py" --output "$task_dir/environment.after.json" > "$task_dir/environment.after.log" 2>&1
cat "$task_dir/build.log"
if [ -f "$task_dir/run.log" ]; then tail -8 "$task_dir/run.log"; fi
exit "$task_status"
