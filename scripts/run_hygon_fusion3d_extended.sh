#!/usr/bin/env bash
set -euo pipefail
umask 022
task_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)
source "$task_root/scripts/setup_env.sh"
task_source="${SONGSHAN3_FUSION_SOURCE:?Set SONGSHAN3_FUSION_SOURCE to the successful main experiment from this run}"
test -f "$task_source/acceptance/acceptance.json"
test "$(cat "$task_source/exit-code.txt")" = 0
source "$task_root/scripts/experiment_common.sh"
songshan3_begin hygon-fusion3d-extended
mkdir -p "$task_dir/task-source"
cp "$task_root/development/core_fusion3d_standalone.cpp" "$task_root/development/core_h3_fusion.hpp" \
   "$task_root/benchmarks/hygon_fusion3d_benchmark.py" "$task_root/benchmarks/hygon_fusion3d_host.py" \
   "$task_root/benchmarks/hygon_fusion3d.py" "$task_root/scripts/run_hygon_fusion3d_extended.sh" "$task_dir/task-source/"
printf '%s\n' "$task_source" > "$task_dir/source-experiment.txt"
python "$task_root/scripts/gpu_preflight.py" --output "$task_dir/preflight" > "$task_dir/preflight.log" 2>&1
python "$task_root/scripts/snapshot_environment.py" --output "$task_dir/environment.before.json" > "$task_dir/environment.before.log" 2>&1
set +e
g++ -std=c++17 -O2 -I/opt/dtk/include -I "$INFINI_ROOT/include" "$task_dir/task-source/core_fusion3d_standalone.cpp" \
    -L "$INFINI_ROOT/lib" -L/opt/dtk/lib -L "$task_source" -Wl,-rpath,"$task_source" \
    -Wl,-rpath,"$INFINI_ROOT/lib" -Wl,-rpath,/opt/dtk/lib \
    -Wl,-rpath-link,/usr/local/lib/python3.11/site-packages/torch.libs \
    -l:core-h3-fusion.so -linfinicore_cpp_api -linfiniop -linfinirt -linfiniccl -lamdhip64 -ldl -Wl,-z,defs \
    -o "$task_dir/core-fusion3d-standalone" > "$task_dir/standalone-build.log" 2>&1
task_cpp=$?
if [ "$task_cpp" -eq 0 ]; then
    timeout 90 "$task_dir/core-fusion3d-standalone" \
        "$task_source/acceptance/kernels/h3_adaln3d_float16/h3_adaln3d_float16.native.so" \
        "$task_source/acceptance/kernels/h3_residual3d_float16/h3_residual3d_float16.native.so" > "$task_dir/standalone.log" 2>&1
    task_cpp=$?
fi
printf '%s\n' "$task_cpp" > "$task_dir/standalone.exit-code.txt"
task_python=1
export PYTHONPATH="$task_source/ninetoothed/src:$task_dir/task-source:$task_root/benchmarks"
if [ "$task_cpp" -eq 0 ]; then
    timeout 900 python -X faulthandler -u "$task_dir/task-source/hygon_fusion3d_benchmark.py" \
        --source "$task_source" --output "$task_dir/benchmark" > "$task_dir/run.log" 2>&1
    task_python=$?
fi
printf '%s\n' "$task_python" > "$task_dir/benchmark.exit-code.txt"
set -e
unset PYTHONPATH
python "$task_root/scripts/snapshot_environment.py" --output "$task_dir/environment.after.json" > "$task_dir/environment.after.log" 2>&1
if [ "$task_cpp" -eq 0 ] && [ "$task_python" -eq 0 ]; then task_status=0; else task_status=1; fi
printf '%s\n' "$task_status" > "$task_dir/exit-code.txt"
cat "$task_dir/standalone-build.log"
if [ -f "$task_dir/standalone.log" ]; then cat "$task_dir/standalone.log"; fi
if [ -f "$task_dir/run.log" ]; then tail -5 "$task_dir/run.log"; fi
exit "$task_status"
