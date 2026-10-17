#!/usr/bin/env bash
set -euo pipefail
umask 022
task_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)
source "$task_root/scripts/setup_env.sh"
task_source="${SONGSHAN3_RMS3D_SOURCE:?Set SONGSHAN3_RMS3D_SOURCE to the successful main experiment from this run}"
test -f "$task_source/acceptance/acceptance.json"
test "$(cat "$task_source/exit-code.txt")" = 0
source "$task_root/scripts/experiment_common.sh"
songshan3_begin hygon-rms3d-extended
mkdir -p "$task_dir/task-source"
cp "$task_root/development/core_rms3d_standalone.cpp" "$task_root/benchmarks/hygon_rmsnorm3d_extended.py" "$task_dir/task-source/"
cp "$task_root/benchmarks/hygon_rmsnorm3d_graph_benchmark.py" "$task_root/scripts/run_hygon_rmsnorm3d_extended.sh" "$task_dir/task-source/"
printf '%s\n' "$task_source" > "$task_dir/source-experiment.txt"
python "$task_root/scripts/gpu_preflight.py" --output "$task_dir/preflight" > "$task_dir/preflight.log" 2>&1
python "$task_root/scripts/snapshot_environment.py" --output "$task_dir/environment.before.json" > "$task_dir/environment.before.log" 2>&1
g++ -std=c++17 -O2 -I/opt/dtk/include -I "$INFINI_ROOT/include" "$task_root/development/core_rms3d_standalone.cpp" \
    -L "$INFINI_ROOT/lib" -L/opt/dtk/lib -Wl,-rpath,"$INFINI_ROOT/lib" -Wl,-rpath,/opt/dtk/lib \
    -Wl,-rpath-link,/usr/local/lib/python3.11/site-packages/torch.libs \
    -linfinicore_cpp_api -linfiniop -linfinirt -linfiniccl -lamdhip64 -ldl -Wl,-z,defs \
    -o "$task_dir/core-rms3d-standalone" > "$task_dir/standalone-build.log" 2>&1
task_library="$task_source/acceptance/kernels/hygon_rms3d_float32_float32/hygon_rms3d_float32_float32.native.so"
set +e
timeout 60 "$task_dir/core-rms3d-standalone" "$task_source/core-ninetoothed-bridge-v2.so" "$task_library" hygon_rms3d_float32_float32 > "$task_dir/standalone.log" 2>&1
task_cpp=$?
printf '%s\n' "$task_cpp" > "$task_dir/standalone.exit-code.txt"
export PYTHONPATH="$task_root/development/ninetoothed-hygon/src:$task_root/benchmarks"
timeout 300 python -X faulthandler -u "$task_root/benchmarks/hygon_rmsnorm3d_extended.py" --source "$task_source" --output "$task_dir/extended" > "$task_dir/run.log" 2>&1
task_python=$?
printf '%s\n' "$task_python" > "$task_dir/extended.exit-code.txt"
task_graph=1
if [ "$task_cpp" -eq 0 ] && [ "$task_python" -eq 0 ]; then
    timeout 600 python -u "$task_root/benchmarks/hygon_rmsnorm3d_graph_benchmark.py" --source "$task_source" --output "$task_dir/hipgraph" > "$task_dir/hipgraph.log" 2>&1
    task_graph=$?
fi
printf '%s\n' "$task_graph" > "$task_dir/hipgraph.exit-code.txt"
set -e
unset PYTHONPATH
python "$task_root/scripts/snapshot_environment.py" --output "$task_dir/environment.after.json" > "$task_dir/environment.after.log" 2>&1
if [ "$task_cpp" -eq 0 ] && [ "$task_python" -eq 0 ] && [ "$task_graph" -eq 0 ]; then task_status=0; else task_status=1; fi
printf '%s\n' "$task_status" > "$task_dir/exit-code.txt"
cat "$task_dir/standalone.log"
tail -1 "$task_dir/run.log"
if [ -f "$task_dir/hipgraph.log" ]; then tail -1 "$task_dir/hipgraph.log"; fi
exit "$task_status"
