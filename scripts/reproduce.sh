#!/usr/bin/env bash
set -euo pipefail
export GIT_LFS_SKIP_SMUDGE=1
task_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)
task_suite=all
task_offline=()
task_skip_build=0
task_prepare_only=0
while [ "$#" -gt 0 ]; do
    case "$1" in
        --suite)
            if [ "$#" -lt 2 ]; then printf 'Missing --suite value.\n' >&2; exit 2; fi
            task_suite="$2"; shift 2 ;;
        --offline) task_offline=(--offline); shift ;;
        --skip-build) task_skip_build=1; shift ;;
        --prepare-only) task_prepare_only=1; shift ;;
        -h|--help)
            printf 'Usage: bash scripts/reproduce.sh [--suite all|rms2d|rms3d|fusion3d] [--offline] [--skip-build] [--prepare-only]\n'
            exit 0 ;;
        *) printf 'Unknown option: %s\n' "$1" >&2; exit 2 ;;
    esac
done
case "$task_suite" in all|rms2d|rms3d|fusion3d) ;; *) printf 'Unknown suite: %s\n' "$task_suite" >&2; exit 2 ;; esac
if [ "$task_prepare_only" -eq 1 ]; then
    bash "$task_root/scripts/prepare.sh" "${task_offline[@]}" --download-only
    exit 0
fi
if [ "$task_skip_build" -eq 0 ]; then
    bash "$task_root/scripts/prepare.sh" "${task_offline[@]}"
fi
source "$task_root/scripts/setup_env.sh"
mkdir -p "$task_root"/{results,cache,development}
task_dir=$(mktemp -d "$task_root/results/hygon-operators-$(date -u +%Y%m%dT%H%M%SZ)-XXXX")
printf 'Experiment directory: %s\n' "$task_dir"
trap 'task_status=$?; printf "%s\n" "$task_status" > "$task_dir/exit-code.txt"' EXIT
exec 9>"${SONGSHAN3_GPU_LOCK:-$task_root/cache/operator-experiment.lock}"
flock -n 9 || { printf 'Another GPU experiment holds the lock.\n' >&2; exit 2; }
task_dev="$task_root/development/ninetoothed-hygon"
if [ ! -e "$task_dev/.git" ]; then
    git clone --no-hardlinks --no-checkout "$task_root/Projects/ninetoothed" "$task_dev"
    git -C "$task_dev" checkout --detach 2c5eb118592f4d2b0f448d16d2e9a3a5c0fbc900
fi
test "$(git -C "$task_dev" rev-parse HEAD)" = 2c5eb118592f4d2b0f448d16d2e9a3a5c0fbc900
python "$task_root/scripts/apply_hygon_aot_patch.py" > "$task_dir/backend.log"

suite_environment() {
    task_output="$task_dir/$1"
    mkdir -p "$task_output"
    export PYTHONPATH="$task_dev/src:$task_root/benchmarks"
    export NINETOOTHED_CACHE_DIR="$task_output/ninetoothed-cache"
    export TRITON_CACHE_DIR="$task_output/triton-cache"
    printf 'Running %s; output: %s\n' "$1" "$task_output"
}

build_bridge() {
    g++ -shared -std=c++17 -O3 -fPIC -pthread "$task_root/development/$1" \
        -I "$INFINI_ROOT/include" -L "$INFINI_ROOT/lib" -Wl,-rpath,"$INFINI_ROOT/lib" \
        -Wl,-rpath-link,/usr/local/lib/python3.11/site-packages/torch.libs \
        -linfinicore_cpp_api -linfiniop -linfinirt -linfiniccl -ldl -Wl,-z,defs \
        -o "$task_output/$2" > "$task_output/build.log" 2>&1
}

if [ "$task_suite" = all ] || [ "$task_suite" = rms2d ]; then
    suite_environment rms2d
    build_bridge core_ninetoothed_bridge.cpp core-ninetoothed-bridge.so
    timeout 1200 python -u "$task_root/benchmarks/hygon_rmsnorm_acceptance.py" \
        --bridge "$task_output/core-ninetoothed-bridge.so" --output "$task_output/acceptance" \
        > "$task_output/run.log" 2>&1
    tail -1 "$task_output/run.log"
fi
if [ "$task_suite" = all ] || [ "$task_suite" = rms3d ]; then
    suite_environment rms3d
    build_bridge core_ninetoothed_bridge_v2.cpp core-ninetoothed-bridge-v2.so
    timeout 1500 python -u "$task_root/benchmarks/hygon_rmsnorm3d_acceptance.py" \
        --bridge "$task_output/core-ninetoothed-bridge-v2.so" --output "$task_output/acceptance" \
        > "$task_output/run.log" 2>&1
    tail -1 "$task_output/run.log"
    timeout 600 python -u "$task_root/benchmarks/hygon_rmsnorm3d_graph_benchmark.py" \
        --source "$task_output" --output "$task_output/hipgraph" > "$task_output/hipgraph.log" 2>&1
    tail -1 "$task_output/hipgraph.log"
fi
if [ "$task_suite" = all ] || [ "$task_suite" = fusion3d ]; then
    suite_environment fusion3d
    python "$task_root/scripts/prepare_hygon_fusion_backend.py" --directory "$task_output/ninetoothed" \
        > "$task_output/backend.log"
    export PYTHONPATH="$task_output/ninetoothed/src:$task_root/benchmarks"
    build_bridge core_h3_fusion.cpp core-h3-fusion.so
    timeout 1200 python -u "$task_root/benchmarks/hygon_fusion3d_acceptance.py" \
        --bridge "$task_output/core-h3-fusion.so" --output "$task_output/acceptance" \
        > "$task_output/run.log" 2>&1
    tail -1 "$task_output/run.log"
    timeout 900 python -u "$task_root/benchmarks/hygon_fusion3d_benchmark.py" \
        --source "$task_output" --output "$task_output/benchmark" > "$task_output/benchmark.log" 2>&1
    tail -1 "$task_output/benchmark.log"
fi
printf 'Completed %s; results: %s\n' "$task_suite" "$task_dir"
