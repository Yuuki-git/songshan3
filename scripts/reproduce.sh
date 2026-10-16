#!/usr/bin/env bash
# Build the host bridge and reproduce only the RMSNorm experiment.
set -euo pipefail
task_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)
if [ "$(basename -- "$task_root")" != songshan3 ]; then
    printf 'Clone this project into a directory named songshan3.\n' >&2
    exit 2
fi
export WORK_HOME
WORK_HOME=$(dirname -- "$task_root")
export SONGSHAN3_ROOT="$task_root"
task_options=()
task_skip_build=0
for task_arg in "$@"; do
    case "$task_arg" in
        --offline) task_options=(--offline) ;;
        --skip-build) task_skip_build=1 ;;
        *) printf 'Usage: bash scripts/reproduce.sh [--offline] [--skip-build]\n' >&2; exit 2 ;;
    esac
done
mkdir -p "$task_root"/{logs,manifests,results,cache,tooling,venvs}
if [ "$task_skip_build" -eq 0 ]; then
    bash "$task_root/scripts/prepare.sh" "${task_options[@]}"
fi
source "$task_root/scripts/setup_env.sh"
task_dev="$task_root/development/ninetoothed-hygon"
if [ ! -e "$task_dev/.git" ]; then
    git clone --no-hardlinks --no-checkout "$task_root/Projects/ninetoothed" "$task_dev"
    git -C "$task_dev" checkout --detach 2c5eb118592f4d2b0f448d16d2e9a3a5c0fbc900
fi
test "$(git -C "$task_dev" rev-parse HEAD)" = 2c5eb118592f4d2b0f448d16d2e9a3a5c0fbc900
task_dir=$(mktemp -d "$task_root/results/hygon-rmsnorm-native-$(date -u +%Y%m%dT%H%M%SZ)-XXXX")
printf 'Experiment directory: %s\n' "$task_dir"
export PYTHONPATH="$task_dev/src:$task_root/benchmarks${PYTHONPATH:+:$PYTHONPATH}"
export NINETOOTHED_CACHE_DIR="$task_dir/ninetoothed-cache"
export TRITON_CACHE_DIR="$task_dir/triton-cache"
python "$task_root/scripts/apply_hygon_aot_patch.py" > "$task_dir/patch.log"
g++ -shared -std=c++17 -O3 -fPIC -pthread "$task_root/development/core_ninetoothed_bridge.cpp" \
    -I "$INFINI_ROOT/include" -L "$INFINI_ROOT/lib" -Wl,-rpath,"$INFINI_ROOT/lib" \
    -Wl,-rpath-link,/usr/local/lib/python3.11/site-packages/torch.libs \
    -linfinicore_cpp_api -linfiniop -linfinirt -linfiniccl -ldl -Wl,-z,defs \
    -o "$task_dir/core-ninetoothed-bridge.so" > "$task_dir/build.log" 2>&1
set +e
timeout 1200 python -u "$task_root/benchmarks/hygon_rmsnorm_acceptance.py" \
    --bridge "$task_dir/core-ninetoothed-bridge.so" --output "$task_dir/acceptance" \
    > "$task_dir/run.log" 2>&1
task_status=$?
set -e
printf '%s\n' "$task_status" > "$task_dir/exit-code.txt"
tail -40 "$task_dir/run.log"
exit "$task_status"
