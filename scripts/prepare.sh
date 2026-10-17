#!/usr/bin/env bash
# Prepare exactly the Core/NineToothed environment required by RMSNorm.
set -euo pipefail
export GIT_LFS_SKIP_SMUDGE=1
task_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)
export WORK_HOME
WORK_HOME=$(dirname -- "$task_root")
export SONGSHAN3_ROOT="$task_root"
task_offline=0
task_download_only=0
for task_arg in "$@"; do
    case "$task_arg" in
        --offline) task_offline=1 ;;
        --download-only) task_download_only=1 ;;
        *) printf 'Usage: bash scripts/prepare.sh [--offline] [--download-only]\n' >&2; exit 2 ;;
    esac
done
mkdir -p "$task_root"/{Projects,downloads/cpp,wheels,tooling,logs,manifests,results,venvs}
exec > >(tee -a "$task_root/logs/prepare.log") 2>&1

fetch_repo() {
    local task_url="$1" task_commit="$2" task_path="$3"
    if [ ! -e "$task_path/.git" ]; then
        if [ "$task_offline" -eq 1 ]; then printf 'Missing source: %s\n' "$task_path" >&2; return 1; fi
        if [ -d "$task_path" ] && [ -n "$(ls -A "$task_path")" ]; then
            printf 'Preserving existing non-Git directory: %s\n' "$task_path" >&2
            return 1
        fi
        mkdir -p "$task_path"
        git -C "$task_path" init --quiet
        git -C "$task_path" config core.autocrlf false
        git -C "$task_path" remote add origin "$task_url"
    fi
    local task_current
    task_current=$(git -C "$task_path" rev-parse --verify HEAD 2>/dev/null || true)
    if [ -z "$task_current" ] && [ "$task_offline" -eq 0 ]; then
        git -C "$task_path" fetch --depth 1 origin "$task_commit"
        git -C "$task_path" checkout --detach FETCH_HEAD
    fi
    test "$(git -C "$task_path" rev-parse HEAD)" = "$task_commit"
    git -c core.autocrlf=false -C "$task_path" diff --quiet
    if [ "$task_offline" -eq 0 ]; then
        git -c core.autocrlf=false -C "$task_path" submodule update --init --recursive --depth 1
    fi
    if git -C "$task_path" submodule status --recursive | grep -q '^[-+U]'; then
        # Optional Core modules have update=none and were unused in the baseline.
        if git -C "$task_path" submodule status --recursive | grep '^[-+U]' \
            | grep -vE ' third_party/(cutlass|flash-attention|mate)( |$)'; then
            printf 'A required submodule is missing or differs from the recorded commit.\n' >&2
            return 1
        fi
    fi
}

fetch_archive() {
    local task_url="$1" task_file="$2" task_digest="$3"
    if [ ! -f "$task_file" ]; then
        if [ "$task_offline" -eq 1 ]; then printf 'Missing dependency: %s\n' "$task_file" >&2; return 1; fi
        curl --fail --location --retry 3 "$task_url" -o "$task_file.download"
        printf '%s  %s\n' "$task_digest" "$task_file.download" | sha256sum -c -
        mv -- "$task_file.download" "$task_file"
    fi
    printf '%s  %s\n' "$task_digest" "$task_file" | sha256sum -c -
}

fetch_repo https://github.com/InfiniTensor/InfiniCore.git \
    f32e2fbb2a9b2ca912b5711f4f3836c403196a74 "$task_root/Projects/InfiniCore"
fetch_repo https://github.com/InfiniTensor/ninetoothed.git \
    2c5eb118592f4d2b0f448d16d2e9a3a5c0fbc900 "$task_root/Projects/ninetoothed"
fetch_repo https://github.com/xmake-io/xmake-repo.git \
    bd29d77f242e247589457638272d9936a24a3fe0 "$task_root/tooling/xmake-repo"
fetch_archive https://github.com/xmake-io/xmake/releases/download/v3.0.5/xmake-v3.0.5.tar.gz \
    "$task_root/downloads/xmake-v3.0.5.tar.gz" b947666281222f79e082283b6f84e68880c499305890f6ab8b03b8bac82456dc
fetch_archive https://github.com/boostorg/boost/releases/download/boost-1.92.0/boost-1.92.0-cmake.tar.gz \
    "$task_root/downloads/cpp/boost-1.92.0-cmake.tar.gz" f51707c27359a0df0cac1beada86de31bb5eed5e8285592dadec384df99c2984
fetch_archive https://github.com/pybind/pybind11/archive/refs/tags/v3.1.0.zip \
    "$task_root/downloads/cpp/v3.1.0.zip" affea1ada7b39fe1d835559fcb78800c7927d514bd480ed01b71b88205a5e536

if [ "$task_offline" -eq 0 ]; then
    PIP_NO_INDEX=0 PIP_INDEX_URL=https://pypi.org/simple "${PYTHON_BIN:-python}" -m pip download \
        --only-binary=:all: --no-deps --dest "$task_root/wheels" \
        --platform manylinux_2_17_x86_64 --platform manylinux2014_x86_64 \
        --platform manylinux_2_27_x86_64 --platform manylinux_2_28_x86_64 \
        --implementation cp --python-version 311 --abi cp311 --abi abi3 --abi none \
        -r "$task_root/requirements.txt"
fi
if [ "$task_download_only" -eq 1 ]; then exit 0; fi

source "$task_root/scripts/setup_env.sh"
python - <<'PY'
import platform, sys, torch
assert platform.system() == 'Linux' and platform.machine() == 'x86_64', 'Linux x86_64 is required'
assert sys.version_info[:2] == (3, 11), 'Use the baseline Python 3.11 environment'
assert torch.__version__.split('+')[0] == '2.9.0', 'Use Hygon PyTorch 2.9.0 from the base image'
assert torch.version.hip and torch.version.hip.startswith('6.3.'), 'Use the baseline HIP 6.3 environment'
assert torch.cuda.is_available(), 'A visible Hygon gfx906 device is required'
print('Runtime:', sys.version, torch.__version__, torch.version.hip, torch.cuda.get_device_name(0))
PY
if [ ! -f "$task_root/venvs/h3/bin/activate" ]; then
    python -m venv --system-site-packages "$task_root/venvs/h3"
fi
source "$task_root/venvs/h3/bin/activate"
python -m pip install --no-index --find-links "$task_root/wheels" --no-deps --ignore-installed -r "$task_root/requirements.txt"
python -m pip install -e "$task_root/Projects/ninetoothed" --no-deps --no-build-isolation
python -m pip check

task_amd=$(python -c 'from pathlib import Path; import triton; print(Path(triton.__file__).parent / "backends/amd")')
g++ -std=c++17 -I "$task_amd/include" "$task_root/benchmarks/probe_hip_properties.cpp" \
    -ldl -o "$task_root/tooling/probe-hip-properties"
"$task_root/tooling/probe-hip-properties" | tee "$task_root/manifests/hip-legacy-properties.json"
python "$task_root/scripts/patch_triton_dtk.py"

if [ ! -x "$task_root/tooling/xmake-3.0.5/bin/xmake" ]; then
    mkdir -p "$task_root/tooling/xmake-source"
    tar -xzf "$task_root/downloads/xmake-v3.0.5.tar.gz" --strip-components=1 -C "$task_root/tooling/xmake-source"
    (cd "$task_root/tooling/xmake-source"
     bash ./configure --prefix="$task_root/tooling/xmake-3.0.5"
     make -j 8
     make install)
fi
source "$task_root/scripts/setup_env.sh"
ln -sf boost-1.92.0-cmake.tar.gz "$task_root/downloads/cpp/boost-1.92.0.tar.gz"
ln -sf v3.1.0.zip "$task_root/downloads/cpp/pybind11-v3.1.0.zip"
xmake global --network=private --pkg_searchdirs="$task_root/downloads/cpp"
(
    cd "$task_root/Projects/InfiniCore"
    xmake repo --add songshan3-official "$task_root/tooling/xmake-repo"
    xmake f -y -cv --hygon-dcu=y --hygon-arch=gfx906 --cuda=/opt/dtk/cuda/cuda --ccl=y --aten=y
    xmake build -j 8
    xmake install
    xmake build -j 8 _infinicore
    xmake install _infinicore
    python -m pip install -e . --no-deps --no-build-isolation
)
python -m pip check
python -c 'import torch, infinicore, ninetoothed; print(torch.__version__, infinicore.__file__, ninetoothed.__file__)'
