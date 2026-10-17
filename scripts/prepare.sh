#!/usr/bin/env bash
# Build the pinned Core baseline in this checkout; source inputs stay unmodified.
set -euo pipefail
task_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)
task_offline=()
task_download_only=0
for task_arg in "$@"; do
    case "$task_arg" in
        --offline) task_offline=(--offline) ;;
        --download-only|--prepare-only) task_download_only=1 ;;
        *) printf 'Usage: bash scripts/prepare.sh [--offline] [--download-only]\n' >&2; exit 2 ;;
    esac
done
mkdir -p "$task_root"/{downloads/cpp,wheels,tooling,logs,manifests,results,cache,venvs}
exec > >(tee -a "$task_root/logs/prepare.log") 2>&1
"${PYTHON_BIN:-python}" "$task_root/scripts/fetch_reproduction_assets.py" "${task_offline[@]}"
if [ "$task_download_only" -eq 1 ]; then exit 0; fi
source "$task_root/scripts/setup_env.sh"
"$SONGSHAN3_BASE_PYTHON" - <<'PY'
import platform, sys, torch
assert platform.system() == 'Linux' and platform.machine() == 'x86_64', 'Linux x86_64 is required'
assert sys.version_info[:2] == (3, 11), 'Use baseline Python 3.11'
assert torch.__version__.split('+')[0] == '2.9.0', 'Use the image Hygon PyTorch 2.9.0'
assert torch.version.hip and torch.version.hip.startswith('6.3.'), 'Use baseline HIP 6.3'
print('Base runtime:', sys.version, torch.__version__, torch.version.hip, torch.__file__)
PY
"$SONGSHAN3_BASE_PYTHON" - <<'PY'
import os, subprocess, sys
from pathlib import Path
root=Path(os.environ['SONGSHAN3_ROOT'])
freeze=subprocess.check_output([sys.executable,'-m','pip','freeze','--all'],text=True)
path=root/'manifests/global-python.before.txt'
if path.exists() and path.read_text()!=freeze:
    raise RuntimeError('Base-image packages differ from this checkout\'s initial snapshot')
if not path.exists():path.write_text(freeze)
PY
if [ ! -f "$task_root/venvs/h3/bin/activate" ]; then
    "$SONGSHAN3_BASE_PYTHON" -m venv --system-site-packages "$task_root/venvs/h3"
fi
source "$task_root/venvs/h3/bin/activate"
python -m pip install --no-index --find-links "$task_root/wheels" --no-deps --ignore-installed \
    --require-hashes -r "$task_root/requirements-lock.txt"
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
    xmake f -y -cv --hygon-dcu=y --hygon-arch=gfx906 --cuda=/opt/dtk/cuda/cuda --ccl=y --aten=y --graph=n
    xmake build -j 8
    xmake install
    xmake build -j 8 _infinicore
    xmake install _infinicore
    python -m pip install -e . --no-deps --no-build-isolation
)
python -m pip check
python -c 'import torch,infinicore,ninetoothed; print(torch.__version__,infinicore.__file__,ninetoothed.__file__)'
