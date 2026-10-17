#!/usr/bin/env bash
set -euo pipefail
task_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)
source "$task_root/scripts/setup_env.sh"
test -f "$task_root/venvs/h3/bin/activate"
test -f "$INFINI_ROOT/lib/libinfiniop.so"
task_dev="$task_root/development/ninetoothed-hygon"
task_source="$task_root/Projects/ninetoothed"
task_commit=2c5eb118592f4d2b0f448d16d2e9a3a5c0fbc900
test "$(git -C "$task_source" rev-parse HEAD)" = "$task_commit"
if [ ! -e "$task_dev/.git" ]; then
    if [ -e "$task_dev" ]; then printf 'Preserving existing non-Git backend: %s\n' "$task_dev" >&2; exit 1; fi
    git -c safe.directory="$task_source" clone --no-hardlinks --no-checkout "$task_source" "$task_dev"
    git -C "$task_dev" checkout --detach "$task_commit"
fi
test "$(git -C "$task_dev" rev-parse HEAD)" = "$task_commit"
python "$task_root/scripts/apply_hygon_aot_patch.py"
