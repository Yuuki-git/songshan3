#!/usr/bin/env bash
set -euo pipefail
task_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)
exec "${PYTHON_BIN:-python}" "$task_root/scripts/reproduce.py" "$@"
