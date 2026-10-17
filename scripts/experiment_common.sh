#!/usr/bin/env bash
# Sourced by experiment launchers after setup_env.sh.
songshan3_begin() {
    local task_prefix="$1"
    mkdir -p "$task_root/results" "$task_root/cache"
    task_dir=$(mktemp -d "$task_root/results/$task_prefix-$(date -u +%Y%m%dT%H%M%SZ)-XXXX")
    chmod 0755 "$task_dir"
    mkdir -p "$task_dir/task-source"
    printf 'Experiment directory: %s\n' "$task_dir"
    trap 'songshan3_finish $?' EXIT
    if [ -n "${SONGSHAN3_STAGE_RESULT_FILE:-}" ]; then
        (set -C; printf '%s\n' "$task_dir" > "$SONGSHAN3_STAGE_RESULT_FILE")
    fi
    exec 9>"${SONGSHAN3_GPU_LOCK:-$task_root/cache/operator-experiment.lock}"
    flock -n 9 || { printf 'Another GPU experiment holds the lock.\n' >&2; exit 2; }
}

songshan3_finish() {
    local task_final_status="$1"
    trap - EXIT
    if [ -f "$task_dir/environment.before.json" ] && [ ! -f "$task_dir/environment.after.json" ]; then
        if ! env -u PYTHONPATH python "$task_root/scripts/snapshot_environment.py" \
            --output "$task_dir/environment.after.json" > "$task_dir/environment.after.log" 2>&1; then
            if [ "$task_final_status" -eq 0 ]; then task_final_status=1; fi
        fi
    fi
    printf '%s\n' "$task_final_status" > "$task_dir/exit-code.txt"
    exit "$task_final_status"
}
