"""Reproduce the operator suites serially, using only results from this run."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import signal
import subprocess
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]
SUITES = {
    "rms2d": [("rms2d", "run_hygon_rmsnorm.sh", None)],
    "rms3d": [("rms3d", "run_hygon_rmsnorm3d.sh", None),
              ("rms3d_extended", "run_hygon_rmsnorm3d_extended.sh", "SONGSHAN3_RMS3D_SOURCE")],
    "fusion3d": [("fusion3d", "run_hygon_fusion3d_core.sh", None),
                 ("fusion3d_extended", "run_hygon_fusion3d_extended.sh", "SONGSHAN3_FUSION_SOURCE")],
}

def utc():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def write_json(path, value):
    stage = path.with_suffix(path.suffix + ".tmp")
    stage.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    stage.replace(path)

def plan(suite):
    return [item for name in (SUITES if suite == "all" else [suite]) for item in SUITES[name]]

def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--suite", choices=["all", *SUITES], default="all")
    p.add_argument("--offline", action="store_true", help="Verify prepared inputs without downloading")
    p.add_argument("--prepare-only", action="store_true", help="Fetch/verify only; no DTK, install, build or GPU")
    p.add_argument("--skip-build", "--skip-bootstrap", dest="skip_build", action="store_true",
                   help="Reuse this checkout's independently prepared venv and Core libraries")
    p.add_argument("--list", action="store_true", help="Print the plan without GPU or downloads")
    return p

def git_identity(root):
    def query(*args):
        p = subprocess.run(["git", "-C", str(root), *args], capture_output=True, text=True)
        return p.stdout.strip() if p.returncode == 0 else None
    return {"commit": query("rev-parse", "HEAD"), "status": query("status", "--porcelain=v1")}

def execute_stage(root, run_dir, record, name, argv, env, expects_result=False):
    result_file = run_dir / (name + ".result-path")
    existing_results = {p.resolve() for p in (root / "results").iterdir()} if expects_result else set()
    stage = {"name": name, "command": argv, "started": utc(), "status": "running",
             "log": str(run_dir / (name + ".log"))}
    record["stages"].append(stage)
    write_json(run_dir / "run.json", record)
    stage_env = dict(env)
    if expects_result:
        stage_env["SONGSHAN3_STAGE_RESULT_FILE"] = str(result_file)
    print(f"Running {name}; log: {stage['log']}", flush=True)
    try:
        with Path(stage["log"]).open("w", encoding="utf-8") as log:
            process = subprocess.Popen(argv, cwd=root, env=stage_env, stdout=log, stderr=subprocess.STDOUT,
                                       start_new_session=os.name == "posix")
            code = process.wait()
    except KeyboardInterrupt:
        if "process" in locals() and process.poll() is None:
            if os.name == "posix":
                os.killpg(process.pid, signal.SIGTERM)
            else:
                process.terminate()
            process.wait(timeout=15)
        stage.update(status="interrupted", exit_code=130, finished=utc())
        write_json(run_dir / "run.json", record)
        raise
    except OSError as error:
        stage["error"] = str(error)
        code = 1
    if expects_result and result_file.exists():
        result = Path(result_file.read_text(encoding="utf-8").strip()).resolve()
        if result.parent != (root / "results").resolve() or result == run_dir or result in existing_results:
            stage["error"] = "Experiment output is outside this checkout or was already present before this stage"
            code = code or 1
        else:
            stage["result"] = str(result)
            exit_file = result / "exit-code.txt"
            if not exit_file.is_file() or exit_file.read_text().strip() != "0":
                stage["error"] = "Experiment did not record successful completion"
                code = code or 1
    elif expects_result and code == 0:
        stage["error"] = "Experiment did not report a newly created output directory"
        code = 1
    stage.update(status="passed" if code == 0 else "failed", exit_code=code, finished=utc())
    write_json(run_dir / "run.json", record)
    if code:
        print(f"Failed {name} (exit {code}); inspect {stage['log']}", file=sys.stderr, flush=True)
    else:
        print(f"Passed {name}" + (f"; result: {stage['result']}" if "result" in stage else ""), flush=True)
    return code, stage.get("result")

def main(argv=None):
    args = parser().parse_args(argv)
    if args.prepare_only and args.skip_build:
        parser().error("--prepare-only and --skip-build cannot be combined")
    stages = plan(args.suite)
    if args.list:
        print(json.dumps({"suite": args.suite, "stages": [s[0] for s in stages], "model_end_to_end": False}, indent=2))
        return 0
    if not args.prepare_only and (platform.system() != "Linux" or platform.machine() != "x86_64"):
        parser().error("Execution requires Linux x86_64/Hygon gfx906; --prepare-only works on other hosts")
    (ROOT / "results").mkdir(exist_ok=True)
    run_dir = Path(tempfile.mkdtemp(prefix="reproduction-" + time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()) + "-", dir=ROOT / "results"))
    files = [line.strip() for line in (ROOT / "manifests/reproduction-files.txt").read_text().splitlines() if line.strip() and not line.startswith("#")]
    record = {"schema": "songshan3-reproduction-v1", "started": utc(), "status": "running", "suite": args.suite,
              "prepare_only": args.prepare_only, "offline": args.offline, "skip_build": args.skip_build,
              "root": str(ROOT), "repository": git_identity(ROOT),
              "source_sha256": {name: digest(ROOT / name) for name in files}, "stages": [],
              "scope": "single-GPU operators and pointwise subgraphs; not complete H3 FL2VA"}
    write_json(run_dir / "run.json", record)
    print(f"Reproduction directory: {run_dir}", flush=True)
    env = dict(os.environ, SONGSHAN3_ROOT=str(ROOT), WORK_HOME=str(ROOT.parent), PYTHONUNBUFFERED="1")
    fetch = [sys.executable, str(ROOT / "scripts/fetch_reproduction_assets.py")]
    if args.offline:
        fetch.append("--offline")
    code = 1
    try:
        code, _ = execute_stage(ROOT, run_dir, record, "inputs", fetch, env)
        if code == 0 and not args.prepare_only:
            if not args.skip_build:
                code, _ = execute_stage(ROOT, run_dir, record, "build", ["bash", "scripts/prepare.sh", "--offline"], env)
            if code == 0:
                code, _ = execute_stage(ROOT, run_dir, record, "backend", ["bash", "scripts/prepare_ninetoothed.sh"], env)
            previous = None
            for name, script, source_env in stages:
                if code:
                    break
                stage_env = dict(env)
                if source_env:
                    if not previous:
                        raise RuntimeError("Missing successful main experiment in this invocation")
                    stage_env[source_env] = previous
                code, previous = execute_stage(ROOT, run_dir, record, name, ["bash", "scripts/" + script], stage_env, expects_result=True)
            if code == 0:
                code, _ = execute_stage(ROOT, run_dir, record, "summary", [sys.executable, "scripts/summarize_reproduction.py", "--run", str(run_dir / "run.json")], env)
    except KeyboardInterrupt:
        code = 130
    except Exception as error:
        record["error"] = str(error)
        print(str(error), file=sys.stderr)
        code = 1
    finally:
        record.update(status=("prepared" if args.prepare_only else "passed") if code == 0 else "failed", exit_code=code, finished=utc())
        write_json(run_dir / "run.json", record)
        (run_dir / "exit-code.txt").write_text(str(code) + "\n")
    print(f"Reproduction {record['status']}; record: {run_dir / 'run.json'}", flush=True)
    return code

if __name__ == "__main__":
    raise SystemExit(main())
