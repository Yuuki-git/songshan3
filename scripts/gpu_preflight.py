"""Isolate GPU initialization failures before importing Core or starting a run."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

CODE = r"""
import json
import torch
p = torch.cuda.get_device_properties(0)
print(json.dumps(dict(stage='properties', torch=torch.__version__, hip=torch.version.hip,
                      name=p.name, architecture=p.gcnArchName, memory=p.total_memory)), flush=True)
x = torch.arange(8, device='cuda', dtype=torch.float32)
torch.cuda.synchronize()
assert x.cpu().tolist() == list(range(8))
print(json.dumps(dict(stage='allocation_kernel_copy', passed=True)), flush=True)
"""


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    try:
        process = subprocess.run([sys.executable, "-X", "faulthandler", "-u", "-c", CODE],
                                 capture_output=True, text=True, timeout=45)
        record = dict(passed=process.returncode == 0, returncode=process.returncode,
                      stdout=process.stdout, stderr=process.stderr)
    except subprocess.TimeoutExpired as error:
        def decode(value):
            return value.decode(errors="replace") if isinstance(value, bytes) else value
        record = dict(passed=False, timeout_seconds=45, stdout=decode(error.stdout), stderr=decode(error.stderr))
    (args.output / "preflight.json").write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record), flush=True)
    return int(not record["passed"])


if __name__ == "__main__":
    raise SystemExit(main())
