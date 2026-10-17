"""Record the current live environment, source revisions, and loaded libraries."""
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def command(argv):
    completed = subprocess.run(argv, capture_output=True, text=True, check=False)
    return dict(exit_code=completed.returncode, stdout=completed.stdout,
                stderr=completed.stderr)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = Path(os.environ["SONGSHAN3_ROOT"])
    record = dict(utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                  pod=platform.node(), python=sys.version, executable=sys.executable,
                  root=str(root), prefix=os.environ["INFINI_ROOT"], sources={})
    for name in ("InfiniCore", "InfiniLM", "ninetoothed"):
        source = root / "Projects" / name
        prefix = ["git", "-c", f"safe.directory={source}", "-C", str(source)]
        record["sources"][name] = {
            "commit": command(prefix + ["rev-parse", "HEAD"]),
            "status": command(prefix + ["status", "--porcelain=v1"]),
            "submodules": command(prefix + ["submodule", "status", "--recursive"]),
        }
    for name in ("torch", "numpy", "triton", "ninetoothed", "infinicore", "infinilm"):
        try:
            module = __import__(name)
            record[name] = dict(path=module.__file__,
                                version=getattr(module, "__version__", None))
        except Exception as error:
            record[name] = dict(error=repr(error))
    import torch
    p = torch.cuda.get_device_properties(0)
    record["gpu"] = dict(name=p.name, memory_bytes=p.total_memory,
                          architecture=getattr(p, "gcnArchName", None),
                          torch_hip=torch.version.hip, devices=torch.cuda.device_count())
    record["libraries"] = {}
    for library in sorted((root / "infini/lib").glob("*.so*")):
        if library.is_file():
            record["libraries"][str(library)] = sha256(library)
    maps = Path("/proc/self/maps").read_text().splitlines()
    record["loaded_infini_libraries"] = sorted({line.split()[-1] for line in maps
                                                if "/libinfini" in line})
    record["packages"] = command([sys.executable, "-m", "pip", "freeze", "--all"])
    record["global_packages"] = command([os.environ.get("SONGSHAN3_BASE_PYTHON", "/usr/local/bin/python"), "-m", "pip", "freeze", "--all"])
    before = root / "manifests/global-python.before.txt"
    record["global_package_list_unchanged"] = (
        before.read_text() == record["global_packages"]["stdout"]) if before.is_file() else None
    record["benchmark_sources"] = {str(path.relative_to(root)): sha256(path)
                                    for path in sorted((root / "benchmarks").glob("*"))
                                    if path.is_file()}
    args.output.write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps({"output": str(args.output), "gpu": record["gpu"],
                      "global_package_list_unchanged": record["global_package_list_unchanged"],
                      "loaded_libraries": record["loaded_infini_libraries"]}, indent=2))


if __name__ == "__main__":
    main()
