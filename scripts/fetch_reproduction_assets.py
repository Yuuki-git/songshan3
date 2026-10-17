"""Fetch pinned official sources and byte-verified dependencies for reproduction.

This program uses only the Python standard library and Git. It does not import
PyTorch, activate DTK, install packages, or execute a GPU workload.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import urllib.parse
import urllib.request
import uuid


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def checked_path(root, relative):
    path = (root / relative).resolve()
    if not path.is_relative_to(root):
        raise RuntimeError(f"Input path escapes the project: {relative}")
    return path


def git(repo, *args, capture=False, check=True):
    return subprocess.run(
        ["git", "-c", f"safe.directory={repo}", "-c", "core.autocrlf=false", "-C", str(repo), *args],
        check=check, capture_output=capture, text=True, encoding="utf-8",
        timeout=900,
    )


def revision(repo):
    result = git(repo, "rev-parse", "HEAD", capture=True, check=False)
    return result.stdout.strip() if result.returncode == 0 else None


def fetch_repository(root, spec, offline):
    repo = checked_path(root, spec["destination"])
    if not (repo / ".git").exists():
        if offline:
            raise RuntimeError(f"Missing offline source: {repo}")
        if repo.exists() and any(repo.iterdir()):
            raise RuntimeError(f"Existing non-Git directory: {repo}")
        repo.mkdir(parents=True, exist_ok=True)
        git(repo, "init", "--quiet")
        git(repo, "config", "core.autocrlf", "false")
        git(repo, "remote", "add", "origin", spec["url"])
    current = revision(repo)
    if current is None:
        if offline:
            raise RuntimeError(f"Missing offline commit: {repo}")
        git(repo, "fetch", "--depth", "1", spec["url"], spec["commit"])
        git(repo, "checkout", "--detach", "FETCH_HEAD")
    elif current != spec["commit"]:
        raise RuntimeError(f"Source commit differs from the manifest: {repo} ({current})")

    submodules = spec.get("submodules", {})
    missing = [relative for relative, commit in submodules.items()
               if not (repo / relative / ".git").exists() or revision(repo / relative) != commit]
    if missing and offline:
        raise RuntimeError(f"Missing or mismatched offline submodules: {repo}: {missing}")
    if missing:
        # Initializing the recorded top-level modules recursively also restores
        # InfiniOps/.ci. Optional platform modules match the baseline defaults.
        top_level = [path for path in submodules
                     if not any(path.startswith(other + "/") for other in submodules if other != path)]
        git(repo, "submodule", "update", "--init", "--recursive", "--depth", "1", "--", *top_level)
    for relative, commit in submodules.items():
        if revision(repo / relative) != commit:
            raise RuntimeError(f"Submodule revision mismatch: {repo / relative}")
    status = git(repo, "status", "--porcelain=v1", capture=True).stdout.strip()
    if status:
        raise RuntimeError(f"Original source has local changes: {repo}")
    print(f"Verified {spec['name']} {spec['commit']}", flush=True)


def verify_file(path, record):
    if not path.is_file():
        return False
    if path.stat().st_size != record["bytes"] or sha256(path) != record["sha256"]:
        raise RuntimeError(f"Dependency checksum mismatch: {path}")
    return True


def request(url):
    return urllib.request.urlopen(
        urllib.request.Request(url, headers={"User-Agent": "songshan3-reproduction/1"}), timeout=120,
    )


def download(path, record, url, offline):
    if verify_file(path, record):
        return
    if offline:
        raise RuntimeError(f"Missing offline dependency: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    stage = path.with_name(path.name + ".download-" + uuid.uuid4().hex)
    print(f"Downloading {path.name}", flush=True)
    try:
        with request(url) as source, stage.open("xb") as output:
            for chunk in iter(lambda: source.read(1024 * 1024), b""):
                output.write(chunk)
        verify_file(stage, record)
        stage.replace(path)
    finally:
        stage.unlink(missing_ok=True)


def fetch_wheel(root, record, offline):
    path = checked_path(root, "wheels/" + record["filename"])
    if verify_file(path, record):
        return
    if offline:
        raise RuntimeError(f"Missing offline wheel: {path}")
    name, version = record["filename"].split("-", 2)[:2]
    metadata_url = f"https://pypi.org/pypi/{name}/{version}/json"
    with request(metadata_url) as response:
        metadata = json.load(response)
    matches = [item for item in metadata["urls"] if item["filename"] == record["filename"]]
    if len(matches) != 1 or matches[0]["digests"]["sha256"] != record["sha256"]:
        raise RuntimeError(f"Official PyPI metadata differs from the wheel manifest: {path.name}")
    url = matches[0]["url"]
    if urllib.parse.urlparse(url).hostname != "files.pythonhosted.org":
        raise RuntimeError(f"Unexpected official wheel host for {path.name}")
    download(path, record, url, offline)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--offline", action="store_true", help="Only verify existing inputs; perform no downloads")
    parser.add_argument("--assets-only", action="store_true", help="Fetch/verify wheels and archives without Git sources")
    parser.add_argument("--sources-only", action="store_true", help="Fetch/verify Git sources without wheels and archives")
    args = parser.parse_args()
    if args.assets_only and args.sources_only:
        parser.error("--assets-only and --sources-only cannot be combined")
    root = args.root.resolve()
    spec = json.loads((root / "manifests/upstreams.json").read_text(encoding="utf-8"))
    wheels = json.loads((root / "manifests/workstation-wheels.json").read_text(encoding="utf-8"))["wheels"]
    if not args.assets_only:
        for repo in spec["repositories"]:
            fetch_repository(root, repo, args.offline)
    if not args.sources_only:
        for record in spec["archives"]:
            download(checked_path(root, record["path"]), record, record["url"], args.offline)
        for record in wheels:
            fetch_wheel(root, record, args.offline)
        print(f"Verified {len(spec['archives'])} archives and {len(wheels)} exact wheels", flush=True)
    print("Reproduction inputs ready", flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (RuntimeError, OSError, subprocess.SubprocessError, ValueError) as error:
        print(f"Input preparation failed: {error}", file=sys.stderr)
        raise SystemExit(1)
