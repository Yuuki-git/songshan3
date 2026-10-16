"""Opt-in, host-only Triton 3.8.0 property ABI fix for this DTK installation."""
import difflib
import hashlib
import json
import os
from pathlib import Path

import triton


def digest(data):
    return hashlib.sha256(data).hexdigest()


root = Path(os.environ["SONGSHAN3_ROOT"])
probe = json.loads((root / "manifests/hip-legacy-properties.json").read_text())
if triton.__version__ != "3.8.0" or not probe["arch"].startswith("gfx906"):
    raise RuntimeError("This patch is restricted to the probed Triton 3.8.0 / DTK gfx906 environment")
if not Path("/opt/dtk/lib/libgalaxyhip.so").exists():
    raise RuntimeError("DTK's vendor HIP library was not found")
source = Path(triton.__file__).parent / "backends/amd/driver.c"
patch_dir = root / "patches/triton-3.8.0-dtk-property-abi"
patch_dir.mkdir(parents=True, exist_ok=True)
original_path = patch_dir / "driver.c.original"
current = source.read_bytes()
if original_path.exists():
    original = original_path.read_bytes()
else:
    original = current
text = original.decode()
replacements = [
    ("#include <hip/hip_runtime_api.h>",
     "#include <hip/hip_runtime_api.h>\n#include <hip/hip_deprecated.h>\n"
     "// This environment opts into DTK's unversioned legacy device-property ABI."),
    ("hipDeviceProp_t *prop", "hipDeviceProp_tR0000 *prop"),
    ("hipDeviceProp_t props;", "hipDeviceProp_tR0000 props = {0};"),
]
for old, new in replacements:
    if text.count(old) != 1:
        raise RuntimeError(f"Unexpected upstream source: expected one occurrence of {old!r}")
    text = text.replace(old, new)
patched = text.encode()
if current not in (original, patched):
    raise RuntimeError("Refusing to overwrite an unrelated driver modification")
if not original_path.exists():
    original_path.write_bytes(original)
source.write_bytes(patched)
(patch_dir / "driver.patch").write_text("".join(difflib.unified_diff(
    original.decode().splitlines(keepends=True), text.splitlines(keepends=True),
    fromfile="a/triton/backends/amd/driver.c", tofile="b/triton/backends/amd/driver.c")))
record = dict(triton=triton.__version__, original_sha256=digest(original),
              patched_sha256=digest(patched), installed_path=str(source),
              scope="host device-property ABI only; no device code or model arithmetic changes",
              property_probe=probe)
(patch_dir / "manifest.json").write_text(json.dumps(record, indent=2) + "\n")
print(json.dumps(record, indent=2))
