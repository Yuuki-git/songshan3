"""Install the reviewable host adapter only into this round's development copy."""
import difflib
import hashlib
import json
import os
from pathlib import Path

root = Path(os.environ["SONGSHAN3_ROOT"]).resolve()
baseline = root / "Projects/ninetoothed"
development = root / "development/ninetoothed-hygon"
patch_dir = root / "patches/ninetoothed-hygon-aot"
patch_dir.mkdir(parents=True, exist_ok=True)
edits = {}

relative = "src/ninetoothed/backends/materializers/triton.py"
text = (baseline / relative).read_text()
old = "def _compile_aot_library(compilation, source: Path, library: Path) -> None:\n    _ensure_c_compiler()"
new = "def _compile_aot_library(compilation, source: Path, library: Path) -> None:\n    from .hip import is_hip, compile_library\n\n    if is_hip(compilation):\n        return compile_library(compilation, source, library)\n\n    _ensure_c_compiler()"
assert text.count(old) == 1
edits[relative] = text.replace(old, new)

relative = "src/ninetoothed/targets.py"
text = (baseline / relative).read_text()
old = '        PlatformProfile(\n            name="hygon-bw1000",'
new = '''        PlatformProfile(
            name="hygon-gfx906",
            accelerator_vendor="hygon",
            accelerator_name="z200sm-80",
            compute_arch="gfx906",
            device_types=("cuda",),
            backend_modes=_backend_modes((Target.TRITON,)),
            metadata={"triton_aot_driver": "hip", "wavefront_size": 64},
        ),
''' + old
assert text.count(old) == 1
edits[relative] = text.replace(old, new)

relative = "src/ninetoothed/backends/materializers/cpp.py"
text = (baseline / relative).read_text()
old = '''    return (
        '#include "ninetoothed.h"\\n#include <cuda.h>\\n#include <stdint.h>\\n\\n'
'''
new = old.replace("    return (", "    source = (")
assert text.count(old) == 1
text = text.replace(old, new)
old = '\n\ndef _configuration_value(value):'
new = '''
    from .hip import is_hip, host_source
    return host_source(source) if is_hip(compilation) else source


def _configuration_value(value):'''
assert text.count(old) == 1
text = text.replace(old, new)
old = '    directory = Path(output_dir)\n    write_header(directory)'
new = '''    from .hip import is_hip, host_source
    if is_hip(variants[0][1]):
        source = host_source(source)
    directory = Path(output_dir)
    write_header(directory)'''
assert text.count(old) == 1
edits[relative] = text.replace(old, new)

edits["src/ninetoothed/backends/materializers/hip.py"] = (root / "development/hygon_aot.py").read_text()
diffs, manifest = [], {}
for relative, text in edits.items():
    original = (baseline / relative).read_text() if (baseline / relative).is_file() else ""
    target = development / relative
    assert target.resolve().is_relative_to(development.resolve())
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(text)
    manifest[relative] = dict(
        original_sha256=hashlib.sha256(original.encode()).hexdigest(),
        modified_sha256=hashlib.sha256(text.encode()).hexdigest(),
    )
    diffs.extend(difflib.unified_diff(original.splitlines(True), text.splitlines(True),
                                    fromfile=f"a/{relative}", tofile=f"b/{relative}"))
(patch_dir / "backend.patch").write_text("".join(diffs))
(patch_dir / "manifest.json").write_text(json.dumps(dict(
    base_commit="2c5eb118592f4d2b0f448d16d2e9a3a5c0fbc900",
    development=str(development), scope="HIP host AOT adapter; NineToothed-generated device computation",
    files=manifest), indent=2) + "\n")
print(json.dumps(dict(patched=str(development), files=list(edits)), indent=2))
