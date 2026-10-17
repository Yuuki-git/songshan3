"""Copy the active development backend and add opt-in FMA control in isolation."""
import argparse
import difflib
import hashlib
import json
import os
from pathlib import Path
import shutil

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--directory", type=Path, required=True)
args = parser.parse_args()
root = Path(os.environ["SONGSHAN3_ROOT"]).resolve()
target = args.directory.resolve()
assert target.is_relative_to((root / "results").resolve()) and not target.exists()
source = root / "development/ninetoothed-hygon/src"
shutil.copytree(source, target / "src", ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
patches = target / "patches"
patches.mkdir()
edits = {}

relative = "ninetoothed/backends/triton.py"
text = (source / relative).read_text()
old = "class TritonBackend(Backend):\n    name = Target.TRITON\n"
new = old + '''    supported_options = frozenset({"enable_fp_fusion"})

    def normalize_options(self, options):
        result = super().normalize_options(options)
        if "enable_fp_fusion" in result and type(result["enable_fp_fusion"]) is not bool:
            raise TypeError("enable_fp_fusion must be a bool")
        return result

'''
assert text.count(old) == 1
edits[relative] = text.replace(old, new)

relative = "ninetoothed/backends/emitters/triton.py"
text = (source / relative).read_text()
old = "        num_stages=_ninetoothed_num_stages,\n"
new = old + "        enable_fp_fusion={kernel.compiler_options.get('backend_options', {}).get('enable_fp_fusion', True)!r},\n"
assert text.count(old) == 1
edits[relative] = text.replace(old, new)

relative = "ninetoothed/backends/materializers/hip.py"
text = (source / relative).read_text()
old = '    environment["TRITON_CACHE_DIR"] = str(TRITON_CACHE_DIR)\n'
new = old + '''    enable_fp_fusion = compilation.request.backend_options.get("enable_fp_fusion", True)
    environment["NINETOOTHED_ENABLE_FP_FUSION"] = "1" if enable_fp_fusion else "0"
'''
assert text.count(old) == 1
text = text.replace(old, new)
old = 'sys.executable, "-m", "triton.tools.compile", str(source),'
new = 'sys.executable, "-m", "ninetoothed.backends.materializers.hip_compile_options", str(source),'
assert text.count(old) == 1
edits[relative] = text.replace(old, new)
edits["ninetoothed/backends/materializers/hip_compile_options.py"] = (root / "development/hygon_compile_options.py").read_text()

diffs, manifest = [], {}
for relative, text in edits.items():
    original = (source / relative).read_text() if (source / relative).exists() else ""
    (target / "src" / relative).write_text(text)
    manifest[relative] = dict(original_sha256=hashlib.sha256(original.encode()).hexdigest(),
                              modified_sha256=hashlib.sha256(text.encode()).hexdigest())
    diffs.extend(difflib.unified_diff(original.splitlines(True), text.splitlines(True),
                                    fromfile="a/src/" + relative, tofile="b/src/" + relative))
(patches / "fp-fusion-control.patch").write_text("".join(diffs))
(patches / "manifest.json").write_text(json.dumps(dict(source=str(source), directory=str(target),
    base_commit="2c5eb118592f4d2b0f448d16d2e9a3a5c0fbc900", files=manifest,
    scope="opt-in compiler option; default True; both JIT and HIP AOT; no device kernel source edited"), indent=2) + "\n")
print(json.dumps(dict(directory=str(target), files=manifest), indent=2))
