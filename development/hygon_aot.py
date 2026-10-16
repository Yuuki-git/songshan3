"""HIP host materializer for NineToothed-generated Triton artifacts.

This module contains host compilation, module loading and launch ABI adaptation
only. Device code is emitted by NineToothed and compiled by Triton unchanged.
"""
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile


def is_hip(compilation):
    target = getattr(compilation, "target", None)
    profile = getattr(target, "platform", None)
    return getattr(profile, "metadata", {}).get("triton_aot_driver") == "hip"


def host_source(source):
    """Translate only the public launcher wrapper's driver API types."""
    for old, new in (
        ("#include <cuda.h>", "#define __HIP_PLATFORM_AMD__\n#include <hip/hip_runtime.h>"),
        ("CUDA_ERROR_INVALID_VALUE", "hipErrorInvalidValue"),
        ("CUDA_SUCCESS", "hipSuccess"),
        ("CUdeviceptr", "hipDeviceptr_t"),
        ("CUstream", "hipStream_t"),
        ("CUresult", "hipError_t"),
    ):
        source = source.replace(old, new)
    return source


def kernel_names(sources):
    patterns = (
        r"^hipModule_t ([A-Za-z_][A-Za-z0-9_]*)_mod = NULL;$",
        r"^hipFunction_t ([A-Za-z_][A-Za-z0-9_]*)_func = NULL;$",
        r"^void load_([A-Za-z_][A-Za-z0-9_]*)\(\) \{$",
    )
    names = []
    for source in sources:
        groups = [re.findall(pattern, source.read_text(), re.MULTILINE) for pattern in patterns]
        if not groups[0] or any(len(group) != len(set(group)) for group in groups):
            raise ValueError(f"Unsupported HIP AOT source: {source}")
        if any(set(group) != set(groups[0]) for group in groups[1:]):
            raise ValueError(f"Mismatched HIP module/function/loader symbols: {source}")
        names.extend(groups[0])
    if len(names) != len(set(names)):
        raise ValueError("Duplicate HIP AOT kernel symbols")
    return tuple(names)


def context_guard_source(names):
    declarations = "\n".join(
        f'extern "C" hipModule_t {name}_mod;\n'
        f'extern "C" hipFunction_t {name}_func;\n'
        f'extern "C" unsigned char {name}_hsaco[];'
        for name in names
    )
    loads = "\n".join(
        f"            result = hipModuleLoadData(&state.modules[{i}], {name}_hsaco);\n"
        f"            if (result == hipSuccess) result = hipModuleGetFunction(\n"
        f'                &state.functions[{i}], state.modules[{i}], "{name.rsplit("_", 2)[0]}");\n'
        "            if (result != hipSuccess) {\n"
        "                unload(state);\n"
        "                launch_mutex.unlock();\n"
        "                return result;\n"
        "            }"
        for i, name in enumerate(names)
    )
    restores = "\n".join(
        f"        {name}_mod = found->second.modules[{i}];\n"
        f"        {name}_func = found->second.functions[{i}];"
        for i, name in enumerate(names)
    )
    resets = "\n".join(f"    {name}_mod = nullptr; {name}_func = nullptr;" for name in names)
    return f'''#include <array>
#include <mutex>
#include <unordered_map>
#define __HIP_PLATFORM_AMD__
#include <hip/hip_runtime.h>

{declarations}
namespace {{
struct State {{
    std::array<hipModule_t, {len(names)}> modules{{}};
    std::array<hipFunction_t, {len(names)}> functions{{}};
    int primary_device = -1;
}};
std::mutex launch_mutex;
std::unordered_map<hipCtx_t, State> context_states;
hipError_t unload(State &state) {{
    hipError_t first = hipSuccess;
    for (auto module : state.modules) {{
        if (module) {{
            auto error = hipModuleUnload(module);
            if (first == hipSuccess) first = error;
        }}
    }}
    if (state.primary_device >= 0) {{
        auto error = hipDevicePrimaryCtxRelease(state.primary_device);
        if (first == hipSuccess) first = error;
    }}
    return first;
}}
// DTK's runtime uses an implicit primary context while GetCurrent returns null.
// Retain it for identity/lifetime without changing the caller's current context.
hipError_t current_context(hipCtx_t &context, int &retained_device) {{
    auto result = hipCtxGetCurrent(&context);
    if (result != hipSuccess || context != nullptr) return result;
    result = hipGetDevice(&retained_device);
    if (result != hipSuccess) {{ retained_device = -1; return result; }}
    result = hipDevicePrimaryCtxRetain(&context, retained_device);
    if (result != hipSuccess) retained_device = -1;
    return result;
}}
void release_temporary(int device) {{
    if (device >= 0) (void)hipDevicePrimaryCtxRelease(device);
}}
}}

extern "C" hipError_t ninetoothed_triton_enter(void) {{
    launch_mutex.lock();
    hipCtx_t context = nullptr;
    int retained_device = -1;
    hipError_t result = current_context(context, retained_device);
    if (result != hipSuccess || context == nullptr) {{
        launch_mutex.unlock();
        release_temporary(retained_device);
        return result != hipSuccess ? result : hipErrorInvalidContext;
    }}
    State state{{}};
    bool inserted = false;
    try {{
        auto found = context_states.find(context);
        if (found == context_states.end()) {{
            state.primary_device = retained_device;
            retained_device = -1;
{loads}
            found = context_states.emplace(context, state).first;
            inserted = true;
        }}
{restores}
    }} catch (...) {{
        if (!inserted) (void)unload(state);
        release_temporary(retained_device);
        launch_mutex.unlock();
        return hipErrorOutOfMemory;
    }}
    release_temporary(retained_device);
    return hipSuccess;
}}

extern "C" void ninetoothed_triton_leave(void) {{ launch_mutex.unlock(); }}

// The caller must release artifacts before destroying a context it owns.
extern "C" hipError_t ninetoothed_triton_release(void) {{
    std::lock_guard<std::mutex> lock(launch_mutex);
    hipCtx_t context = nullptr;
    int retained_device = -1;
    hipError_t result = current_context(context, retained_device);
    if (result != hipSuccess) return result;
    auto found = context_states.find(context);
    if (found == context_states.end()) {{ release_temporary(retained_device); return hipSuccess; }}
    result = hipDeviceSynchronize();
    if (result != hipSuccess) {{ release_temporary(retained_device); return result; }}
    result = unload(found->second);
    context_states.erase(found);
{resets}
    release_temporary(retained_device);
    return result;
}}

extern "C" size_t ninetoothed_triton_live_contexts(void) {{
    std::lock_guard<std::mutex> lock(launch_mutex);
    return context_states.size();
}}
'''


def compile_library(compilation, source, library):
    import triton
    from ninetoothed.compiler.cache import TRITON_CACHE_DIR, write_manifest
    from . import triton as materializer

    target = triton.runtime.driver.active.get_current_target()
    if target.backend != "hip" or not target.arch.startswith("gfx906") or target.warp_size != 64:
        raise RuntimeError(f"This experimental profile requires the verified gfx906 HIP target: {target}")
    compiler = shutil.which(os.environ.get("CXX", "g++"))
    if compiler is None:
        raise RuntimeError("A host C++ compiler is required")
    hip_root = Path(os.environ.get("HIP_PATH", "/opt/dtk"))
    if not (hip_root / "include/hip/hip_runtime.h").is_file():
        raise RuntimeError(f"HIP development headers missing under {hip_root}")
    name = compilation.artifact.kernel_name
    warps, stages = materializer._compile_schedule(compilation)
    library.parent.mkdir(parents=True, exist_ok=True)
    TRITON_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    environment = os.environ.copy()
    environment["TRITON_CACHE_DIR"] = str(TRITON_CACHE_DIR)
    with tempfile.TemporaryDirectory(dir=library.parent) as directory:
        directory = Path(directory)
        compiled, linked = directory / "compiled", directory / "linked"
        subprocess.run([
            sys.executable, "-m", "triton.tools.compile", str(source),
            "--kernel-name", f"{name}_kernel", "--signature", materializer._compile_signature(compilation),
            "--grid", materializer._compile_grid(compilation), "--num-warps", str(warps),
            "--num-stages", str(stages), "--out-name", f"{name}_kernel", "--out-path", str(compiled),
        ], check=True, env=environment)
        headers = tuple(sorted(directory.glob("compiled.*.h")))
        sources = tuple(sorted(directory.glob("compiled.*.c")))
        if not headers or not sources:
            raise RuntimeError("Triton HIP compiler produced no C artifacts")
        names = kernel_names(sources)
        subprocess.run([
            sys.executable, "-m", "triton.tools.link", *(str(path) for path in headers), "--out", str(linked)
        ], check=True, env=environment)
        files = {}
        for path in (*sources, linked.with_suffix(".c")):
            files[f"{name}.triton.{path.stem}.cpp"] = (
                '#include <cstdlib>\n#define __HIP_PLATFORM_AMD__\n#include <hip/hip_runtime.h>\nextern "C" {\n'
                + path.read_text() + "\n}\n"
            )
        guard = context_guard_source(names)
        guard_name = f"{name}.triton.launch.cpp"
        files[guard_name] = guard
        for filename, text in files.items():
            (directory / filename).write_text(text)
        output = directory / library.name
        subprocess.run([
            compiler, "-shared", "-std=c++17", "-fPIC", "-pthread", "-O3",
            f"-I{hip_root / 'include'}",
            *(str(directory / filename) for filename in files),
            f"-L{hip_root / 'lib'}", f"-Wl,-rpath,{hip_root / 'lib'}", "-lamdhip64",
            "-Wl,-z,defs", "-o", str(output),
        ], check=True)
        os.replace(output, library)
        files[guard_name] = guard.replace("ninetoothed_triton_", f"{name}_triton_")
        write_manifest(library.with_suffix(".sources.json"), files)
