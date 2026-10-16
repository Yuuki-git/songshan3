"""Host utilities shared by this round's native kernel experiments."""
import ctypes
import hashlib
from pathlib import Path
import subprocess

from ninetoothed.backends.materializers.triton import publish_cpp_sources


class NativeTensor(ctypes.Structure):
    _fields_ = [("data", ctypes.c_void_p), ("shape", ctypes.POINTER(ctypes.c_uint64)),
                ("strides", ctypes.POINTER(ctypes.c_int64))]


def view(value):
    return NativeTensor(value.data_ptr(), (ctypes.c_uint64 * value.ndim)(*value.shape),
                        (ctypes.c_int64 * value.ndim)(*value.stride()))


def export_native(kernel, directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    publish_cpp_sources(kernel._compilation, directory)
    name = kernel._artifact.kernel_name
    native = directory / f"{name}.native.so"
    command = ["g++", "-shared", "-std=c++17", "-fPIC", "-pthread", "-O3", "-I/opt/dtk/include",
               f"-I{directory}", *(str(path) for path in sorted(directory.glob("*.cpp"))),
               "-L/opt/dtk/lib", "-Wl,-rpath,/opt/dtk/lib", "-lamdhip64", "-Wl,-z,defs", "-o", str(native)]
    result = subprocess.run(command, capture_output=True, text=True)
    (directory / "host-build.log").write_text(result.stdout + result.stderr)
    result.check_returncode()
    library = ctypes.CDLL(str(native))
    launch = getattr(library, f"launch_{name}_variant")
    launch.argtypes = [ctypes.c_void_p] + [NativeTensor] * len(kernel._compilation.launch_abi.public_args)
    launch.restype = ctypes.c_int
    release = getattr(library, f"{name}_triton_release")
    release.restype = ctypes.c_int
    return library, launch, dict(path=str(native), sha256=hashlib.sha256(native.read_bytes()).hexdigest(), name=name)


def bridge_library(path):
    bridge = ctypes.CDLL(str(path))
    bridge.songshan3_register_rms.argtypes = [ctypes.c_void_p] * 5
    bridge.songshan3_register_rms.restype = None
    bridge.songshan3_restore_dispatch.restype = None
    bridge.songshan3_call_count.argtypes = [ctypes.c_int]
    bridge.songshan3_call_count.restype = ctypes.c_uint64
    bridge.songshan3_current_stream.restype = ctypes.c_void_p
    bridge.songshan3_dispatch_address.argtypes = [ctypes.c_int]
    bridge.songshan3_dispatch_address.restype = ctypes.c_void_p
    return bridge
