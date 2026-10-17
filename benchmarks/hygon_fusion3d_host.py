"""ctypes host bindings for the independent Core fusion graph operators."""
import ctypes
import torch

COMBINATIONS = (("adaln", "float16"), ("adaln", "bfloat16"),
                ("residual", "float16"), ("residual", "bfloat16"))


class FusionTensor(ctypes.Structure):
    _fields_ = [("data", ctypes.c_void_p), ("ndim", ctypes.c_uint64),
                ("shape", ctypes.POINTER(ctypes.c_uint64)),
                ("strides", ctypes.POINTER(ctypes.c_int64)), ("dtype", ctypes.c_int)]


def descriptor(value):
    if value is None:
        return FusionTensor(None, 0, None, None, 0)
    dtype = {torch.float16: 0, torch.bfloat16: 1, torch.float32: 2}[value.dtype]
    return FusionTensor(value.data_ptr(), value.ndim,
                        (ctypes.c_uint64 * value.ndim)(*value.shape),
                        (ctypes.c_int64 * value.ndim)(*value.stride()), dtype)


class Bridge:
    def __init__(self, path):
        self.library = ctypes.CDLL(str(path))
        lib = self.library
        lib.songshan3_fusion_register.argtypes = [ctypes.c_void_p] * 4
        lib.songshan3_fusion_register.restype = None
        lib.songshan3_fusion_clear.restype = None
        lib.songshan3_fusion_call.argtypes = [ctypes.c_int] + [FusionTensor] * 6 + [ctypes.c_int]
        lib.songshan3_fusion_call.restype = ctypes.c_int
        lib.songshan3_fusion_count.argtypes = [ctypes.c_int]
        lib.songshan3_fusion_count.restype = ctypes.c_uint64
        lib.songshan3_fusion_stream.restype = ctypes.c_void_p
        lib.songshan3_fusion_error.restype = ctypes.c_char_p

    def register(self, functions):
        self.library.songshan3_fusion_register(*(ctypes.cast(fn, ctypes.c_void_p) for fn in functions))

    def clear(self):
        self.library.songshan3_fusion_clear()

    def count(self, index):
        return self.library.songshan3_fusion_count(index)

    def counts(self):
        return [self.count(index) for index in range(7)]

    def operation(self, op, inputs, output, workspaces=(), original=False):
        workspaces = tuple(workspaces) + (None,) * (2 - len(workspaces))
        state = dict(keep=(inputs, output, workspaces),
                     descriptors=[descriptor(v) for v in (output, *inputs, *workspaces)])
        kind = 0 if op == "adaln" else 1

        def operation():
            status = self.library.songshan3_fusion_call(kind, *state["descriptors"], int(original))
            if status:
                raise RuntimeError(self.library.songshan3_fusion_error().decode())
        operation.state = state
        return operation


def assert_exact(actual, expected):
    host = actual.cpu()
    torch.testing.assert_close(host, expected, atol=0, rtol=0)
    integer = torch.int16 if host.dtype in (torch.float16, torch.bfloat16) else torch.int32
    assert torch.equal(host.contiguous().view(integer), expected.contiguous().view(integer)), "floating-point bit patterns differ"
    return dict(passed=True, bitwise=True, max_error=0.0)
