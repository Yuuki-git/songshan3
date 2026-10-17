"""Extra runtime checks: real fallback, nondefault native stream and aliasing."""
import argparse
import ctypes
import hashlib
import json
from pathlib import Path

import infinicore as ic
import infinicore.nn.functional as F
import torch

from h3_operator_baseline import synchronize, wrap
from hygon_native import NativeTensor, bridge_library, view
from hygon_rmsnorm3d_probe import COMBINATIONS, TOLERANCES, oracle


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    source = json.loads((args.source / "acceptance/acceptance.json").read_text())
    bridge = bridge_library(args.source / "core-ninetoothed-bridge-v2.so")
    bridge.songshan3_register_rms3d.argtypes = [ctypes.c_void_p] * 5
    bridge.songshan3_register_rms3d.restype = None
    bridge.songshan3_rms3d_count.argtypes = [ctypes.c_int]
    bridge.songshan3_rms3d_count.restype = ctypes.c_uint64
    ic.set_device(ic.device("cuda", 0))
    torch.cuda.set_stream(torch.cuda.ExternalStream(bridge.songshan3_current_stream()))
    originals = bridge.songshan3_dispatch_address(1)
    libraries, launchers = [], []
    for artifact in source["artifacts"]:
        path = Path(artifact["path"])
        assert hashlib.sha256(path.read_bytes()).hexdigest() == artifact["sha256"]
        library = ctypes.CDLL(str(path))
        launch = getattr(library, "launch_" + artifact["name"] + "_variant")
        launch.argtypes = [ctypes.c_void_p] + [NativeTensor] * 4
        launch.restype = ctypes.c_int
        libraries.append(library)
        launchers.append(launch)
    assert len(launchers) == len(COMBINATIONS)
    bridge.songshan3_register_rms3d(*(ctypes.cast(f, ctypes.c_void_p) for f in launchers))
    records = []
    with (args.output / "checks.jsonl").open("w", buffering=1) as raw:
        for combo_index, (dtype, weight_dtype) in enumerate(COMBINATIONS):
            dt, wt = getattr(torch, dtype), getattr(torch, weight_dtype)
            for shape in ((7, 128), (7, 3, 256), (1, 3, 5376)):
                seed = 20261026 + combo_index
                torch.manual_seed(seed)
                x = torch.randn(shape, device="cuda", dtype=dt)
                weight = torch.randn(shape[-1], device="cuda", dtype=wt)
                output = torch.empty_like(x)
                expected = oracle(x, weight, 1e-5)
                cx, cw, cy = map(wrap, (x, weight, output))
                synchronize()
                before_native = bridge.songshan3_rms3d_count(0)
                before_fallback = bridge.songshan3_rms3d_count(1)
                F.rms_norm(cx, [shape[-1]], cw, 1e-5, out=cy)
                synchronize()
                assert bridge.songshan3_rms3d_count(0) == before_native
                assert bridge.songshan3_rms3d_count(1) == before_fallback + 1
                torch.testing.assert_close(output.cpu(), expected, atol=TOLERANCES[dtype][0], rtol=TOLERANCES[dtype][1])
                record = dict(check="real_original_fallback", dtype=dtype, weight_dtype=weight_dtype,
                              shape=shape, seed=seed, epsilon=1e-5, atol=TOLERANCES[dtype][0],
                              rtol=TOLERANCES[dtype][1], passed=True,
                              max_error=float((output.cpu().float() - expected.float()).abs().max()))
                records.append(record)
                raw.write(json.dumps(record) + "\n")
            nondefault = torch.cuda.Stream()
            with torch.cuda.stream(nondefault):
                torch.manual_seed(20261031 + combo_index)
                packed = torch.randn(7, 168, 128, device="cuda", dtype=dt)
                x = packed[:, 56:112]
                weight = torch.randn(256, device="cuda", dtype=wt)[::2]
                expected = oracle(x, weight, 1e-6)
                scalar = ctypes.c_double(1e-6)
                status = launchers[combo_index](nondefault.cuda_stream, view(x), view(weight),
                                                NativeTensor(ctypes.addressof(scalar), None, None), view(x))
                assert status == 0, status
                nondefault.synchronize()
                torch.testing.assert_close(x.cpu(), expected, atol=TOLERANCES[dtype][0], rtol=TOLERANCES[dtype][1])
                record = dict(check="nondefault_stream_native_inplace", dtype=dtype, weight_dtype=weight_dtype,
                              shape=list(x.shape), input_stride=list(x.stride()), weight_stride=list(weight.stride()),
                              seed=20261031 + combo_index, epsilon=1e-6, atol=TOLERANCES[dtype][0],
                              rtol=TOLERANCES[dtype][1], stream=int(nondefault.cuda_stream), passed=True,
                              max_error=float((x.cpu().float() - expected.float()).abs().max()))
                records.append(record)
                raw.write(json.dumps(record) + "\n")
            print(json.dumps(dict(combo=[dtype, weight_dtype], extended_checks=4, passed=True)), flush=True)
    synchronize()
    bridge.songshan3_restore_dispatch()
    assert bridge.songshan3_dispatch_address(1) == originals
    for library, artifact in zip(libraries, source["artifacts"]):
        release = getattr(library, artifact["name"] + "_triton_release")
        release.restype = ctypes.c_int
        assert release() == 0
    result = dict(total=len(records), passed=sum(r["passed"] for r in records),
                  real_original_fallbacks=bridge.songshan3_rms3d_count(1),
                  native_core_calls=bridge.songshan3_rms3d_count(0),
                  nondefault_stream_inplace_cases=5, restored=True, source=str(args.source),
                  artifacts=source["artifacts"])
    (args.output / "extended.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
