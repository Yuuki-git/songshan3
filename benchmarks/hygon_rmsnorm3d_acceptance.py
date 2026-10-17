"""Validate true 3-D Core RMSNorm and measure paired original/optimized APIs.

Requires a working GPU. Compile-only results never enter these result tables.
"""
import argparse
import ctypes
import json
from pathlib import Path
import statistics

import infinicore as ic
import infinicore.nn.functional as F
import torch

from h3_operator_baseline import synchronize, wrap
from hygon_native import NativeTensor, bridge_library, export_native, view
from hygon_rmsnorm3d_probe import COMBINATIONS, TOLERANCES, make_kernel, oracle


def checked(output, expected, dtype):
    actual = output.cpu()
    atol, rtol = TOLERANCES[dtype]
    torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol)
    return float((actual.float() - expected.float()).abs().max())


def case_inputs(dtype, weight_dtype, tokens, heads, layout, distribution, seed):
    torch.manual_seed(seed)
    dt = getattr(torch, dtype)
    if layout in ("q", "k"):
        packed = torch.randn(tokens, 3 * heads, 128, device="cuda", dtype=dt)
        offset = 0 if layout == "q" else heads
        x = packed[:, offset:offset + heads]
    elif layout == "strided":
        x = torch.randn(tokens, 2 * heads, 256, device="cuda", dtype=dt)[:, ::2, ::2]
    else:
        x = torch.randn(tokens, heads, 128, device="cuda", dtype=dt)
    if distribution == "zeros":
        x.zero_()
    elif distribution == "tiny":
        x.mul_(1e-4)
    elif distribution == "large":
        x.mul_(1000)
    elif distribution == "mixed":
        x[:, :, ::2].mul_(1e-4)
    weight = torch.randn(256, device="cuda", dtype=getattr(torch, weight_dtype))[::2]
    return x, weight


def event_measure(operation, stream, iterations):
    synchronize()
    begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    begin.record(stream)
    for _ in range(iterations):
        operation()
    end.record(stream)
    end.synchronize()
    return begin.elapsed_time(end) / iterations


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bridge", type=Path, required=True)
    parser.add_argument("--vector", action="store_true")
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    bridge = bridge_library(args.bridge)
    bridge.songshan3_register_rms3d.argtypes = [ctypes.c_void_p] * 5
    bridge.songshan3_register_rms3d.restype = None
    bridge.songshan3_rms3d_count.argtypes = [ctypes.c_int]
    bridge.songshan3_rms3d_count.restype = ctypes.c_uint64
    ic.set_device(ic.device("cuda", 0))
    stream = torch.cuda.ExternalStream(bridge.songshan3_current_stream())
    torch.cuda.set_stream(stream)
    original_dispatch = bridge.songshan3_dispatch_address(1)
    kernels = {}
    artifacts = []
    for dtype, weight_dtype in COMBINATIONS:
        name = f"hygon_rms3d_{dtype}_{weight_dtype}"
        directory = args.output / "kernels" / name
        directory.mkdir(parents=True, exist_ok=True)
        jit = make_kernel(dtype, weight_dtype, args.vector, kernel_name=name + "_jit")
        aot = make_kernel(dtype, weight_dtype, args.vector, caller="cuda", kernel_name=name, output_dir=directory)
        library, launch, record = export_native(aot, directory)
        kernels[dtype, weight_dtype] = dict(jit=jit, aot=aot, library=library, native=launch, record=record)
        artifacts.append(record)
        print(json.dumps(dict(compiled=name)), flush=True)
    pointers = [ctypes.cast(kernels[combo]["native"], ctypes.c_void_p) for combo in COMBINATIONS]

    def install():
        bridge.songshan3_register_rms3d(*pointers)

    def restore():
        bridge.songshan3_restore_dispatch()

    specs = ((1, 56, "q", "normal"), (7, 56, "k", "normal"), (17, 7, "strided", "normal"),
             (65, 1, "contiguous", "normal"), (7, 7, "q", "zeros"),
             (7, 7, "q", "tiny"), (7, 7, "k", "large"), (7, 7, "k", "mixed"))
    correctness = []
    install()
    assert bridge.songshan3_dispatch_address(1) != original_dispatch
    with (args.output / "correctness.jsonl").open("w", buffering=1) as raw:
        for combo_index, (dtype, weight_dtype) in enumerate(COMBINATIONS):
            kernel = kernels[dtype, weight_dtype]
            for case_index, (tokens, heads, layout, distribution) in enumerate(specs):
                seed = 20261019 + combo_index * 100 + case_index
                x, weight = case_inputs(dtype, weight_dtype, tokens, heads, layout, distribution, seed)
                epsilon = 1e-6 if distribution == "tiny" else 1e-5
                before = x.cpu().clone()
                expected = oracle(x, weight, epsilon)
                backings = [torch.full((tokens, 2 * heads, 256), float("nan"), device="cuda", dtype=x.dtype)
                            for _ in range(4)]
                outputs = [b[:, ::2, ::2] for b in backings]
                synchronize()
                kernel["jit"](x, weight, epsilon, outputs[0])
                kernel["aot"](x, weight, epsilon, outputs[1])
                scalar = ctypes.c_double(epsilon)
                status = kernel["native"](stream.cuda_stream, view(x), view(weight),
                                           NativeTensor(ctypes.addressof(scalar), None, None), view(outputs[2]))
                assert status == 0, status
                cx, cw, cy = map(wrap, (x, weight, outputs[3]))
                count = bridge.songshan3_rms3d_count(0)
                F.rms_norm(cx, [128], cw, epsilon, out=cy)
                synchronize()
                assert bridge.songshan3_rms3d_count(0) == count + 1
                errors = [checked(output, expected, dtype) for output in outputs]
                assert torch.equal(x.cpu(), before)
                for backing in backings:
                    assert bool(backing[:, 1::2].isnan().all())
                    assert bool(backing[:, ::2, 1::2].isnan().all())
                record = dict(dtype=dtype, weight_dtype=weight_dtype, shape=list(x.shape),
                              input_stride=list(x.stride()), weight_stride=list(weight.stride()),
                              output_stride=list(outputs[0].stride()), layout=layout,
                              distribution=distribution, seed=seed, epsilon=epsilon,
                              atol=TOLERANCES[dtype][0], rtol=TOLERANCES[dtype][1],
                              paths=["jit", "python_aot", "native_cpp", "core"],
                              max_errors=errors, input_unchanged=True, output_guards=True, passed=True)
                correctness.append(record)
                raw.write(json.dumps(record) + "\n")
            print(json.dumps(dict(correctness_combo=[dtype, weight_dtype], passed=len(specs))), flush=True)

    paired, graph_checks, inplace_checks = [], [], []
    with (args.output / "paired.jsonl").open("w", buffering=1) as raw:
        for combo_index, (dtype, weight_dtype) in enumerate(COMBINATIONS):
            for tokens in (1, 128, 512):
                seed = 20261020 + combo_index
                x, weight = case_inputs(dtype, weight_dtype, tokens, 56, "k", "normal", seed)
                # Original infiniop requires contiguous columns and weights.
                weight = weight.contiguous()
                output = torch.empty_like(x)
                cx, cw, cy = map(wrap, (x, weight, output))
                operation = lambda: F.rms_norm(cx, [128], cw, 1e-5, out=cy)
                expected = oracle(x, weight, 1e-5)
                graphs, errors = {}, {}
                for mode in ("baseline", "optimized"):
                    restore() if mode == "baseline" else install()
                    before_calls = bridge.songshan3_rms3d_count(0)
                    operation()
                    synchronize()
                    errors[mode] = checked(output, expected, dtype)
                    assert bridge.songshan3_rms3d_count(0) - before_calls == int(mode == "optimized")
                    for _ in range(10):
                        operation()
                    synchronize()
                    ic.start_graph_recording()
                    operation()
                    graphs[mode] = ic.stop_graph_recording()
                    graphs[mode].run()
                    synchronize()
                    checked(output, expected, dtype)
                timing = {f"{mode}_{kind}_ms": [] for mode in ("baseline", "optimized") for kind in ("api", "graph")}
                for repeat in range(args.repeats):
                    order = ("baseline", "optimized") if repeat % 2 == 0 else ("optimized", "baseline")
                    for mode in order:
                        restore() if mode == "baseline" else install()
                        timing[f"{mode}_api_ms"].append(event_measure(operation, stream, args.iterations))
                        timing[f"{mode}_graph_ms"].append(event_measure(graphs[mode].run, stream, args.iterations))
                x.add_(0.125)
                synchronize()
                expected_changed = oracle(x, weight, 1e-5)
                for mode in ("baseline", "optimized"):
                    graphs[mode].run()
                    synchronize()
                    checked(output, expected_changed, dtype)
                graph_checks.append(dict(dtype=dtype, weight_dtype=weight_dtype, tokens=tokens,
                                         baseline=True, optimized=True, changed_input=True))
                record = dict(dtype=dtype, weight_dtype=weight_dtype, shape=list(x.shape),
                              input_stride=list(x.stride()), output_stride=list(output.stride()),
                              seed=seed, epsilon=1e-5, atol=TOLERANCES[dtype][0], rtol=TOLERANCES[dtype][1],
                              baseline_error=errors["baseline"], optimized_error=errors["optimized"],
                              iterations=args.iterations, repeats=args.repeats, warmup=10, **timing)
                for kind in ("api", "graph"):
                    for mode in ("baseline", "optimized"):
                        record[f"{mode}_{kind}_median_ms"] = statistics.median(timing[f"{mode}_{kind}_ms"])
                    record[f"{kind}_speedup"] = record[f"baseline_{kind}_median_ms"] / record[f"optimized_{kind}_median_ms"]
                paired.append(record)
                raw.write(json.dumps(record) + "\n")
                print(json.dumps(record), flush=True)
            install()
            x, weight = case_inputs(dtype, weight_dtype, 7, 3, "k", "normal", 20261021 + combo_index)
            expected = oracle(x, weight, 1e-5)
            cx, cw = map(wrap, (x, weight))
            F.rms_norm(cx, [128], cw, 1e-5, out=cx)
            synchronize()
            error = checked(x, expected, dtype)
            inplace_checks.append(dict(dtype=dtype, weight_dtype=weight_dtype, max_error=error, passed=True))

    restore()
    assert bridge.songshan3_dispatch_address(1) == original_dispatch
    for kernel in kernels.values():
        release = getattr(kernel["library"], kernel["record"]["name"] + "_triton_release")
        assert release() == 0
        raw_library = ctypes.CDLL(str(kernel["aot"]._library))
        raw_library.ninetoothed_triton_release.restype = ctypes.c_int
        assert raw_library.ninetoothed_triton_release() == 0
    result = dict(schema="songshan3-hygon-rms3d-v1", vector=args.vector, correctness_passed=len(correctness),
                  correctness_total=len(COMBINATIONS) * len(specs), paired_cases=len(paired), graphs=graph_checks,
                  inplace=inplace_checks, native_3d_calls=bridge.songshan3_rms3d_count(0),
                  fallback_calls=bridge.songshan3_rms3d_count(1), restored=True,
                  artifacts=artifacts, paired=paired,
                  timing="HIP event elapsed Core stream timeline including submission gaps; preallocated outputs; same Core API or Core graph replay; not pure kernel or full model time")
    (args.output / "acceptance.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k not in ("paired", "artifacts")}), flush=True)


if __name__ == "__main__":
    main()
