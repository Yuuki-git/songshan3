"""Correctness and paired original/optimized Core RMSNorm measurements.

Every output is preallocated. Event times measure the Core stream's elapsed
timeline, including gaps between submissions; they are not pure kernel times.
"""
import argparse
import ctypes
import hashlib
import json
from pathlib import Path
import statistics
import time
import traceback

import infinicore as ic
import infinicore.nn.functional as F
import torch

from h3_operator_baseline import cases, create, synchronize, wrap
from hygon_native import NativeTensor, bridge_library, export_native, view
from hygon_rmsnorm_probe import make_kernel

COMBINATIONS = (("float16", "float16"), ("float16", "float32"),
                ("bfloat16", "bfloat16"), ("bfloat16", "float32"), ("float32", "float32"))
TOLERANCES = {"float16": (2e-3, 2e-3), "bfloat16": (1e-2, 1e-2), "float32": (1e-5, 1e-4)}


def oracle(x, weight, epsilon):
    host, hw = x.cpu().double(), weight.cpu().double()
    return (host * torch.rsqrt(host.square().mean(-1, keepdim=True) + epsilon) * hw).to(x.dtype)


def check_output(output, expected, dtype):
    atol, rtol = TOLERANCES[dtype]
    torch.testing.assert_close(output.cpu(), expected, atol=atol, rtol=rtol)
    return (output.cpu().float() - expected.float()).abs().max().item()


def measure(operation, iterations, stream):
    synchronize()
    start = time.perf_counter_ns()
    for _ in range(iterations):
        operation()
    synchronize()
    host_ms = (time.perf_counter_ns() - start) / iterations / 1e6
    begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    begin.record(stream)
    for _ in range(iterations):
        operation()
    end.record(stream)
    end.synchronize()
    return host_ms, begin.elapsed_time(end) / iterations


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bridge", type=Path, required=True)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    bridge = bridge_library(args.bridge)
    ic.set_device(ic.device("cuda", 0))
    stream = torch.cuda.ExternalStream(bridge.songshan3_current_stream())
    original_dispatch = bridge.songshan3_dispatch_address(1)
    kernels, artifacts = {}, []
    for dtype, weight_dtype in COMBINATIONS:
        name = f"hygon_rms128_{dtype}_{weight_dtype}"
        directory = args.output / "kernels" / name
        directory.mkdir(parents=True, exist_ok=True)
        jit = make_kernel(dtype, weight_dtype, kernel_name=f"{name}_jit")
        aot = make_kernel(dtype, weight_dtype, caller="cuda", kernel_name=name, output_dir=directory)
        library, launch, record = export_native(aot, directory)
        kernels[dtype, weight_dtype] = dict(jit=jit, aot=aot, library=library, native=launch, record=record)
        artifacts.append(record)
        print(json.dumps(dict(compiled=name, native=record)), flush=True)
    pointers = [ctypes.cast(kernels[combination]["native"], ctypes.c_void_p) for combination in COMBINATIONS]

    def install():
        bridge.songshan3_register_rms(*pointers)

    def restore():
        bridge.songshan3_restore_dispatch()

    install()
    development_dispatch = bridge.songshan3_dispatch_address(1)
    assert development_dispatch != original_dispatch
    correctness = []
    with (args.output / "correctness.jsonl").open("w", buffering=1) as raw:
        for combo_index, (dtype, weight_dtype) in enumerate(COMBINATIONS):
            kernel = kernels[dtype, weight_dtype]
            for rows in (1, 7, 65):
                for distribution in ("normal", "zeros", "tiny", "large", "mixed"):
                    seed = 20261016 + combo_index * 100 + rows
                    torch.manual_seed(seed)
                    stride = 2 if rows == 7 else 1
                    x = torch.randn(rows, 128 * stride, device="cuda", dtype=getattr(torch, dtype))[:, ::stride]
                    if distribution == "zeros": x.zero_()
                    if distribution == "tiny": x.mul_(1e-4)
                    if distribution == "large": x.mul_(1000)
                    if distribution == "mixed": x[:, ::2].mul_(1e-4)
                    weight = torch.randn(256, device="cuda", dtype=getattr(torch, weight_dtype))[::2]
                    epsilon = 1e-6 if distribution == "tiny" else 1e-5
                    expected = oracle(x, weight, epsilon)
                    outputs = [torch.empty(rows, 256, device="cuda", dtype=x.dtype)[:, ::2] for _ in range(4)]
                    synchronize()
                    kernel["jit"](x, weight, epsilon, outputs[0])
                    kernel["aot"](x, weight, epsilon, outputs[1])
                    scalar = ctypes.c_double(epsilon)
                    scalar_tensor = NativeTensor(ctypes.addressof(scalar), None, None)
                    status = kernel["native"](torch.cuda.current_stream().cuda_stream,
                                               view(x), view(weight), scalar_tensor, view(outputs[2]))
                    assert status == 0, status
                    cx, cw, cy = map(wrap, (x, weight, outputs[3]))
                    count = bridge.songshan3_call_count(1)
                    F.rms_norm(cx, [128], cw, epsilon, out=cy)
                    synchronize()
                    assert bridge.songshan3_call_count(1) == count + 1
                    errors = [check_output(output, expected, dtype) for output in outputs]
                    record = dict(dtype=dtype, weight_dtype=weight_dtype, rows=rows, seed=seed,
                                  input_stride=list(x.stride()), weight_stride=list(weight.stride()),
                                  output_stride=list(outputs[0].stride()), distribution=distribution, epsilon=epsilon,
                                  atol=TOLERANCES[dtype][0], rtol=TOLERANCES[dtype][1],
                                  paths=["jit", "python_aot", "native_cpp", "core"], max_errors=errors, passed=True)
                    correctness.append(record)
                    raw.write(json.dumps(record) + "\n")
            print(json.dumps(dict(correctness_combo=[dtype, weight_dtype], passed=15)), flush=True)

    paired, frozen = [], []
    with (args.output / "paired.jsonl").open("w", buffering=1) as raw:
        for case in cases():
            if case["op"] != "rms_norm":
                continue
            reference, operation, output, keep, atol, rtol = create(case)
            expected = oracle(keep[0], keep[2], case["epsilon"])
            synchronize()
            restore()
            operation()
            synchronize()
            baseline_error = check_output(output, expected, case["dtype"])
            install()
            before_calls = bridge.songshan3_call_count(1)
            before_fallbacks = bridge.songshan3_call_count(3)
            operation()
            synchronize()
            optimized_error = check_output(output, expected, case["dtype"])
            supported = case["shape"][1] == 128
            assert bridge.songshan3_call_count(1) - before_calls == int(supported)
            assert bridge.songshan3_call_count(3) - before_fallbacks == int(not supported)
            frozen.append(dict(case, passed=True, native=supported, baseline_error=baseline_error, optimized_error=optimized_error))
            if not supported:
                continue
            timing = dict(baseline_host_ms=[], optimized_host_ms=[], baseline_event_ms=[], optimized_event_ms=[])
            for mode in ("baseline", "optimized"):
                restore() if mode == "baseline" else install()
                for _ in range(10): operation()
                synchronize()
            for repeat in range(args.repeats):
                order = ("baseline", "optimized") if repeat % 2 == 0 else ("optimized", "baseline")
                for mode in order:
                    restore() if mode == "baseline" else install()
                    host_ms, event_ms = measure(operation, args.iterations, stream)
                    timing[f"{mode}_host_ms"].append(host_ms)
                    timing[f"{mode}_event_ms"].append(event_ms)
            record = dict(case, input_stride=list(keep[0].stride()), weight_stride=list(keep[2].stride()),
                          output_stride=list(output.stride()), atol=atol, rtol=rtol,
                          baseline_error=baseline_error, optimized_error=optimized_error,
                          iterations=args.iterations, repeats=args.repeats, **timing)
            for mode in ("baseline", "optimized"):
                for kind in ("host", "event"):
                    record[f"{mode}_{kind}_median_ms"] = statistics.median(record[f"{mode}_{kind}_ms"])
            record["host_speedup"] = record["baseline_host_median_ms"] / record["optimized_host_median_ms"]
            record["event_speedup"] = record["baseline_event_median_ms"] / record["optimized_event_median_ms"]
            paired.append(record)
            raw.write(json.dumps(record) + "\n")
            print(json.dumps({k: record[k] for k in ("id", "dtype", "weight_dtype", "shape", "baseline_host_median_ms",
                                                   "optimized_host_median_ms", "host_speedup")}), flush=True)

    install()
    graph_record = {}
    try:
        torch.manual_seed(20261016)
        x = torch.randn(65, 128, device="cuda", dtype=torch.float16)
        weight = torch.randn(128, device="cuda")
        output = torch.empty_like(x)
        cx, cw, cy = map(wrap, (x, weight, output))
        synchronize()
        F.rms_norm(cx, [128], cw, 1e-5, out=cy)
        synchronize()
        ic.start_graph_recording()
        F.rms_norm(cx, [128], cw, 1e-5, out=cy)
        graph = ic.stop_graph_recording()
        graph.run()
        synchronize()
        check_output(output, oracle(x, weight, 1e-5), "float16")
        x.add_(0.125)
        synchronize()
        graph.run()
        synchronize()
        check_output(output, oracle(x, weight, 1e-5), "float16")
        graph_record = dict(passed=True, replay_with_changed_input=True)
    except Exception:
        graph_record = dict(passed=False, error=traceback.format_exc())
    finally:
        restore()
    assert bridge.songshan3_dispatch_address(1) == original_dispatch
    for kernel in kernels.values():
        release = getattr(kernel["library"], kernel["record"]["name"] + "_triton_release")
        assert release() == 0
        raw_library = ctypes.CDLL(str(kernel["aot"]._library))
        raw_library.ninetoothed_triton_release.restype = ctypes.c_int
        assert raw_library.ninetoothed_triton_release() == 0
    result = dict(schema="songshan3-hygon-rms128-v1", correctness_passed=len(correctness), correctness_total=75,
                  frozen_rms_passed=len(frozen), frozen_rms_total=40, frozen_native=sum(item["native"] for item in frozen),
                  paired_cases=len(paired), graph=graph_record, restored=True, tile_rows=1, warps=1,
                  intermediate="FP32 square/sum; FP64 scalar sqrt/reciprocal; FP32 scale and product; original output dtype",
                  host_timing="synchronized Core Python API with preallocated output",
                  event_timing="elapsed Core stream timeline including host submission gaps, not pure kernel time",
                  core_stream=bridge.songshan3_current_stream(), native_calls=bridge.songshan3_call_count(1),
                  fallback_calls=bridge.songshan3_call_count(3), original_dispatch=original_dispatch,
                  development_dispatch=development_dispatch, bridge=str(args.bridge),
                  bridge_sha256=hashlib.sha256(args.bridge.read_bytes()).hexdigest(),
                  artifacts=artifacts, frozen=frozen, paired=paired)
    (args.output / "acceptance.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k not in ("artifacts", "frozen", "paired")}), flush=True)
    return int(not graph_record.get("passed"))


if __name__ == "__main__":
    raise SystemExit(main())
