"""Pair original/optimized Core calls captured in actual Torch HIP graphs."""
import argparse
import ctypes
import hashlib
import json
from pathlib import Path
import statistics
import time

import infinicore as ic
import infinicore.nn.functional as F
import torch

from h3_operator_baseline import synchronize, wrap
from hygon_native import bridge_library
from hygon_rmsnorm3d_probe import COMBINATIONS, TOLERANCES, oracle


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--unroll", type=int, default=32)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    source = json.loads((args.source / "acceptance/acceptance.json").read_text())
    bridge = bridge_library(args.source / "core-ninetoothed-bridge-v2.so")
    bridge.songshan3_register_rms3d.argtypes = [ctypes.c_void_p] * 5
    bridge.songshan3_register_rms3d.restype = None
    bridge.songshan3_rms3d_count.argtypes = [ctypes.c_int]
    bridge.songshan3_rms3d_count.restype = ctypes.c_uint64
    ic.set_device(ic.device("cuda", 0))
    stream = torch.cuda.ExternalStream(bridge.songshan3_current_stream())
    torch.cuda.set_stream(stream)
    libraries, functions = [], []
    for artifact in source["artifacts"]:
        assert hashlib.sha256(Path(artifact["path"]).read_bytes()).hexdigest() == artifact["sha256"]
        library = ctypes.CDLL(artifact["path"])
        function = getattr(library, "launch_" + artifact["name"] + "_variant")
        libraries.append(library)
        functions.append(ctypes.cast(function, ctypes.c_void_p))

    def install():
        bridge.songshan3_register_rms3d(*functions)

    def restore():
        bridge.songshan3_restore_dispatch()

    def measure(operation, divisor=1):
        synchronize()
        begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        started = time.perf_counter_ns()
        begin.record(stream)
        for _ in range(args.iterations):
            operation()
        end.record(stream)
        end.synchronize()
        return ((time.perf_counter_ns() - started) / 1e6 / args.iterations / divisor,
                begin.elapsed_time(end) / args.iterations / divisor)

    paired = []
    with (args.output / "paired.jsonl").open("w", buffering=1) as raw:
        for combo_index, (dtype, weight_dtype) in enumerate(COMBINATIONS):
            for tokens in (1, 128, 512):
                seed = 20261101 + combo_index
                torch.manual_seed(seed)
                packed = torch.randn(tokens, 168, 128, device="cuda", dtype=getattr(torch, dtype))
                x = packed[:, 56:112]
                weight = torch.randn(128, device="cuda", dtype=getattr(torch, weight_dtype))
                output = torch.empty(tokens, 56, 128, device="cuda", dtype=x.dtype)
                cx, cw, cy = map(wrap, (x, weight, output))
                operation = lambda: F.rms_norm(cx, [128], cw, 1e-5, out=cy)
                expected = oracle(x, weight, 1e-5)
                graphs, errors = {}, {}
                for mode in ("baseline", "optimized"):
                    restore() if mode == "baseline" else install()
                    before_calls = bridge.songshan3_rms3d_count(0)
                    for _ in range(10):
                        operation()
                    synchronize()
                    torch.testing.assert_close(output.cpu(), expected, atol=TOLERANCES[dtype][0], rtol=TOLERANCES[dtype][1])
                    errors[mode] = float((output.cpu().float() - expected.float()).abs().max())
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph, stream=stream):
                        for _ in range(args.unroll):
                            operation()
                    graphs[mode] = graph
                    for _ in range(10):
                        graph.replay()
                    synchronize()
                    assert bridge.songshan3_rms3d_count(0) - before_calls == (10 + args.unroll if mode == "optimized" else 0)
                    torch.testing.assert_close(output.cpu(), expected, atol=TOLERANCES[dtype][0], rtol=TOLERANCES[dtype][1])
                timing = {f"{mode}_{kind}_ms": [] for mode in ("baseline", "optimized")
                          for kind in ("api_host", "api_event", "hipgraph_host", "hipgraph_event")}
                for repeat in range(args.repeats):
                    order = ("baseline", "optimized") if repeat % 2 == 0 else ("optimized", "baseline")
                    for mode in order:
                        restore() if mode == "baseline" else install()
                        api_host, api_event = measure(operation)
                        graph_host, graph_event = measure(graphs[mode].replay, args.unroll)
                        for kind, value in zip(("api_host", "api_event", "hipgraph_host", "hipgraph_event"),
                                               (api_host, api_event, graph_host, graph_event)):
                            timing[f"{mode}_{kind}_ms"].append(value)
                x.add_(0.125)
                synchronize()
                expected_changed = oracle(x, weight, 1e-5)
                for graph in graphs.values():
                    graph.replay()
                    synchronize()
                    torch.testing.assert_close(output.cpu(), expected_changed, atol=TOLERANCES[dtype][0], rtol=TOLERANCES[dtype][1])
                record = dict(dtype=dtype, weight_dtype=weight_dtype, shape=list(x.shape),
                              input_stride=list(x.stride()), output_stride=list(output.stride()), weight_stride=list(weight.stride()),
                              seed=seed, epsilon=1e-5, atol=TOLERANCES[dtype][0], rtol=TOLERANCES[dtype][1],
                              baseline_error=errors["baseline"], optimized_error=errors["optimized"],
                              unroll=args.unroll, warmup=10, iterations=args.iterations, repeats=args.repeats,
                              changed_input_replay=True, **timing)
                for kind in ("api_host", "api_event", "hipgraph_host", "hipgraph_event"):
                    for mode in ("baseline", "optimized"):
                        record[f"{mode}_{kind}_median_ms"] = statistics.median(timing[f"{mode}_{kind}_ms"])
                    record[f"{kind}_speedup"] = record[f"baseline_{kind}_median_ms"] / record[f"optimized_{kind}_median_ms"]
                paired.append(record)
                raw.write(json.dumps(record) + "\n")
                print(json.dumps({k: record[k] for k in ("dtype", "weight_dtype", "shape", "api_host_speedup",
                                                        "baseline_hipgraph_event_median_ms",
                                                        "optimized_hipgraph_event_median_ms", "hipgraph_event_speedup")}), flush=True)
    restore()
    for library, artifact in zip(libraries, source["artifacts"]):
        release = getattr(library, artifact["name"] + "_triton_release")
        release.restype = ctypes.c_int
        assert release() == 0
    result = dict(paired=paired, cases=len(paired), restored=True, source=str(args.source),
                  api_timing="Same preallocated Core API; synchronized host time and elapsed Core stream event time",
                  hipgraph_timing=f"Same Torch HIP graph replay, {args.unroll} Core calls per graph; elapsed time per RMSNorm including graph execution overhead; not pure instruction or model time")
    (args.output / "graph-benchmark.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "paired"}), flush=True)


if __name__ == "__main__":
    main()
