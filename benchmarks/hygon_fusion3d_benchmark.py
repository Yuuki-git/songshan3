"""Same-entry Core fusion timings, real graph replays, and H3 pointwise subgraphs."""
import argparse
import ctypes
import hashlib
import json
from pathlib import Path
import statistics
import time
import infinicore as ic
import torch
from h3_operator_baseline import synchronize
from hygon_native import NativeTensor
from hygon_fusion3d import create_inputs, oracle
from hygon_fusion3d_host import Bridge, COMBINATIONS, assert_exact


def load(source):
    manifest = json.loads((source / "acceptance/acceptance.json").read_text())
    path = source / "core-h3-fusion.so"
    assert hashlib.sha256(path.read_bytes()).hexdigest() == manifest["bridge_sha256"]
    bridge = Bridge(path)
    libraries, functions = [], []
    for artifact in manifest["artifacts"]:
        assert hashlib.sha256(Path(artifact["path"]).read_bytes()).hexdigest() == artifact["sha256"]
        library = ctypes.CDLL(artifact["path"])
        function = getattr(library, "launch_" + artifact["name"] + "_variant")
        function.argtypes = [ctypes.c_void_p] + [NativeTensor] * 4
        function.restype = ctypes.c_int
        libraries.append(library)
        functions.append(function)
    bridge.register(functions)
    return bridge, libraries, manifest, functions


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    ic.set_device(ic.device("cuda", 0))
    bridge, libraries, manifest, functions = load(args.source)
    import ninetoothed
    assert Path(ninetoothed.__file__).is_relative_to(args.source / "ninetoothed/src")
    stream = torch.cuda.ExternalStream(bridge.library.songshan3_fusion_stream())
    torch.cuda.set_stream(stream)
    iterations, repeats, warmup, unroll = 100, 5, 10, 32

    def measure(operation, divisor=1):
        synchronize()
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        clock = time.perf_counter_ns()
        start.record(stream)
        for _ in range(iterations):
            operation()
        end.record(stream)
        end.synchronize()
        return ((time.perf_counter_ns() - clock) / 1e6 / iterations / divisor,
                start.elapsed_time(end) / iterations / divisor)

    def experiment(label, dtype, tokens, index):
        seed = 20261230 + index
        keep, operations, outputs, expected = [], {}, {}, None
        if label != "pointwise_subgraph":
            inputs = create_inputs(label, dtype, 1, tokens, seed=seed)
            expected = oracle(label, inputs)
            change = inputs[0]
            counts_per_call = [int(label == "adaln"), int(label == "residual")]
            children = 3 if label == "adaln" else 2
            for mode in ("baseline", "optimized"):
                out = torch.empty_like(inputs[0])
                scratch = [torch.empty_like(out) for _ in range(2 if label == "adaln" else 1)]
                operations[mode] = bridge.operation(label, inputs, out, scratch, original=mode == "baseline")
                outputs[mode] = out
                keep.extend(scratch)
            def reference():
                return oracle(label, inputs)
            layout = [list(v.stride()) for v in inputs]
        else:
            adaln1 = create_inputs("adaln", dtype, 1, tokens, seed=seed)
            adaln2 = create_inputs("adaln", dtype, 1, tokens, seed=seed + 100)
            residual1 = create_inputs("residual", dtype, 1, tokens, seed=seed + 200)
            residual2 = create_inputs("residual", dtype, 1, tokens, seed=seed + 300)
            change = residual1[0]
            counts_per_call, children = [2, 2], 10
            def reference():
                # Provided branch outputs stand in for attention/MLP; their math is excluded.
                return oracle("residual", (oracle("residual", residual1), residual2[1], residual2[2]))
            expected = reference()
            for mode in ("baseline", "optimized"):
                views = [torch.empty_like(change) for _ in range(4)]
                scratch = [[torch.empty_like(change) for _ in range(n)] for n in (2, 1, 2, 1)]
                nodes = [
                    bridge.operation("adaln", adaln1, views[0], scratch[0], original=mode == "baseline"),
                    bridge.operation("residual", residual1, views[1], scratch[1], original=mode == "baseline"),
                    bridge.operation("adaln", adaln2, views[2], scratch[2], original=mode == "baseline"),
                    bridge.operation("residual", (views[1], residual2[1], residual2[2]), views[3], scratch[3],
                                     original=mode == "baseline")]
                def run(nodes=nodes):
                    for node in nodes:
                        node()
                operations[mode] = run
                outputs[mode] = views[3]
                keep.append((views, scratch, nodes))
            layout = [list(v.stride()) for v in (*adaln1, *residual1, *adaln2, *residual2)]
        graphs, lists = {}, {}
        dispatch_checks = {}
        for mode in ("baseline", "optimized"):
            before = bridge.counts()
            operations[mode]()
            synchronize()
            after = bridge.counts()
            assert_exact(outputs[mode], expected)
            assert [after[i] - before[i] for i in (0, 1)] == (counts_per_call if mode == "optimized" else [0, 0])
            assert after[5] - before[5] == (children if mode == "baseline" else 0)
            dispatch_checks[mode] = [a - b for a, b in zip(after, before)]
            # Core was built graph=false: this is an operator-list replay.
            ic.start_graph_recording()
            operations[mode]()
            lists[mode] = ic.stop_graph_recording()
            for _ in range(warmup):
                lists[mode].run()
            synchronize()
            assert_exact(outputs[mode], expected)
            for _ in range(warmup):
                operations[mode]()
            synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=stream):
                for _ in range(unroll):
                    operations[mode]()
            graphs[mode] = graph
            for _ in range(warmup):
                graph.replay()
            synchronize()
            assert_exact(outputs[mode], expected)
        if label == "pointwise_subgraph":
            # Verify the two AdaLN outputs too; they are boundary outputs of this subgraph.
            for views, scratch, nodes in keep:
                assert_exact(views[0], oracle("adaln", adaln1))
                assert_exact(views[2], oracle("adaln", adaln2))
        kinds = ("api_host", "api_event", "list_host", "list_event", "hipgraph_host", "hipgraph_event")
        timing = {f"{mode}_{kind}_ms": [] for mode in operations for kind in kinds}
        for repeat in range(repeats):
            for mode in (("baseline", "optimized") if repeat % 2 == 0 else ("optimized", "baseline")):
                values = (*measure(operations[mode]), *measure(lists[mode].run),
                          *measure(graphs[mode].replay, unroll))
                for kind, value in zip(kinds, values):
                    timing[f"{mode}_{kind}_ms"].append(value)
        change.add_(0.125)
        synchronize()
        changed = reference()
        for mode in operations:
            for replay in (lists[mode].run, graphs[mode].replay):
                replay()
                synchronize()
                assert_exact(outputs[mode], changed)
        record = dict(op=label, dtype=dtype, shape=[1, tokens, 5376], input_strides=layout, seed=seed,
                      atol=0, rtol=0, bitwise=True, dispatch=dispatch_checks, changed_input_replays=True,
                      warmup=warmup, iterations=iterations, repeats=repeats, unroll=unroll, **timing)
        for kind in kinds:
            for mode in operations:
                record[f"{mode}_{kind}_median_ms"] = statistics.median(timing[f"{mode}_{kind}_ms"])
            record[f"{kind}_speedup"] = record[f"baseline_{kind}_median_ms"] / record[f"optimized_{kind}_median_ms"]
        return record

    checks = []
    for index, (op, dtype) in enumerate(COMBINATIONS):
        # Deliberately unsafe partially overlapping view must be rejected before launching.
        inputs = create_inputs(op, dtype, 1, 7, seed=20261330 + index)
        backing = torch.randn(1, 7, 5377, device="cuda", dtype=inputs[0].dtype)
        x, out = backing[:, :, :-1], backing[:, :, 1:]
        before = bridge.counts()
        try:
            bridge.operation(op, (x, *inputs[1:]), out)()
            raise AssertionError("partial overlap was not rejected")
        except RuntimeError as error:
            assert "partially overlapping" in str(error), str(error)
        assert bridge.counts() == before
        checks.append(dict(check="partial_alias_rejected_without_launch", op=op, dtype=dtype, passed=True))
    paired = []
    cases = [(op, dtype, tokens) for op, dtype in COMBINATIONS for tokens in (1, 128, 512)]
    cases += [("pointwise_subgraph", dtype, tokens) for dtype in ("float16", "bfloat16") for tokens in (1, 128, 512)]
    with (args.output / "paired.jsonl").open("w", buffering=1) as raw:
        for index, (op, dtype, tokens) in enumerate(cases):
            record = experiment(op, dtype, tokens, index)
            paired.append(record)
            raw.write(json.dumps(record) + "\n")
            print(json.dumps({key:record[key] for key in ("op", "dtype", "shape", "api_host_speedup",
                                                         "list_event_speedup", "hipgraph_event_speedup")}), flush=True)
    bridge.clear()
    for library, artifact in zip(libraries, manifest["artifacts"]):
        release = getattr(library, artifact["name"] + "_triton_release")
        release.restype = ctypes.c_int
        assert release() == 0
    result = dict(cases=len(paired), paired=paired, extra_checks=checks, source=str(args.source),
                  counters=bridge.counts(), cleared=True, native_resources_released=True,
                  runtime_imports={"ninetoothed": ninetoothed.__file__, "torch": torch.__file__, "infinicore": ic.__file__},
                  loaded_libraries=sorted({line.split(maxsplit=5)[-1] for line in Path("/proc/self/maps").read_text().splitlines()
                                           if "/libinfini" in line or "/core-h3-fusion.so" in line or ".native.so" in line}),
                  api_timing="same host C ABI and preallocated scratch/output policy",
                  list_timing="Core graph=false, cached operator-list replay",
                  hipgraph_timing="actual Torch HIP graph, 32 calls or pointwise subgraphs per graph, divided by 32",
                  scope="pointwise subgraph only; supplied normalization and branch outputs; excludes RMS, MHA, GEMM and MLP math")
    (args.output / "benchmark.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({key:value for key,value in result.items() if key != "paired"}), flush=True)


if __name__ == "__main__":
    main()
