"""Bounded development baseline for H3 operators; this is not FL2VA evaluation.

Timings include Python submission and completed device execution. Both runtimes
are synchronized, so different default streams cannot produce premature timing.
Official evaluation scripts and source checkouts are never modified.
"""

import argparse
import csv
import json
import os
from pathlib import Path
import platform
import statistics
import sys
import time
import traceback

import infinicore as ic
import infinicore.nn.functional as F
import torch


def cases():
    result = []
    for dtype in ("float16", "bfloat16", "float32"):
        weights = [dtype] if dtype == "float32" else [dtype, "float32"]
        for rows, width, padded in [(56, 128, False), (7168, 128, False),
                                     (28672, 128, False), (512, 128, True),
                                     (1, 5376, False), (16, 5376, False),
                                     (128, 5376, False), (512, 5376, False)]:
            for weight_dtype in weights:
                result.append(dict(op="rms_norm", dtype=dtype,
                                   weight_dtype=weight_dtype, shape=[rows, width],
                                   padded=padded, epsilon=1e-5))
    for dtype in ("float16", "bfloat16"):
        for op in ("swiglu", "silu", "add", "mul"):
            for rows in (1, 128, 512):
                width = 14336 if op == "swiglu" else 5376
                result.append(dict(op=op, dtype=dtype, shape=[rows, width]))
        for rows in (1, 128):
            result.append(dict(op="matmul", dtype=dtype, shape=[rows, 5376],
                               output_width=14336, weight_transposed=True))
        for tokens in (1, 128, 512):
            result.append(dict(op="mrope", dtype=dtype, shape=[tokens, 56 * 128],
                               heads=56, head_size=128, rotary_dim=96,
                               sections=[16, 16, 16]))
    for i, case in enumerate(result):
        case.update(id=f"{case['op']}-{i:03}", seed=20261015 + i)
    return result


def synchronize():
    ic.sync_device()
    torch.cuda.synchronize()


def wrap(tensor):
    # Preserve strides; the convenience from_torch API assumes contiguous input.
    return ic.strided_from_blob(tensor.data_ptr(), list(tensor.shape),
                                list(tensor.stride()),
                                dtype=getattr(ic, str(tensor.dtype).split(".")[-1]),
                                device=ic.device("cuda", 0))


def create(case):
    torch.manual_seed(case["seed"])
    dtype = getattr(torch, case["dtype"])
    rows, width = case["shape"]
    x = torch.randn(rows, width * (3 if case.get("padded") else 1),
                    device="cuda", dtype=dtype)[:, :width]
    y = torch.empty((rows, width), device="cuda", dtype=dtype)
    op = case["op"]
    atol, rtol = {"float16": (2e-3, 2e-3), "bfloat16": (1e-2, 1e-2),
                  "float32": (1e-5, 1e-4)}[case["dtype"]]
    keep = [x, y]
    if op == "rms_norm":
        weight = torch.randn(width, device="cuda", dtype=getattr(torch, case["weight_dtype"]))
        keep.append(weight)
        cx, cw, cy = map(wrap, (x, weight, y))

        def reference():
            xf = x.float()
            return (xf * torch.rsqrt(xf.square().mean(-1, keepdim=True) + case["epsilon"])
                    * weight.float()).to(dtype)

        def candidate():
            F.rms_norm(cx, [width], cw, case["epsilon"], out=cy)

    elif op in ("swiglu", "silu", "add", "mul"):
        other = torch.randn_like(x)
        keep.append(other)
        cx, co, cy = map(wrap, (x, other, y))
        if op == "swiglu":
            atol, rtol = ((1e-3, 1e-3) if dtype == torch.float16 else (5e-3, 1e-2))

            def reference():
                return (x.float() * torch.nn.functional.silu(other.float())).to(dtype)

            def candidate():
                F.swiglu(cx, co, out=cy)

        elif op == "silu":
            def reference():
                return torch.nn.functional.silu(x)

            def candidate():
                F.silu(cx, out=cy)

        else:
            def reference():
                return torch.add(x, other) if op == "add" else torch.mul(x, other)

            def candidate():
                getattr(ic, op)(cx, co, out=cy)

    elif op == "matmul":
        weight = (torch.randn(case["output_width"], width, device="cuda", dtype=dtype) * 0.02).t()
        y = torch.empty(rows, case["output_width"], device="cuda", dtype=dtype)
        keep.extend((weight, y))
        cx, cw, cy = map(wrap, (x, weight, y))
        atol, rtol = ((2e-2, 2e-2) if dtype == torch.float16 else (5e-2, 5e-2))

        def reference():
            return x @ weight

        def candidate():
            ic.matmul(cx, cw, out=cy)

    elif op == "mrope":
        k = torch.randn_like(x)
        ky = torch.empty_like(k)
        half = case["rotary_dim"] // 2
        angles = torch.randn(256, half, device="cuda")
        cache = torch.cat((angles.cos(), angles.sin()), dim=-1).to(dtype)
        cos, sin = cache[:, :half], cache[:, half:]
        pos = torch.stack([torch.arange(rows, device="cuda") % 256,
                           (torch.arange(rows, device="cuda") + 1) % 256,
                           (torch.arange(rows, device="cuda") + 2) % 256])
        axes = torch.cat([torch.full((size,), axis, dtype=torch.long, device="cuda")
                          for axis, size in enumerate(case["sections"])])
        dim = torch.arange(half, device="cuda")
        keep.extend((k, ky, cos, sin, pos, axes, dim))
        cx, ck, cc, cs, cp, cy, cky = map(wrap, (x, k, cos, sin, pos, y, ky))

        def rotate(value):
            lookup = pos[axes].t()
            c, s = cos[lookup, dim].float()[:, None], sin[lookup, dim].float()[:, None]
            value = value.view(rows, case["heads"], case["head_size"]).float()
            a, b = value[..., :half], value[..., half:case["rotary_dim"]]
            return torch.cat((a * c - b * s, b * c + a * s,
                              value[..., case["rotary_dim"]:]), dim=-1).to(dtype).reshape(rows, width)

        def reference():
            return rotate(x), rotate(k)

        def candidate():
            ic.mrope(cx, ck, cc, cs, cp, case["head_size"], case["rotary_dim"],
                      *case["sections"], False, out=(cy, cky))

        y = (y, ky)
        atol = rtol = 1e-3 if dtype == torch.float16 else 2e-2
    else:
        raise ValueError(op)
    return reference, candidate, y, keep, atol, rtol


def elapsed(function, iterations):
    synchronize()
    start = time.perf_counter_ns()
    for _ in range(iterations):
        function()
    synchronize()
    return (time.perf_counter_ns() - start) / 1e6 / iterations


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    torch.cuda.reset_peak_memory_stats()
    suite_started = time.perf_counter()
    suite = cases()
    manifest = dict(schema="songshan3-h3-operators-v1", cases=suite,
                    warmup=args.warmup, iterations=args.iterations, repeats=args.repeats,
                    timing="synchronized host submission plus completed device execution; preallocated Core outputs",
                    reference_timing="PyTorch expressions allocate outputs; reference ratio is not an optimization speedup",
                    pod=platform.node(), python=sys.version, torch=torch.__version__, hip=torch.version.hip,
                    infinicore=ic.__file__, ntops_enabled=ic.use_ntops,
                    dimensions="InfiniLM source defaults, not verified official FL2VA input shapes")
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    records = []
    with (args.output / "results.jsonl").open("w", buffering=1) as raw:
        for case in suite:
            record = dict(case)
            try:
                ref, op, actual, keep, atol, rtol = create(case)
                record["input_stride"] = list(keep[0].stride())
                if case["op"] in ("rms_norm", "matmul"):
                    record["weight_stride"] = list(keep[2].stride())
                if case["op"] == "mrope":
                    record["cos_stride"] = list(keep[4].stride())
                    record["sin_stride"] = list(keep[5].stride())
                expected = ref()
                synchronize()
                op()
                synchronize()
                actuals = actual if isinstance(actual, tuple) else (actual,)
                expecteds = expected if isinstance(expected, tuple) else (expected,)
                for a, e in zip(actuals, expecteds):
                    torch.testing.assert_close(a, e, atol=atol, rtol=rtol)
                if case["op"] == "rms_norm":
                    host_x, host_weight = keep[0].cpu().double(), keep[2].cpu().double()
                    oracle = (host_x * torch.rsqrt(host_x.square().mean(-1, keepdim=True)
                                                   + case["epsilon"]) * host_weight).to(actual.dtype)
                    torch.testing.assert_close(actual.cpu(), oracle, atol=atol, rtol=rtol)
                    torch.testing.assert_close(expected.cpu(), oracle, atol=atol, rtol=rtol)
                    record.update(cpu_oracle="float64 RMSNorm",
                                  cpu_oracle_max_abs_error=(actual.cpu().float() - oracle.float()).abs().max().item())
                record.update(atol=atol, rtol=rtol,
                              max_abs_error=max((a.float() - e.float()).abs().max().item()
                                                for a, e in zip(actuals, expecteds)))
                for _ in range(args.warmup):
                    ref()
                    op()
                core_times, ref_times = [], []
                for repeat in range(args.repeats):
                    order = [(ref, ref_times), (op, core_times)]
                    if repeat % 2:
                        order.reverse()
                    for function, values in order:
                        values.append(elapsed(function, args.iterations))
                record.update(status="passed", core_ms=core_times, reference_ms=ref_times,
                              core_median_ms=statistics.median(core_times),
                              reference_median_ms=statistics.median(ref_times))
                print(f"{case['id']} {case['dtype']} {case['shape']}: PASS; Core {record['core_median_ms']:.6f} ms", flush=True)
                del ref, op, actual, expected, actuals, expecteds, keep
            except Exception:
                record.update(status="failed", error=traceback.format_exc())
                print(f"{case['id']}: FAIL\n{record['error']}", flush=True)
            raw.write(json.dumps(record) + "\n")
            records.append(record)
    fields = ["id", "op", "dtype", "shape", "weight_dtype", "padded", "status",
              "atol", "rtol", "max_abs_error", "core_median_ms", "reference_median_ms"]
    with (args.output / "results.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records)
    passed = sum(record["status"] == "passed" for record in records)
    import resource
    (args.output / "resources.json").write_text(json.dumps(dict(
        wall_seconds=time.perf_counter() - suite_started,
        torch_peak_allocated_bytes=torch.cuda.max_memory_allocated(),
        torch_peak_reserved_bytes=torch.cuda.max_memory_reserved(),
        driver_free_bytes_at_end=torch.cuda.mem_get_info()[0],
        process_peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        note="Torch memory statistics exclude allocations made directly by native libraries; this is not model peak memory",
    ), indent=2) + "\n")
    print(f"RESULT: {passed}/{len(records)} passed", flush=True)
    return int(passed != len(records))


if __name__ == "__main__":
    sys.exit(main())
