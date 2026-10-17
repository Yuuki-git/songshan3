"""Five-path exact fusion checks through the same original/optimized Core API."""
import argparse
import ctypes
import hashlib
import json
from pathlib import Path
import torch
import infinicore as ic

from h3_operator_baseline import synchronize
from hygon_native import export_native, view
from hygon_fusion3d import create_inputs, make_kernel, oracle
from hygon_fusion3d_host import Bridge, COMBINATIONS, assert_exact


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bridge", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    ic.set_device(ic.device("cuda", 0))
    bridge = Bridge(args.bridge)
    stream = torch.cuda.ExternalStream(bridge.library.songshan3_fusion_stream())
    torch.cuda.set_stream(stream)
    kernels, artifacts = {}, []
    for op, dtype in COMBINATIONS:
        name = f"h3_{op}3d_{dtype}"
        directory = args.output / "kernels" / name
        directory.mkdir(parents=True, exist_ok=True)
        jit = make_kernel(op, dtype, kernel_name=name + "_jit")
        aot = make_kernel(op, dtype, caller="cuda", kernel_name=name, output_dir=directory)
        library, launch, artifact = export_native(aot, directory)
        kernels[op, dtype] = dict(jit=jit, aot=aot, native=launch, library=library, artifact=artifact)
        artifacts.append(artifact)
        print(json.dumps(dict(compiled=name)), flush=True)
    bridge.register([kernels[combo]["native"] for combo in COMBINATIONS])
    specs = ((1, 1, 1, "normal"), (1, 7, 1, "normal"), (2, 7, 2, "normal"),
             (1, 17, 1, "mixed"), (1, 7, 1, "large"), (1, 65, 1, "tiny"),
             (1, 7, 2, "zeros"))
    records = []
    with (args.output / "correctness.jsonl").open("w", buffering=1) as raw:
        for combo_index, (op, dtype) in enumerate(COMBINATIONS):
            kernel = kernels[op, dtype]
            for case_index, (batch, tokens, stride, distribution) in enumerate(specs):
                seed = 20261120 + combo_index * 100 + case_index
                inputs = create_inputs(op, dtype, batch, tokens, stride=stride, distribution=distribution, seed=seed)
                saved = [v.cpu().clone() for v in inputs]
                expected = oracle(op, inputs)
                backings = [torch.full((batch, tokens, 10752), float("nan"), device="cuda", dtype=inputs[0].dtype) for _ in range(5)]
                outputs = [b[:, :, ::2] for b in backings]
                workspaces = [torch.empty(batch, tokens, 5376, device="cuda", dtype=inputs[0].dtype)
                              for _ in range(2 if op == "adaln" else 1)]
                synchronize()
                kernel["jit"](*inputs, outputs[0])
                kernel["aot"](*inputs, outputs[1])
                status = kernel["native"](stream.cuda_stream, *(view(v) for v in (*inputs, outputs[2])))
                assert status == 0, status
                before = bridge.counts()
                bridge.operation(op, inputs, outputs[3], workspaces)()
                bridge.operation(op, inputs, outputs[4], workspaces, original=True)()
                synchronize()
                kind = 0 if op == "adaln" else 1
                after = bridge.counts()
                assert after[kind] == before[kind] + 1 and after[2 + kind] == before[2 + kind] + 1
                assert after[5] == before[5] + (3 if op == "adaln" else 2)
                for out in outputs:
                    assert_exact(out, expected)
                for backing in backings:
                    assert bool(backing[:, :, 1::2].isnan().all())
                for value, original in zip(inputs, saved):
                    assert_exact(value, original)
                record = dict(op=op, dtype=dtype, shape=list(inputs[0].shape), input_strides=[list(v.stride()) for v in inputs],
                              output_stride=list(outputs[0].stride()), distribution=distribution, seed=seed,
                              paths=["jit", "python_aot", "native_cpp", "core_fused", "core_original"],
                              passed=True, bitwise=True, output_guards=True, inputs_unchanged=True, atol=0, rtol=0)
                records.append(record)
                raw.write(json.dumps(record) + "\n")
            print(json.dumps(dict(op=op, dtype=dtype, five_path_cases=len(specs), passed=True)), flush=True)

    extras = []
    with (args.output / "extra.jsonl").open("w", buffering=1) as raw:
        for combo_index, (op, dtype) in enumerate(COMBINATIONS):
            kind = 0 if op == "adaln" else 1
            inputs = create_inputs(op, dtype, 1, 7, seed=20261130 + combo_index)
            expected = oracle(op, inputs)
            before = bridge.count(kind)
            bridge.operation(op, inputs, inputs[0])()
            synchronize()
            assert_exact(inputs[0], expected)
            assert bridge.count(kind) == before + 1
            record = dict(check="core_inplace_x", op=op, dtype=dtype, passed=True, bitwise=True)
            extras.append(record); raw.write(json.dumps(record) + "\n")

            nondefault = torch.cuda.Stream()
            with torch.cuda.stream(nondefault):
                inputs = create_inputs(op, dtype, 1, 7, stride=2, seed=20261201 + combo_index)
                output = torch.empty_like(inputs[0])
                expected = oracle(op, inputs)
                status = kernels[op, dtype]["native"](nondefault.cuda_stream, *(view(v) for v in (*inputs, output)))
                assert status == 0, status
                nondefault.synchronize()
                assert_exact(output, expected)
            record = dict(check="native_nondefault_stream", op=op, dtype=dtype, passed=True, bitwise=True)
            extras.append(record); raw.write(json.dumps(record) + "\n")

            for shape, fallback_dtype, broadcast in (((1, 7, 1024), dtype, False),
                                                      ((1, 7, 5376), "float32", False),
                                                      ((2, 7, 5376), dtype, True)):
                torch.manual_seed(20261210 + combo_index)
                dt = getattr(torch, fallback_dtype)
                x = torch.randn(shape, device="cuda", dtype=dt)
                a = torch.randn((1, 1, shape[-1]) if broadcast else shape, device="cuda", dtype=dt)
                b = torch.randn_like(a) if op == "adaln" else torch.randn_like(x)
                if broadcast:
                    # Original Add descriptors require equal shapes. Preserve the
                    # broadcast as same-shape zero-stride views, without copying.
                    a = a.expand(shape)
                    b = b.expand(shape)
                expected = oracle(op, (x, a, b))
                output = torch.empty_like(x)
                before = bridge.count(2 + kind)
                bridge.operation(op, (x, a, b), output)()
                synchronize()
                assert_exact(output, expected)
                assert bridge.count(2 + kind) == before + 1
                record = dict(check="real_original_fallback", op=op, dtype=fallback_dtype, shape=list(shape),
                              input_strides=[list(value.stride()) for value in (x, a, b)],
                              broadcast=broadcast, passed=True, bitwise=True,
                              fp32_add_rounding="toward_negative_infinity" if dt == torch.float32 else None)
                extras.append(record); raw.write(json.dumps(record) + "\n")

            inputs = tuple(torch.empty(1, 0, 5376, device="cuda", dtype=getattr(torch, dtype)) for _ in range(3))
            output = torch.empty_like(inputs[0])
            before = bridge.counts()
            bridge.operation(op, inputs, output)()
            synchronize()
            after = bridge.counts()
            assert after[4] == before[4] + 1 and after[kind] == before[kind] and after[5] == before[5]
            record = dict(check="empty_no_launch", op=op, dtype=dtype, passed=True)
            extras.append(record); raw.write(json.dumps(record) + "\n")

            inputs = create_inputs(op, dtype, 1, 7, seed=20261220 + combo_index)
            output = torch.empty_like(inputs[0])
            expected = oracle(op, inputs)
            bridge.clear()
            before = bridge.count(2 + kind)
            bridge.operation(op, inputs, output)()
            synchronize()
            assert_exact(output, expected)
            assert bridge.count(2 + kind) == before + 1
            bridge.register([kernels[combo]["native"] for combo in COMBINATIONS])
            record = dict(check="cleared_registration_fallback", op=op, dtype=dtype, passed=True, bitwise=True)
            extras.append(record); raw.write(json.dumps(record) + "\n")
        print(json.dumps(dict(extra_cases=len(extras), passed=True)), flush=True)
    bridge.clear()
    for kernel in kernels.values():
        release = getattr(kernel["library"], kernel["artifact"]["name"] + "_triton_release")
        release.restype = ctypes.c_int
        assert release() == 0
        library = ctypes.CDLL(str(kernel["aot"]._library))
        library.ninetoothed_triton_release.restype = ctypes.c_int
        assert library.ninetoothed_triton_release() == 0
    result = dict(five_path_cases=len(records), five_path_outputs=5 * len(records), extra_cases=len(extras),
                  counters=bridge.counts(), artifacts=artifacts, bridge=str(args.bridge),
                  bridge_sha256=hashlib.sha256(args.bridge.read_bytes()).hexdigest(),
                  registration_cleared=True, native_resources_released=True, width=5376, block=256, warps=1,
                  enable_fp_fusion=False, numerical_requirement="bit pattern equality to CPU float64 with original dtype rounding at every original operation")
    (args.output / "acceptance.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k:v for k,v in result.items() if k != "artifacts"}), flush=True)


if __name__ == "__main__":
    main()
