"""NineToothed RMS128 with true token/head strides, including packed Q/K views."""
import argparse
import functools
import json
from pathlib import Path
import traceback

import ninetoothed as nt
import ninetoothed.language as ntl
from ninetoothed import Tensor
import torch

from hygon_rmsnorm_probe import application as row_application

COMBINATIONS = (("float16", "float16"), ("float16", "float32"),
                ("bfloat16", "bfloat16"), ("bfloat16", "float32"), ("float32", "float32"))
TOLERANCES = {"float16": (2e-3, 2e-3), "bfloat16": (1e-2, 1e-2), "float32": (1e-5, 1e-4)}


def arrangement(x, weight, epsilon, out, vector):
    x = x.tile((1, 1, 128))
    out = out.tile((1, 1, 128))
    squeezed = (0, 1) if vector else 0
    x.dtype = x.dtype.squeeze(squeezed)
    out.dtype = out.dtype.squeeze(squeezed)
    weight = weight.tile((128,)).unsqueeze(0).unsqueeze(0).expand(x.shape)
    return x, weight, epsilon, out


def vector_application(x, weight, epsilon, out):
    xf = ntl.cast(x, ntl.float32)
    mean_square = ntl.sum(xf * xf, axis=0) / 128.0
    denominator = ntl.cast(mean_square, ntl.float64) + ntl.cast(epsilon, ntl.float64)
    scale = ntl.cast(1.0 / ntl.sqrt(denominator), ntl.float32)
    out = xf * scale * ntl.cast(weight, ntl.float32)


def make_kernel(dtype, weight_dtype, vector=False, **kwargs):
    tensors = (Tensor(shape=(None, None, 128), dtype=getattr(nt, dtype), other=0),
               Tensor(shape=(128,), dtype=getattr(nt, weight_dtype)),
               Tensor(0, dtype=nt.float32),
               Tensor(shape=(None, None, 128), dtype=getattr(nt, dtype)))
    return nt.make(functools.partial(arrangement, vector=vector),
                   vector_application if vector else row_application, tensors,
                   platform="hygon-gfx906", num_warps=1, num_stages=1, max_num_configs=1, **kwargs)


def oracle(x, weight, epsilon):
    host, hw = x.cpu().double(), weight.cpu().double()
    return (host * torch.rsqrt(host.square().mean(-1, keepdim=True) + epsilon) * hw).to(x.dtype)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--generate-only", action="store_true", help="Generate schedules and source without accessing a GPU")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    records = []
    with (args.output / "probe.jsonl").open("w", buffering=1) as raw:
        for vector in (False, True):
            for combo_index, (dtype, weight_dtype) in enumerate(COMBINATIONS):
                candidate = f"rms3d_{'vector' if vector else 'row'}_{dtype}_{weight_dtype}"
                try:
                    kernel = make_kernel(dtype, weight_dtype, vector, kernel_name=candidate)
                    (args.output / (candidate + ".generated.py")).write_text(Path(kernel._source).read_text())
                    (args.output / (candidate + ".schedule.txt")).write_text(str(kernel._artifact.metadata.get("ssa_schedule")))
                    if args.generate_only:
                        records.append(dict(candidate=candidate, vector=vector, dtype=dtype, weight_dtype=weight_dtype,
                                            generated=True, passed=False, executed=False))
                        print(json.dumps(records[-1]), flush=True)
                        continue
                    for tokens, heads, layout in ((1, 56, "q"), (7, 56, "k"),
                                                   (17, 7, "strided"), (65, 1, "contiguous")):
                        seed = 20261019 + combo_index
                        torch.manual_seed(seed)
                        if layout in ("q", "k"):
                            packed = torch.randn(tokens, heads * 3, 128, device="cuda", dtype=getattr(torch, dtype))
                            start = 0 if layout == "q" else heads
                            x = packed[:, start:start + heads]
                        elif layout == "strided":
                            x = torch.randn(tokens, heads * 2, 256, device="cuda", dtype=getattr(torch, dtype))[:, ::2, ::2]
                        else:
                            x = torch.randn(tokens, heads, 128, device="cuda", dtype=getattr(torch, dtype))
                        weight = torch.randn(256, device="cuda", dtype=getattr(torch, weight_dtype))[::2]
                        backing = torch.full((tokens, heads * 2, 256), float("nan"), device="cuda", dtype=x.dtype)
                        output = backing[:, ::2, ::2]
                        kernel(x, weight, 1e-5, output)
                        torch.cuda.synchronize()
                        expected = oracle(x, weight, 1e-5)
                        torch.testing.assert_close(output.cpu(), expected, atol=TOLERANCES[dtype][0], rtol=TOLERANCES[dtype][1])
                        record = dict(candidate=candidate, vector=vector, dtype=dtype, weight_dtype=weight_dtype,
                                      shape=list(x.shape), input_stride=list(x.stride()), weight_stride=list(weight.stride()),
                                      output_stride=list(output.stride()), layout=layout, seed=seed, epsilon=1e-5,
                                      atol=TOLERANCES[dtype][0], rtol=TOLERANCES[dtype][1], passed=True,
                                      max_error=(output.cpu().float() - expected.float()).abs().max().item())
                        records.append(record)
                        raw.write(json.dumps(record) + "\n")
                    print(json.dumps(dict(candidate=candidate, passed=4)), flush=True)
                except Exception:
                    record = dict(candidate=candidate, vector=vector, dtype=dtype, weight_dtype=weight_dtype,
                                  passed=False, error=traceback.format_exc())
                    records.append(record)
                    raw.write(json.dumps(record) + "\n")
                    print(json.dumps(record), flush=True)
    passed_candidates = [vector for vector in (False, True)
                         if sum(bool(r["passed"]) for r in records if r["vector"] == vector) == 20
                         and all(r["passed"] for r in records if r["vector"] == vector)]
    result = dict(passed=sum(r["passed"] for r in records), total=len(records), passed_candidates=passed_candidates)
    result["generate_only"] = args.generate_only
    (args.output / "probe.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)
    return int(not passed_candidates) if not args.generate_only else int(not all(r.get("generated") for r in records))


if __name__ == "__main__":
    raise SystemExit(main())
