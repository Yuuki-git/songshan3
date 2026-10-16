"""Find a correct NineToothed RMSNorm schedule before native integration."""
import argparse
import functools
import json
from pathlib import Path
import traceback

import ninetoothed as nt
import ninetoothed.language as ntl
from ninetoothed import Tensor
import torch


def arrangement(x, weight, epsilon, out, rows_per_program):
    x = x.tile((rows_per_program, 128))
    weight = weight.tile((128,)).unsqueeze(0).expand(x.shape)
    return x, weight, epsilon, out.tile((rows_per_program, 128))


def application(x, weight, epsilon, out):
    xf = ntl.cast(x, ntl.float32)
    mean_square = ntl.sum(xf * xf, axis=1) / 128.0
    # This gfx906 stack miscomputes the tested FP32 sqrt/rsqrt path.
    # Keep the sum in FP32; use NineToothed FP64 scalar sqrt and reciprocal,
    # then return to FP32 before applying the unchanged output dtype.
    denominator = ntl.cast(mean_square, ntl.float64) + ntl.cast(epsilon, ntl.float64)
    scale = ntl.cast(1.0 / ntl.sqrt(denominator), ntl.float32)
    out = xf * scale[:, None] * ntl.cast(weight, ntl.float32)


def make_kernel(dtype, weight_dtype, rows_per_program=1, warps=1, **kwargs):
    tensors = (Tensor(shape=(None, 128), dtype=getattr(nt, dtype), other=0),
               Tensor(shape=(128,), dtype=getattr(nt, weight_dtype)),
               Tensor(0, dtype=nt.float32), Tensor(shape=(None, 128), dtype=getattr(nt, dtype)))
    return nt.make(functools.partial(arrangement, rows_per_program=rows_per_program), application, tensors,
                   platform="hygon-gfx906", num_warps=warps, num_stages=1, max_num_configs=1, **kwargs)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    records = []
    for tile_rows, warps in ((1, 1), (2, 1), (4, 1), (8, 1), (4, 4)):
        record = dict(tile_rows=tile_rows, warps=warps, seed=20261016)
        try:
            kernel = make_kernel("float16", "float32", tile_rows, warps)
            source = Path(kernel._source).read_text()
            (args.output / f"rows{tile_rows}-warps{warps}.generated.py").write_text(source)
            record["schedule"] = str(kernel._artifact.metadata.get("ssa_schedule"))
            for rows, stride in ((7, 1), (56, 2), (7168, 1)):
                torch.manual_seed(20261016)
                x = torch.randn(rows, 128 * stride, device="cuda", dtype=torch.float16)[:, ::stride]
                weight = torch.randn(128, device="cuda")
                actual = torch.full((rows, 128), float("nan"), device="cuda", dtype=x.dtype)
                kernel(x, weight, 1e-5, actual)
                torch.cuda.synchronize()
                host, hw = x.cpu().double(), weight.cpu().double()
                expected = (host * torch.rsqrt(host.square().mean(-1, keepdim=True) + 1e-5) * hw).to(x.dtype)
                torch.testing.assert_close(actual.cpu(), expected, atol=2e-3, rtol=2e-3)
            record["passed"] = True
        except Exception:
            record.update(passed=False, error=traceback.format_exc())
        records.append(record)
        print(json.dumps(record), flush=True)
    (args.output / "probe.json").write_text(json.dumps(records, indent=2) + "\n")
    return int(not any(record["passed"] for record in records))


if __name__ == "__main__":
    raise SystemExit(main())
