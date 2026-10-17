"""H3 three-dimensional pointwise fusions, generated exclusively by NineToothed."""
import functools
import ninetoothed as nt
import ninetoothed.language as ntl
from ninetoothed import Tensor
import torch


def arrangement(x, a, b, out, block):
    values = []
    for value in (x, a, b, out):
        value = value.tile((1, 1, block))
        value.dtype = value.dtype.squeeze((0, 1))
        values.append(value)
    return tuple(values)


def adaln(x, scale, shift, out):
    xf = ntl.cast(x, ntl.float32)
    product = ntl.cast(ntl.cast(xf * ntl.cast(scale, ntl.float32), out.dtype), ntl.float32)
    summed = ntl.cast(ntl.cast(xf + product, out.dtype), ntl.float32)
    out = summed + ntl.cast(shift, ntl.float32)


def residual(x, gate, branch, out):
    product = ntl.cast(ntl.cast(ntl.cast(gate, ntl.float32) * ntl.cast(branch, ntl.float32), out.dtype), ntl.float32)
    out = ntl.cast(x, ntl.float32) + product


def make_kernel(op, dtype, **kwargs):
    tensors = tuple(Tensor(shape=(None, None, 5376), dtype=getattr(nt, dtype)) for _ in range(4))
    return nt.make(functools.partial(arrangement, block=256), globals()[op], tensors,
                   platform="hygon-gfx906", num_warps=1, num_stages=1, max_num_configs=1,
                   enable_fp_fusion=False, **kwargs)


def oracle(op, inputs):
    dtype = inputs[0].dtype
    x, a, b = (value.cpu().double() for value in inputs)
    def add(left, right):
        exact = left + right
        rounded = exact.to(dtype)
        if dtype == torch.float32:
            # The fixed original Core AddOp<float> calls __fadd_rd, not RN.
            # Keep that original contract on real FP32 fallback, with zero tolerance.
            lower = torch.nextafter(rounded, torch.full_like(rounded, float("-inf")))
            rounded = torch.where(rounded.double() > exact, lower, rounded)
            positive_zero_pair = (left == 0) & (right == 0) & ~torch.signbit(left) & ~torch.signbit(right)
            negative_zero = torch.copysign(torch.zeros_like(rounded), -torch.ones_like(rounded))
            rounded = torch.where((exact == 0) & ~positive_zero_pair, negative_zero, rounded)
        return rounded.double()
    if op == "adaln":
        return add(add(x, (x * a).to(dtype).double()), b).to(dtype)
    return add(x, (a * b).to(dtype).double()).to(dtype)


def create_inputs(op, dtype, batch, tokens, width=5376, stride=1, distribution="normal", seed=20261110):
    torch.manual_seed(seed)
    dt = getattr(torch, dtype)
    x = torch.randn(batch, tokens, width * stride, device="cuda", dtype=dt)[:, :, ::stride]
    modulation = torch.randn(batch, tokens, 6 * width * stride, device="cuda", dtype=dt)[:, :, ::stride]
    if op == "adaln":
        a, b = modulation[:, :, width:2 * width], modulation[:, :, :width]
    else:
        a = modulation[:, :, 2 * width:3 * width]
        b = torch.randn_like(x)
    if distribution == "zeros":
        for value in (x, a, b):
            value.zero_()
    elif distribution == "tiny":
        for value in (x, a, b):
            value.mul_(1e-4)
    elif distribution == "large":
        for value in (x, a, b):
            value.mul_(32)
    elif distribution == "mixed":
        x[:, :, ::2].mul_(1e-4)
        a[:, :, 1::2].mul_(16)
    return x, a, b
