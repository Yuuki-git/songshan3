"""Host entry point forwarding NineToothed AOT options to official Triton CLI.

Triton 3.8.0's CLI omits enable_fp_fusion. This process-local wrapper supplies
that compiler option while retaining its official source parsing, compilation,
HIP templates and export logic. It contains no device computation.
"""
import os
import runpy

import triton

original_compile = triton.compile


def compile_with_options(*args, **kwargs):
    value = os.environ["NINETOOTHED_ENABLE_FP_FUSION"]
    if value not in ("0", "1"):
        raise ValueError("NINETOOTHED_ENABLE_FP_FUSION must be 0 or 1")
    kwargs["options"] = dict(kwargs.get("options") or {}, enable_fp_fusion=value == "1")
    return original_compile(*args, **kwargs)


if __name__ == "__main__":
    triton.compile = compile_with_options
    runpy.run_module("triton.tools.compile", run_name="__main__")
