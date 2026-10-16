#  RMSNorm 优化复现

使用九齿实现 128 维 RMSNorm，通过原生桥接接入 InfiniCore，并重新运行正确性与配对性能测试。仓库仅包含实现、必要脚本和依赖版本；测试结果在运行时生成。

## 运行环境

Linux x86_64，海光 gfx906，Python 3.11.9、镜像自带的海光 PyTorch 2.9.0、HIP 6.3.26093，以及 Git、curl、make、CMake 和 g++。DTK 位于 `/opt/dtk`，系统 Python 位于 `/usr/local/bin/python`，Torch 库位于 `/usr/local/lib/python3.11/site-packages/torch/lib` 和 `torch.libs`，与基线镜像布局一致。

## 运行

```bash
git clone --branch rms128-20261016 --depth 1 https://github.com/Yuuki-git/songshan3.git songshan3
cd songshan3
bash scripts/reproduce.sh
```

脚本获取固定版本的 InfiniCore、九齿和构建依赖，在 `venvs/h3` 安装额外 Python 包，以并发 8 构建原始 Core 和主机桥接，然后串行运行 RMSNorm 测试。原始源码保存在 `Projects`，九齿 `development/ninetoothed-hygon`。



已有同工程独立构建环境时，可用 `bash scripts/reproduce.sh --skip-build` 直接重新构建桥接并运行测试。工程目录名保持为 `songshan3`，父目录可自由选择，无需团队 SSH 账号或固定用户目录。

## 代码与输出

- `benchmarks/hygon_rmsnorm_probe.py`：九齿算子表达和每程序一行的调度。
- `development/hygon_aot.py`：海光 HIP AOT 主机适配。
- `development/core_ninetoothed_bridge.cpp`：Core 原生分派桥接。
- `benchmarks/hygon_rmsnorm_acceptance.py`：正确性、图重放及性能测试。
- `benchmarks/h3_operator_baseline.py` 和 `hygon_native.py`：基线与原生导出所需的辅助代码。

每次运行创建新的 `results/hygon-rmsnorm-native-*` 目录。`acceptance/acceptance.json` 为验收汇总，`correctness.jsonl` 为逐例误差，`paired.jsonl` 为五轮配对计时。两侧均使用相同 Core API、相同输入和预分配输出，各预热 10 次，每轮 100 次，交替顺序重复 5 轮，取单次完成耗时的中位数。

上游版本固定在准备脚本中：InfiniCore `f32e2fbb2a9b2ca912b5711f4f3836c403196a74`，NineToothed `2c5eb118592f4d2b0f448d16d2e9a3a5c0fbc900`，xmake-repo `bd29d77f242e247589457638272d9936a24a3fe0`；xmake 为 3.0.5，Boost 为 1.92.0，pybind11 为 3.1.0。Python 包版本见 `requirements.txt`。
