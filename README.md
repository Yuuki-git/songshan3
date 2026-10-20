# MiniMax H3 算子优化复现

| 优化对象 | 输入形状 | 输入精度 | 九齿实现 |
| --- | --- | --- | --- |
| 二维 RMSNorm | `[M, 128]` | FP16、BF16、FP32 | [hygon_rmsnorm_probe.py](benchmarks/hygon_rmsnorm_probe.py) |
| 三维 Q/K RMSNorm | `[tokens, heads, 128]` | FP16、BF16、FP32 | [hygon_rmsnorm3d_probe.py](benchmarks/hygon_rmsnorm3d_probe.py) |
| 三维 AdaLN 与门控残差融合 | `[batch, tokens, 5376]` | FP16、BF16 | [hygon_fusion3d.py](benchmarks/hygon_fusion3d.py) |

三项算子已在独立海光环境完成离线重建和运行。复现保留原有数值比较与性能计时。

## 环境

Linux x86_64、海光 gfx906，Python 3.11.9、镜像自带海光 PyTorch 2.9.0、HIP 6.3.26093，以及 Git、curl、make、CMake、g++、flock 和 timeout。DTK 位于 `/opt/dtk`，Torch 库位于 `/usr/local/lib/python3.11/site-packages/torch/lib` 和 `torch.libs`。

## 运行

```bash
git clone --branch main --depth 1 https://github.com/Yuuki-git/songshan3.git songshan3
cd songshan3
bash scripts/reproduce.sh
```

默认运行全部三项，也可单独选择：

```bash
bash scripts/reproduce.sh --suite rms2d
bash scripts/reproduce.sh --suite rms3d
bash scripts/reproduce.sh --suite fusion3d
```

脚本准备固定版本源码和依赖，构建 Core、九齿 AOT 和主机桥接，然后运行原有算子数值比较与配对计时。三维和融合的 HIP 图计时直接使用本次新生成的 AOT 导出。

## 代码与结果

[`benchmarks`](benchmarks) 保存九齿表达式、原有数值参考和计时程序；[`development`](development) 保存海光 AOT 主机适配和 Core 桥接；[`scripts/reproduce.sh`](scripts/reproduce.sh) 是统一运行入口。原始 Core 源码和官方脚本保持不变，新增设备计算由九齿编译链生成。

每次运行创建新的 `results/hygon-operators-*`，按 `rms2d`、`rms3d`、`fusion3d` 保存原始数值比较、五轮计时、生成代码、原生库和日志。Core 算子列表重放和实际 HIP 图分别记录；它们的事件耗时不等于纯内核指令时间。

InfiniCore、NineToothed 和构建依赖的固定版本见 [`scripts/prepare.sh`](scripts/prepare.sh)，Python 包版本见 [`requirements.txt`](requirements.txt)。

本仓库复现范围是单卡算子与逐元素子图。主办方 H3 权重、样例及固定配置尚未取得，完整 FL2VA、生成质量、多卡、和模型端到端收益尚未验收。
