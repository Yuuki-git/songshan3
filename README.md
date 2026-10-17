# MiniMax H3 算子优化复现

参照二维 RMSNorm 的最小交付形式，统一复现三项工作：二维 128 维 RMSNorm、三维 Q/K RMSNorm，以及三维 AdaLN/门控残差融合。仓库只保留实现、必要的主机适配和下载构建运行脚本，实验结果在运行时生成。

## 环境

Linux x86_64、海光 gfx906，Python 3.11.9、镜像自带海光 PyTorch 2.9.0、HIP 6.3.26093，以及 Git、curl、make、CMake、g++、flock 和 timeout。DTK 位于 `/opt/dtk`，Torch 库位于 `/usr/local/lib/python3.11/site-packages/torch/lib` 和 `torch.libs`，沿用第一个算子的基础镜像布局。

额外 Python 包只安装到工程的 `venvs/h3`，原生库放在 `infini`。构建并发为 8，GPU 实验串行。工程可以放在自己的工作目录，不依赖团队 SSH 账号或已有实验目录。

## 运行

```bash
git clone --branch codex/unified-operator-reproduction --depth 1 https://github.com/Yuuki-git/songshan3.git songshan3
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

海光机器不联网时，先在联网机器上用 Bash 准备：

```bash
bash scripts/reproduce.sh --prepare-only
```

复制整个已准备工程到海光环境，保留源码 Git 和子模块元数据，然后执行：

```bash
bash scripts/reproduce.sh --offline
```

已有当前工程的独立环境时，使用 `bash scripts/reproduce.sh --skip-build` 重新生成 AOT、桥接和实验结果。该选项不重新构建基础 Core。

## 代码与结果

`benchmarks` 保存九齿表达式、原有数值参考和计时程序；`development` 保存海光 AOT 主机适配和 Core 桥接；`scripts` 保存最少运行入口。原始 Core 源码和官方脚本不修改，新增设备计算由九齿编译链生成。

每次运行创建新的 `results/hygon-operators-*`，下面按 `rms2d`、`rms3d`、`fusion3d` 分目录保存原始误差、五轮计时、生成代码、原生库和日志。顶层 `exit-code.txt` 是本次退出码。Core 算子列表重放和实际 HIP 图分别记录；它们的事件耗时不等于纯内核指令时间。

固定源码为 InfiniCore `f32e2fbb2a9b2ca912b5711f4f3836c403196a74`、NineToothed `2c5eb118592f4d2b0f448d16d2e9a3a5c0fbc900`、xmake-repo `bd29d77f242e247589457638272d9936a24a3fe0`；xmake 为 3.0.5、Boost 为 1.92.0、pybind11 为 3.1.0。Python 包版本见 `requirements.txt`。原 `rms128-20261016` 标签仍对应二维最小交付。

本仓库复现范围是单卡算子与逐元素子图。主办方 H3 权重、样例及固定配置尚未取得，完整 FL2VA、生成质量、多卡和模型端到端收益尚未验收。
