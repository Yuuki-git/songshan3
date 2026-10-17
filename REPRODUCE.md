# GitHub 统一复现说明

从一个 Git 提交复现二维 RMSNorm、三维 Q/K RMSNorm 和三维 AdaLN/门控残差融合。所有数值实现由九齿生成，主机代码只负责 AOT、原生 ABI、Core 接入和资源管理。官方 Core/LM 源码和评测脚本保留原样。

## 环境与输入

| 项目 | 已验证的基线环境 |
| --- | --- |
| 系统及设备 | Linux x86_64，海光 gfx906，单卡，wavefront 64 |
| 基础 Python | 3.11.9，默认 `/usr/local/bin/python` |
| PyTorch | 基础镜像海光 2.9.0，保持原包 |
| HIP / DTK | Torch HIP 6.3.26093，`/opt/dtk/env.sh` 与 `/opt/dtk/lib` |
| Torch 原生库 | `/usr/local/lib/python3.11/site-packages/torch/lib` 和 `torch.libs` |
| 构建 | g++ 11.4，make、CMake；工程内 xmake 3.0.5，并发 8 |
| 额外 Python 包 | 工程内 `venvs/h3`，Triton 3.8.0，准确版本及 SHA256 见锁文件 |
| Core 构建模式 | HYGON gfx906、CCL/ATen 绑定开启、graph=false |

Torch/HIP 报告的设备名为 Z200SM_80，hy-smi 报告 Z100SM，两种接口的差异未自行归因。以上是实际测试环境，不承诺未经测试的设备或工具链也能通过。

工程名称与父目录均可变化；`setup_env.sh` 根据自身位置确定工程根。基础解释器位置可通过 `SONGSHAN3_BASE_PYTHON` 指定，其他 DTK/Torch 库布局仍按上表准备。安装到工程 `infini` 前缀，固定上游 xmake 的 C API 安装规则还会写入 `/usr/local`，运行账户需具备对应权限。

| 官方源码 | 固定提交 |
| --- | --- |
| InfiniTensor/InfiniCore | `f32e2fbb2a9b2ca912b5711f4f3836c403196a74` |
| InfiniTensor/InfiniLM | `65deb91f840ebaef752d028fad2f3217ce0950f4` |
| InfiniTensor/ninetoothed | `2c5eb118592f4d2b0f448d16d2e9a3a5c0fbc900` |
| xmake-io/xmake-repo | `bd29d77f242e247589457638272d9936a24a3fe0` |

InfiniLM 仅固定 H3 模型源码依据，本流程不构建 LM 或下载模型权重。完整来源、子模块提交、长度和校验值见 `manifests/upstreams.json`、`manifests/workstation-wheels.json`。原始官方源码放在 `Projects`，九齿主机适配放在独立的 `development/ninetoothed-hygon`；融合实验再创建独立副本以显式关闭浮点收缩。

## 获取并执行

```bash
git clone https://github.com/Yuuki-git/songshan3.git songshan3
cd songshan3
git rev-parse HEAD
bash scripts/reproduce.sh --suite all
```

使用尚未合并的分支时，克隆命令需加 `--branch 分支名`。交付以准确 Git 提交为准，保存输出中的提交和代码哈希；原二维标签 `rms128-20261016` 不重指向新版本。

流程依次完成输入核验、独立 venv/Core 构建、隔离九齿后端、二维主验收、三维主验收和补充检查、融合主验收和补充检查，最后核验环境及重算统计。GPU 阶段串行，任何阶段非零退出即停止。

只复现一项时，选择 `--suite rms2d`、`--suite rms3d` 或 `--suite fusion3d`。每次仍创建新结果目录和编译缓存，不自动选取以前的实验。

## 先下载后离线

在有网络的机器上执行：

```bash
python scripts/reproduce.py --prepare-only
```

此阶段使用 Python 标准库和 Git，不导入 PyTorch。下载来自清单中的官方 GitHub/PyPI；wheel 的文件名、长度、SHA256 与官方 PyPI 元数据核对。只想核验已有输入时加 `--offline`，缺失或不符会失败。

复制整个已准备工程至海光机器，保留源码 Git/子模块元数据，再执行：

```bash
bash scripts/reproduce.sh --offline --suite all
```

已有当前目录的独立环境时可跳过基础构建：

```bash
bash scripts/reproduce.sh --offline --skip-build --suite all
```

`--skip-bootstrap` 为 `--skip-build` 的兼容别名。`scripts/prepare.sh --download-only` 继续支持仅准备输入；完整入口推荐统一使用 reproduce。`--prepare-only` 与 `--skip-build` 不可同时使用。

## 独立运行单个阶段

每个新 Bash 会话先：

```bash
source scripts/setup_env.sh
bash scripts/prepare_ninetoothed.sh
```

随后可运行对应主入口。三维补充入口必须显式指定刚刚成功的主实验目录，统一入口会自动完成这一步：

```bash
bash scripts/run_hygon_rmsnorm3d.sh
export SONGSHAN3_RMS3D_SOURCE="主验收刚输出的完整路径"
bash scripts/run_hygon_rmsnorm3d_extended.sh
```

融合入口同理：

```bash
bash scripts/run_hygon_fusion3d_core.sh
export SONGSHAN3_FUSION_SOURCE="主验收刚输出的完整路径"
bash scripts/run_hygon_fusion3d_extended.sh
```

每阶段使用 `cache/operator-experiment.lock`。同一块 GPU 上若从多个检出目录运行，令这些调用的 `SONGSHAN3_GPU_LOCK` 指向同一个锁文件。脚本不停止其他进程、会话或 Pod。

## 输出和复核

主目录 `results/reproduction-*` 中：

- `run.json` 固定本次代码哈希、阶段命令、状态和目录关系；`exit-code.txt` 为整体退出码。
- `summary.json` 只核验这个 run 引用的新产物；`paired-summary.csv` 保留实验阶段和计时方式，重新计算五轮中位数、倍率。
- `checksums.json` 覆盖本次阶段文件，原始代码、生成代码、编译缓存、AOT 导出、逐例误差和完整时间仍在各阶段目录中。
- 输入准备或构建失败时，保留 `inputs.log` 或 `build.log`；不生成成功验收汇总。

复核同一调用的结果：

```bash
python scripts/summarize_reproduction.py --run results/本次目录/run.json
```

RMSNorm 使用 CPU float64 参考并转为目标输出精度。FP16 的 atol=rtol=2e-3，BF16 为1e-2，FP32 为atol=1e-5、rtol=1e-4。融合算子要求逐位一致，atol=rtol=0。同步 API 两侧同入口、同输入、预分配输出，预热10次，每轮100次，交替顺序5轮。实际 HIP 图每图32次调用，时间除以32；Core graph=false 的列表重放另行记录。

## 验收范围

二维主验收包含75组四路径、40组冻结用例，其中20组走九齿128维实现；三维包含40组四路径、20项补充检查、独立C++及15组实际HIP图；融合包含28组五路径、28项补充检查、独立C++及18组配对。代码在本轮同一固定工具链下使用，未修改原始Core/LM或通过ATen替代新增数值实现。

完整H3入口、FL2VA、生成质量、峰值内存、多卡和端到端收益仍需独立验收。不能把静态CI、输入准备、环境预检或这些算子通过记为完整模型通过。具体本轮执行情况见 `docs/REPRODUCTION_VALIDATION.md`。
