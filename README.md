# MiniMax H3 单卡算子优化复现

使用九齿实现二维 RMSNorm、真实 Q/K 切片布局的三维 RMSNorm，以及三维 AdaLN/门控残差融合，经原生 AOT 接入 InfiniCore。一个入口完成固定输入校验、独立环境构建、串行验收和原始计时汇总。

当前范围为海光 gfx906 单卡算子和逐元素子图。主办方 H3 权重、样例与固定配置尚未取得，完整 FL2VA、生成质量、多卡及模型端到端收益尚未验收。

## 开始复现

在 Linux x86_64 海光环境中执行：

```bash
git clone https://github.com/Yuuki-git/songshan3.git songshan3
cd songshan3
bash scripts/reproduce.sh --suite all
```

如果使用开发分支，请在 `git clone` 时用 `--branch` 选择该分支。保存本次检出的 `git rev-parse HEAD`，后续按这个提交复现。原 `rms128-20261016` 标签保持为二维最小交付，不包含本次新增的三维和融合流程。

基础镜像保留海光 PyTorch 2.9.0、Python 3.11.9、HIP 6.3.26093 和 `/opt/dtk`。还需 Git、g++、make、CMake、curl、flock 和 timeout。脚本只在工程独立 venv 安装额外 Python 依赖，以并发 8 构建 Core；InfiniLM 源码用于固定模型布局依据，不在算子流程中构建或执行模型。安装位置与基线镜像的具体要求见 [完整复现说明](REPRODUCE.md)。

## 选择任务

| 参数 | 执行内容 |
| --- | --- |
| `--suite all` | 以下三项，依次执行，默认选项 |
| `--suite rms2d` | 二维 128 维 RMSNorm 四路径、冻结用例、Core 列表重放及配对计时 |
| `--suite rms3d` | 三维候选、四路径、独立 C++、回退/原地检查及同步 API/实际 HIP 图 |
| `--suite fusion3d` | AdaLN/门控残差五路径、独立 C++、守卫及单算子/四节点子图配对 |

```bash
bash scripts/reproduce.sh --suite rms3d
bash scripts/reproduce.sh --offline --skip-build --suite all
bash scripts/reproduce.sh --list
```

`--skip-build` 只复用当前检出目录已经准备好的独立 venv、原生库与输入；本次的桥接、AOT 导出、检查和计时仍重新生成。它不表示已验证全新基础镜像重建。

## 联网准备与离线执行

准备阶段只需 Python 3.10+ 和 Git，不加载 DTK、不安装包、不访问 GPU，也可在 Windows 联网机器运行：

```bash
python scripts/reproduce.py --prepare-only
```

把准备后的完整工程复制到海光环境，保留 `Projects`、`tooling/xmake-repo` 的 Git 元数据以及下载文件，然后执行：

```bash
bash scripts/reproduce.sh --offline --suite all
```

四个官方源码仓库、子模块、三个构建依赖和 17 个准确 wheel 按 [输入清单](manifests/upstreams.json)、[wheel 校验清单](manifests/workstation-wheels.json) 和 [带哈希的安装锁文件](requirements-lock.txt) 核验。已存在文件不符时停止，不重置原始源码或替换镜像 PyTorch。

## 查看输出

终端打印新的 `results/reproduction-时间戳-随机后缀`。其中 `run.json` 记录本次仓库提交、代码哈希、命令、退出码和各阶段的新结果目录，主实验成功后自动把路径传给其补充实验。失败即停止，并保留阶段日志。

| 文件 | 内容 |
| --- | --- |
| `run.json`、`exit-code.txt` | 本次调用的状态、配置和退出码 |
| `summary.json` | 正确性、分派、重放及环境不变量核验 |
| `paired-summary.csv` | 从本次五轮原始时间重算的中位数和倍率，各计时方式分列 |
| `checksums.json` | 本次实验产物的 SHA256 |
| 各阶段 `.log`、`.result-path` | 完整阶段日志及实际新建目录 |

每个实验另保存种子、形状、stride、容差、逐例误差、原始计时、生成代码、AOT 库和环境快照。同步 API、Core 算子列表重放、实际 HIP 图分别标注，事件耗时不等于纯内核指令时间。

[算子技术报告](docs/OPERATOR_REPORT.md)汇总已测量的优化结果；[本次交付验证记录](docs/REPRODUCTION_VALIDATION.md)说明统一入口实际完成的检查与限制。GitHub Actions 只做不需要 GPU 的静态与调度检查，不将 CI 通过记为海光或完整模型验收。
