# JITServe 分阶段复现

复现环境使用 Python 3.11、仓库自带的修改版 vLLM（2024-10-27 snapshot）
和 PyTorch 2.5.0。
当前公开可下载且有配套 QRF 的组合是 `Qwen/Qwen2.5-7B-Instruct`
和 `qwen25_7b` 预测器。其架构为 `Qwen2ForCausalLM`，已注册在自带引擎中。
本地 Qwen3-8B 不能代替该模型。

Llama-3.1-8B-Instruct 需要 Hugging Face 授权；本次未取得授权，下载检查返回 401。
若后续改用 Llama，需同时设置 `--qrf-family llama3_8b` 并下载配套 QRF。

## 1. 网络与证书

从仓库根目录执行：

```bash
export JITSERVE_PROXY=http://127.0.0.1:7897
source scripts/repro_env.sh
```

该文件只修改当前 shell 的环境变量。系统 CA 会用于 Python、requests、pip、
curl 和 Git；本机请求通过 `NO_PROXY` 绕过代理。下载启用可继承这些证书配置的
requests 传输，不关闭 TLS 校验。

## 2. GPU 环境

必须在能使用 NVIDIA GPU、CUDA `nvcc`、gcc/g++ 和 Git 的环境中安装引擎。
`/proc/driver/nvidia/version` 存在并不代表容器可访问 GPU；实际还需设备映射。
当前检查容器没有 `/dev/nvidia*`、CUDA 编译器及 Docker socket，所以尚不能构建
和验证 GPU 引擎。先在宿主机配置 GPU 容器，或提供可访问 GPU 的执行环境。

可以在宿主机以 CUDA development 镜像创建一个挂载当前代码和模型目录的容器，
例如从仓库根目录运行以下命令（要求宿主机已有 Docker NVIDIA runtime）：

```bash
docker run --rm -it --gpus all --ipc=host --network=host \
  -v "$PWD":/workspace/JITServe \
  -v "$PWD/../models":/workspace/models \
  -w /workspace/JITServe \
  pytorch/pytorch:2.5.0-cuda12.4-cudnn9-devel bash
```

在此容器内确认 gcc/g++、Git 和 nvcc 均存在；缺失时安装 `build-essential` 和
`git`。然后创建单独的 GPU 虚拟环境：

```bash
export JITSERVE_PROXY=http://127.0.0.1:7897
export JITSERVE_VENV=/workspace/JITServe/.venv-gpu
bash scripts/setup_gpu.sh
```

使用其他 Python 入口时设置 `JITSERVE_BOOTSTRAP_PYTHON`。脚本会按 nvcc 版本
选择 PyTorch CUDA 11.8、12.1 或 12.4 wheel，然后从 `third_party/vllm` 源码构建。
它不会从 PyPI 安装官方 vLLM。完整 GPU 构建尚未在当前容器验证。
后续命令以 `.venv` 为例；使用上述 GPU 环境时请改为 `.venv-gpu`。

## 3. 下载资源

```bash
.venv/bin/python scripts/reproduce.py download --assets-only
.venv/bin/python scripts/reproduce.py download
```

第一条只下载所选 QRF 模型和 vectorizer；第二条同时下载模型权重。
QRF 固定版本为 `1e587124a9671f54f4c659f97bccbbe890b3d8af`。
模型解析为具体 commit 后再下载，成功后将版本写入
`reproduction_results/downloads.json`。下载可再次运行以复用缓存。

QRF 保存在 `assets/qrf/`，LLM 保存在 `../models/Qwen2.5-7B-Instruct/`。
Qwen QRF 约 1.4 GB，LLM 权重约 15 GB。

## 4. 先验证 QRF 和基础推理

QRF 可以单独在 CPU 上运行：

```bash
.venv/bin/python scripts/reproduce.py qrf
```

预测器监听 `localhost:65433`，使用带长度头的 TCP 消息；主服务和预测器需使用
同一份代码。本次已在独立环境中验证 scikit-learn 1.5.1、quantile-forest 1.4.2、
numpy 1.26.4、scipy 1.17.1 和 joblib 1.6.0 可以加载作者资源并执行预测。

进入 GPU 环境后执行：

```bash
.venv/bin/python scripts/reproduce.py check
.venv/bin/python scripts/reproduce.py smoke \
  --policies fcfs --model ../models/Qwen2.5-7B-Instruct
.venv/bin/python scripts/reproduce.py smoke \
  --policies jitserve --model ../models/Qwen2.5-7B-Instruct
```

运行器自动管理预测器和引擎的启动、日志、健康检查及退出清理。
默认使用 `127.0.0.1:8100`；端口被占用会退出，不终止已有服务。
FCFS 不加载 QRF，并关闭图匹配；JITServe 使用对应 QRF 和 online 图匹配。
单条请求必须产生非空的新文本才算通过。

上下文默认 8192，batch size 为 4，显存比例为 0.85。根据实际 GPU 调整
`--max-model-len`、`--batch-size` 和 `--gpu-memory`；配置会保存到命令记录。

## 5. 小规模混合 benchmark

```bash
.venv/bin/python scripts/reproduce.py benchmark \
  --model ../models/Qwen2.5-7B-Instruct \
  --num-prompts 6 --arrival-rate 0.5
```

依次测试 FCFS 和 JITServe。使用确定的三类请求混合 trace，默认每类两条任务，
集合任务会展开为多次请求，因此总请求数可能超过任务数。普通请求生成上限为
128 tokens；ToT 子请求沿用原客户端的 256 tokens 上限。
这一步验证运行链路，不用于宣称已复现论文性能。

输出在 `reproduction_results/`：

- `downloads.json`：下载版本。
- `qrf-validation.json`：实际 QRF 验证记录。
- `mixed-trace.json`：两种策略共用的输入。
- `*-command.json`、`*-server.log`：引擎命令和日志。
- `*-smoke.json`：单条生成响应。
- `*-metrics.json`、`*-benchmark.log`：benchmark 指标和日志。

小规模验证通过后，再按论文设置扩大请求量、到达率、batch size，并运行独立
DeepResearch benchmark。旧 `scripts/*benchmark*.sh` 保留历史实验配置，部分仍含
`concord` 等旧参数；本次复现使用上述新入口。

## CPU 回归检查

```bash
.venv/bin/python -m pytest -q tests
```

测试覆盖本机代理绕过、端口占用、启动失败、超时清理、混合输入以及 TCP 长消息。
它们不代替 GPU 推理和调度验证。
