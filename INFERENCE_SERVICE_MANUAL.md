# JITServe 推理服务操作手册

更新日期：2026-11-06。本文用于当前已安装完成的 GPU 容器环境；安装、下载与编译过程见 [REPRODUCE.md](REPRODUCE.md)。

## 1. 当前配置与使用范围

以下命令在当前容器内执行。宿主机上的 `/home/dell/xwd/JITServe` 与容器内路径不同，不能直接混用 Python 环境路径。

| 项目 | 当前配置 |
| --- | --- |
| 项目目录 | `/workspace/JITServe` |
| Python 环境 | `/workspace/JITServe/.venv-gpu`，Python 3.11 |
| GPU | NVIDIA RTX 4090，24 GiB |
| 推理引擎 | 仓库自带修改版 vLLM，已从源码编译；PyTorch 2.5.0+cu124 |
| CUDA 工具链 | `/workspace/JITServe/.deps/cuda-12.4` |
| 大模型 | `/workspace/models/Qwen2.5-7B-Instruct` |
| 调度策略 | JITServe，启用 QRF、online 图匹配 |
| 运行参数 | 最大上下文 4096 tokens，最大并发序列数 2，显存比例 0.85 |
| HTTP 服务 | `http://127.0.0.1:8100` |
| QRF 内部服务 | `localhost:65433`，CPU 上执行预测 |

此前服务已通过 `/health` 和真实生成验证，随后已按要求停止。后续是否运行应通过健康检查确认。当前入口在本机回环地址监听；从其他机器访问需要另行配置网络入口。论文性能指标尚未复现。

## 2. 检查并启动

进入项目目录：

```bash
cd /workspace/JITServe
```

先检查现有服务：

```bash
curl --noproxy '*' --fail --silent --show-error --max-time 10 \
  -w '\nHTTP %{http_code}\n' http://127.0.0.1:8100/health
```

返回 `HTTP 200` 表示 HTTP 服务可用；该接口正文为空是正常现象。已有服务时直接调用，避免重复启动。连接失败时，先确认不是正在加载模型：

```bash
pgrep -af '[s]cripts/reproduce.py serve'
tail -n 40 reproduction_results/install-serve.log
```

没有正在启动或运行的实例时，在前台执行：

```bash
bash scripts/install_gpu_container.sh serve
```

入口自动配置本地 CUDA 和 `.venv-gpu` 环境，依次启动 QRF、推理引擎，检查健康状态并发送一条真实生成请求。等待以下提示再使用：

```text
jitserve: smoke request passed
JITServe is ready at http://127.0.0.1:8100; press Ctrl-C to stop
```

模型加载与 CUDA 图初始化通常需要数分钟，以就绪提示为准。日常启动无需重新执行 `install`、编译或下载；模型及 QRF 文件已经保存在本地。入口会检查端口占用，发现已有监听者时退出，不会替换已有服务。

需要关闭终端后继续运行时，可改用后台方式（与前台方式二选一）：

```bash
nohup bash scripts/install_gpu_container.sh serve \
  > reproduction_results/serve-console.log 2>&1 &
```

随后用健康检查和日志确认就绪。后台启动的 `$!` 是 shell 包装进程，停止时应按第 5 节定位 Python 运行器。

## 3. 调用生成接口

当前提供的是 JITServe 自定义 `POST /generate` 接口。复制以下命令即可发送一条非流式请求；本机请求显式绕过代理：

```bash
curl --noproxy '*' --fail --show-error --max-time 180 \
  http://127.0.0.1:8100/generate \
  -H 'Content-Type: application/json' \
  --data-binary @- <<'JSON'
{
  "client_id": 0,
  "stream": false,
  "slo_constraint": [60, 10, 120],
  "sampling_params": {
    "max_tokens": 32,
    "temperature": 0.0
  },
  "request_info": {
    "prompt": "Explain why the sky is blue in one sentence.",
    "request_type": 0,
    "collection_id": 0,
    "deadline": 120,
    "input_len": 0,
    "output_len": 32
  }
}
JSON
```

这是当前已通过验证的基础请求。替换 `request_info.prompt` 即可修改问题；它接收原始 prompt，没有在此示例中自动套用 Qwen 聊天模板。需要更长回答时增加 `sampling_params.max_tokens`，并保证输入与输出总长度不超过当前 4096 tokens 的上下文限制。

主要参数：

- `sampling_params.max_tokens`：实际新生成 token 数量的上限；示例的 32 tokens 可能使句子中途结束。
- `request_info.output_len`：供调度使用的长度信息，启用 QRF 后会被预测逻辑更新；生成上限由 `max_tokens` 控制。
- `request_type`：`0` 为延迟敏感，`1` 为吞吐敏感，`2` 为集合任务；基础调用使用 `0`。
- `slo_constraint`：依次为 TTFT（首 token 时间）、TBT（token 间隔）、TTLT（总延迟）的调度约束。示例使用宽松值验证服务，不代表性能测量结果。
- `deadline`、`collection_id` 等为调度协议字段，普通调用可沿用示例；集合任务需要额外的任务组织配置。

响应为 JSON，包含 `text`、`time`、`tbt`、`service_gain` 和 `finish_reason`。其中：

- `text` 是字符串列表，每项包含“原始 prompt + 新生成文本”。展示纯回答时应去掉原始 prompt 前缀。
- 当前非流式接口的 `time` 对应 TTFT，不是完整请求耗时；`tbt` 为 token 间隔记录。
- `finish_reason` 为 `length` 时表示达到长度限制。

当前接口不能直接使用 OpenAI `/v1/chat/completions` 的请求格式。若开启 `stream: true`，返回的是以空字节 `\0` 分隔的 JSON 片段，客户端需要按此协议解析。

## 4. 查看状态与日志

```bash
nvidia-smi
bash scripts/install_gpu_container.sh status
tail -n 60 reproduction_results/jitserve-server.log
tail -n 60 reproduction_results/qrf.log
```

`status` 汇总 GPU 信息和历史日志；实时服务可用性仍以健康检查和真实请求为准。

| 文件（相对于项目根目录） | 内容 |
| --- | --- |
| `reproduction_results/install-serve.log` | 启动入口输出，追加保存 |
| `reproduction_results/jitserve-server.log` | 推理服务日志，下次启动覆盖 |
| `reproduction_results/qrf.log` | 预测器日志，下次启动覆盖 |
| `reproduction_results/jitserve-command.json` | 实际推理服务命令 |
| `reproduction_results/jitserve-smoke.json` | 启动时真实请求的验证响应 |

需要保留故障现场时，重启前先复制日志。

## 5. 停止与重启

### 前台启动

在启动服务的终端按 **Ctrl-C**。Python 运行器会清理推理服务及 QRF 子进程组，然后退出。

### 后台启动，或原终端不可用

先查找并核对完整命令：

```bash
pgrep -af '[s]cripts/reproduce.py serve'
```

找到命令包含当前项目 `scripts/reproduce.py serve` 的 **Python 运行器 PID**，将下面的 `PID` 替换为查到的数字后执行：

```bash
kill -TERM PID
```

运行器收到 SIGTERM 后执行与 Ctrl-C 相同的清理。请定位运行器，而不是仅终止外层 `bash ...install_gpu_container.sh serve`；也应避免使用宽泛的批量杀进程命令。

清理可能持续数十秒。结束后检查：

```bash
pgrep -af '[s]cripts/reproduce.py serve|[j]itserve.server|[j]itserve.request_analyzer.prediction'
curl --noproxy '*' --silent --show-error --max-time 5 \
  http://127.0.0.1:8100/health
nvidia-smi
```

预期当前项目相关进程消失、健康检查连接失败，GPU 中该引擎的占用释放；其他程序可能仍占用显存。确认清理完毕后，再执行第 2 节启动命令即可重启。

## 6. 常见情况

- **8100 或 65433 端口占用**：检查是否已有当前服务或单独启动的 QRF。先确认进程用途，再停止对应实例。
- **启动期间健康检查失败**：查看 `jitserve-server.log` 与 `qrf.log`，等待运行器给出就绪或明确失败提示。
- **CUDA 不可用**：先运行 `nvidia-smi`；容器重建后需保留 GPU 设备权限，以及代码、模型和环境目录。
- **显存不足**：检查其他 GPU 程序；当前参数针对 24 GiB RTX 4090。更改上下文、并发或模型时需要重新评估显存。
- **QRF 文件缺失或加载失败**：检查第 7 节的两个文件及 `qrf.log`。两个文件必须来自配套版本。
- **想重新运行完整 smoke 检查**：先停止常驻服务，再执行 `bash scripts/install_gpu_container.sh smoke`。它依次启动 FCFS、JITServe，测试后自动退出。benchmark 也应与当前实例错开，避免端口和 GPU 资源冲突。

## 7. 为什么自带 QRF 还要下载文件

QRF 是 JITServe 自带的组件。仓库中的 `jitserve/request_analyzer/prediction.py` 提供加载、预测和内部服务代码；训练完成的预测模型及文本向量器由作者单独发布在 Hugging Face 的 [En-2863/jitserve-qrf-length-predictor](https://huggingface.co/En-2863/jitserve-qrf-length-predictor)，需要下载后才能实际执行预测。

当前下载的是作者的 JITServe 复现资源，而不是另加一个预测器，也不是 Qwen 官方模型权重的一部分。当前使用的文件为：

```text
/workspace/JITServe/assets/qrf/qrf_model/0_qrf_lmsys_chat_qwen25_7b.pkl
/workspace/JITServe/assets/qrf/qrf_vectorizer/0_qrf_lmsys_chat_qwen25_7b.pkl
```

这两个文件合计约 1.4 GB，固定版本为 `1e587124a9671f54f4c659f97bccbbe890b3d8af`。第一份是训练好的 QRF，第二份是训练时配套的文本特征向量器。旧代码把后者命名为 `tokenizer`，但它不是 Qwen 大模型的 tokenizer。

QRF 根据 prompt 在 CPU 上估计输出长度的 0.95 分位数，为调度提供较保守的长度信息；GPU 上的 Qwen 才负责生成回答。0.95 分位数不是对每条请求长度的硬保证。

作者为 Qwen2.5-7B 和 Llama-3.1-8B 分别提供训练资源，因为不同大模型的输出长度分布不同。应使用对应模型的 QRF，并将 QRF 与它自己的向量器成对加载。用错模型家族的预测器可能仍能运行，但预测偏差会影响调度；混搭向量器还可能造成特征维度错误。

当前启动入口已明确选择 Qwen 配套资源；底层模块的历史默认路径仍指向 Llama，因此建议使用本文的统一入口。日常运行直接加载已有文件，不会重新训练或重新下载 QRF。作者说明见 [assets/qrf/README.md](assets/qrf/README.md)。

## 8. Benchmark 实验操作

### 8.1 实验范围与准备

Benchmark 由客户端程序自动按到达间隔发送请求，多条请求可以同时在途，并统计吞吐量、延迟和 SLO 指标。当前 `scripts/reproduce.py benchmark` 入口支持 `fcfs`、`jitserve` 两种策略，默认依次运行；每种策略都会重新启动引擎并在测试后清理，JITServe 模式还会自动管理 QRF。

先按第 5 节停止常驻服务，确认 8100 和 65433 端口没有其他实例监听、GPU 显存足够，再运行实验。无需事先执行 `serve`，也无需另外手动启动 QRF。

在容器内准备环境和本轮独立目录：

```bash
cd /workspace/JITServe
export JITSERVE_PROXY=http://127.0.0.1:7897
source scripts/repro_env.sh
export CUDA_HOME=/workspace/JITServe/.deps/cuda-12.4
export PATH="$CUDA_HOME/bin:/workspace/JITServe/.venv-gpu/bin:$PATH"

benchmark_dir="$PWD/reproduction_results/benchmark-$(date -u +%Y%m%dT%H%M%SZ)"
mkdir -p "$benchmark_dir"
echo "$benchmark_dir"
```

记下输出目录。以下前台、后台方式二选一；每次新实验重新生成目录，避免覆盖。

### 8.2 前台运行小规模实验

```bash
.venv-gpu/bin/python scripts/reproduce.py benchmark \
  --model /workspace/models/Qwen2.5-7B-Instruct \
  --qrf-family qwen25_7b \
  --policies fcfs,jitserve \
  --max-model-len 4096 \
  --batch-size 2 \
  --gpu-memory 0.85 \
  --num-prompts 6 \
  --arrival-rate 0.5 \
  --startup-timeout 600 \
  --benchmark-timeout 1800 \
  --output "$benchmark_dir"
```

这是用于先跑通基准测试流程的小实验。首次 benchmark 尚未完成性能验证；此前通过的是两种策略的单条 smoke 推理测试。正常结束时会分别打印 `fcfs: benchmark completed ...`、`jitserve: benchmark completed ...`，并返回命令提示符。运行器只要求至少完成一条请求才报告完成，仍需检查完成数和错误，不能将该提示等同于全部请求成功。

参数说明：

| 参数 | 含义 |
| --- | --- |
| `--num-prompts 6` | 每种策略使用 6 个顶层任务；当前入口轮流选取三类任务，集合任务展开后实际 HTTP 请求数会更多 |
| `--arrival-rate 0.5` | 目标顶层任务到达率约每秒 0.5 个，即平均约 2 秒一个；不等待上一条完成才发送下一条 |
| `--batch-size 2` | 设置引擎最大并发序列数，不是客户端一次只发送两条请求 |
| `--policies fcfs,jitserve` | 用同一份输入依次比较两种策略；也可只写 `jitserve` 或 `fcfs` |
| `--startup-timeout 600` | 单次启动就绪等待超时，单位秒 |
| `--benchmark-timeout 1800` | 每种策略的客户端测试超时，单位秒，不含引擎启动时间 |
| `--output` | 本轮输入、日志和结果的保存目录 |

需要扩大测试时，可以先将 `--num-prompts` 改为 `60`，保持其他设置相同，再逐步调整到达率。到达率越高越容易排队；GPU 性能不足时不能直接照搬论文的请求速率。总耗时包含两次模型加载和请求排空时间，不只是“任务数 ÷ 到达率”。

当前入口将普通请求最大输出设为 128 tokens，ToT 子请求有自己的长度限制；SLO 基值为 TTFT 2 秒、TBT 0.1 秒、总延迟 20 秒，客户端还按任务编号进行 1～4 倍缩放。这些值目前写在代码中，调整它们需要修改并记录实验配置。

### 8.3 后台运行

先执行第 8.1 节的环境与目录准备，再执行：

```bash
nohup .venv-gpu/bin/python scripts/reproduce.py benchmark \
  --model /workspace/models/Qwen2.5-7B-Instruct \
  --qrf-family qwen25_7b \
  --policies fcfs,jitserve \
  --max-model-len 4096 \
  --batch-size 2 \
  --gpu-memory 0.85 \
  --num-prompts 6 \
  --arrival-rate 0.5 \
  --startup-timeout 600 \
  --benchmark-timeout 1800 \
  --output "$benchmark_dir" \
  > "$benchmark_dir/runner.log" 2>&1 &
printf '%s\n' "$!" > "$benchmark_dir/runner.pid"
```

这里直接后台启动 Python 运行器，因此 `runner.pid` 记录的是它的 PID。查看进度：

```bash
tail -n 40 "$benchmark_dir/runner.log"
tail -n 40 "$benchmark_dir/fcfs-benchmark.log"
tail -n 40 "$benchmark_dir/jitserve-benchmark.log"
```

尚未执行到某个阶段时，对应日志可能还不存在。进入新终端后，先将 `benchmark_dir` 设置为本轮目录的完整路径，再使用上述命令。

### 8.4 查看实验结果

本轮目录中的主要文件：

| 文件 | 内容 |
| --- | --- |
| `mixed-trace.json` | 两种策略共用的顶层任务输入 |
| `fcfs-metrics.json`、`jitserve-metrics.json` | 本轮客户端测试完成后写入的统计结果 |
| `fcfs-benchmark.log`、`jitserve-benchmark.log` | 客户端进度、统计表与异常 |
| `fcfs-server.log`、`jitserve-server.log` | 推理引擎日志 |
| `qrf.log` | JITServe 阶段的 QRF 日志 |
| `*-command.json` | 引擎实际启动命令 |
| `*-smoke.json` | 各阶段启动后的预检查响应，不是整场 benchmark 结果 |
| `runner.log`、`runner.pid` | 上述后台方式的总日志及运行器 PID |

可以直接格式化查看结果：

```bash
.venv-gpu/bin/python -m json.tool "$benchmark_dir/fcfs-metrics.json"
.venv-gpu/bin/python -m json.tool "$benchmark_dir/jitserve-metrics.json"
```

结果包含 `settings`（配置）、`requests`（请求统计）和 `tasks`（集合任务统计）。重点查看 `requests` 中的：

- `request_num`、`request_completed`：统计到的请求数与成功完成数，并结合客户端错误日志检查失败；集合任务展开后的数量与顶层任务数不同。
- `request_throughput`、`request_goodput`：完成请求吞吐与当前实现计算的 SLO 请求 goodput。
- `mean_ttft_ms`、`mean_tbt_ms`、`mean_e2el_ms`：平均首 token、token 间隔、端到端延迟，单位毫秒；对应 `percentiles_*` 字段包含分位数。
- `time_window_token_goodput`：按 60 秒窗口保存的 token goodput；当前代码给 decode token 乘以 8，窗口按请求完成时间归属，不能未经口径核对直接作为论文图 11 的数值。

按类型保存的四项列表依次对应“延迟敏感、吞吐／截止时间敏感、集合任务、总计”。少量请求不适合评估长时间曲线或尾延迟。当前统计实现的 SLO 判定与失败计数还需要在正式论文实验前审核。

### 8.5 结束或中止实验

正常完成后运行器会自动停止各策略的引擎及预测器，无需另行停止服务。中途中止：

- 前台方式：在运行终端按 **Ctrl-C**。
- 后台方式：先定位并核对当前实验的 Python 运行器，再终止它：

```bash
pgrep -af '[s]cripts/reproduce.py benchmark'
```

将下方 `PID` 替换为已核对命令和输出目录的运行器 PID：

```bash
kill -TERM PID
```

等待清理后，用 `nvidia-smi` 和进程检查确认退出。未完成阶段可能没有 `*-metrics.json`，已写入的日志和其他阶段的结果会保留。

### 8.6 与论文图 11 的关系

以上是当前 GPU 环境下的小规模混合负载实验入口，不等同于图 11 的完整复现。它只比较两种策略，使用当前 Qwen2.5-7B 和简化工作负载，不会因为增大任务数就自动得到论文的一小时实验。

论文图 11 涉及四个模型、五种策略、一小时在线负载、每轮超过一万条请求和五次独立重复；严格复现还需要对齐到达 trace、应用工作负载、SLO、预测器、硬件配置与 goodput 统计口径。当前入口也没有直接接入单独的 DeepResearch 客户端。正式实验应补齐这些设置，并为每轮保留独立配置与结果。
