#!/usr/bin/env bash
# Installation entrypoint for the GPU-enabled Debian development container.
set -euo pipefail
repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
export JITSERVE_PROXY="${JITSERVE_PROXY:-http://127.0.0.1:7897}"
source "$repo_root/scripts/repro_env.sh"
export DEBIAN_FRONTEND=noninteractive
export PIP_DEFAULT_TIMEOUT=180
export PIP_RETRIES=10
export CUDA_HOME="$repo_root/.deps/cuda-12.4"
export PATH="$CUDA_HOME/bin:$repo_root/.venv-gpu/bin:$PATH"
export TORCH_CUDA_ARCH_LIST=8.9
export MAX_JOBS="${MAX_JOBS:-4}"
export NVCC_THREADS="${NVCC_THREADS:-2}"
export SETUPTOOLS_SCM_PRETEND_VERSION=0.6.3
export FETCHCONTENT_BASE_DIR="$repo_root/.deps/cmake"
mkdir -p "$repo_root/reproduction_results"
action="${1:-check}"
exec > >(tee -a "$repo_root/reproduction_results/install-${action}.log") 2>&1
case "$action" in
    install)
        for stage in cuda python build check test record smoke; do
            echo "Starting JITServe installation stage: $stage"
            bash "$repo_root/scripts/install_gpu_container.sh" "$stage"
        done
        echo "JITServe dependencies and smoke checks completed"
        ;;
    system)
        apt-get -o Acquire::Retries=5 -o Acquire::http::Proxy="$JITSERVE_PROXY" update
        apt-get -o Acquire::Retries=5 -o Acquire::http::Proxy="$JITSERVE_PROXY" install -y \
            --no-install-recommends build-essential git curl ca-certificates pkg-config libnuma-dev
        ;;
    cuda)
        /workspace/conda-env/bin/python "$repo_root/scripts/install_cuda_redist.py"
        "$CUDA_HOME/bin/nvcc" --version
        ;;
    python)
        if [[ ! -x "$repo_root/.venv-gpu/bin/python" ]]; then
            /workspace/conda-env/bin/python -m venv "$repo_root/.venv-gpu"
        fi
        python_bin="$repo_root/.venv-gpu/bin/python"
        "$python_bin" -m pip install 'pip==25.3' wheel 'setuptools>=61,<76' 'setuptools-scm>=8,<9'
        "$python_bin" -m pip install torch==2.5.0 torchvision==0.20.0 \
            --index-url https://download.pytorch.org/whl/cu124
        "$python_bin" -m pip install -r "$repo_root/third_party/vllm/requirements-build.txt" \
            -r "$repo_root/requirements-reproduce.txt"
        "$python_bin" -m pip install -e "$repo_root"
        ;;
    build)
        cd "$repo_root/third_party/vllm"
        python_bin="$repo_root/.venv-gpu/bin/python"
        "$python_bin" -m pip install -c "$repo_root/requirements-reproduce.txt" \
            -r requirements-cuda.txt
        mkdir -p vllm/vllm_flash_attn
        # Keep object files outside pip's disposable editable-build directory.
        "$python_bin" setup.py build_ext --inplace \
            --build-temp "$repo_root/.deps/vllm-build" \
            --build-lib "$repo_root/.deps/vllm-build-lib"
        for extension in vllm/_C.abi3.so vllm/_moe_C.abi3.so \
            vllm/vllm_flash_attn/vllm_flash_attn_c.abi3.so; do
            [[ -s "$extension" ]] || { echo "Missing locally compiled $extension" >&2; exit 1; }
        done
        # Reuse only the extensions just compiled from this checkout.
        VLLM_USE_PRECOMPILED=1 "$python_bin" -m pip install --no-build-isolation \
            -c "$repo_root/requirements-reproduce.txt" -e . -v
        ;;
    check)
        nvidia-smi
        "$repo_root/.venv-gpu/bin/python" -m pip check
        "$repo_root/.venv-gpu/bin/python" "$repo_root/scripts/reproduce.py" check
        "$repo_root/.venv-gpu/bin/python" -c 'import torch; x=torch.arange(1024,device="cuda"); assert x.sum().item()==523776; print("CUDA tensor operation passed")'
        ;;
    test)
        cd "$repo_root"
        "$repo_root/.venv-gpu/bin/python" -m pytest -q tests
        ;;
    record)
        "$repo_root/.venv-gpu/bin/python" -m pip freeze \
            > "$repo_root/reproduction_results/installed-packages.txt"
        nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv \
            > "$repo_root/reproduction_results/gpu-info.csv"
        "$CUDA_HOME/bin/nvcc" --version \
            > "$repo_root/reproduction_results/cuda-version.txt"
        echo "Environment versions saved in $repo_root/reproduction_results"
        ;;
    status)
        nvidia-smi
        for stage in system cuda python build check smoke serve; do
            logfile="$repo_root/reproduction_results/install-${stage}.log"
            if [[ -f "$logfile" ]]; then
                echo "Latest output for $stage:"
                tail -n 12 "$logfile"
            fi
        done
        for logfile in "$repo_root"/reproduction_results/*-server.log; do
            if [[ -f "$logfile" ]]; then
                echo "Latest output for $logfile:"
                tail -n 30 "$logfile"
            fi
        done
        ;;
    smoke)
        "$repo_root/.venv-gpu/bin/python" "$repo_root/scripts/reproduce.py" smoke \
            --model /workspace/models/Qwen2.5-7B-Instruct --max-model-len 4096 \
            --batch-size 2 --gpu-memory 0.85
        ;;
    serve)
        "$repo_root/.venv-gpu/bin/python" "$repo_root/scripts/reproduce.py" serve \
            --policies jitserve --model /workspace/models/Qwen2.5-7B-Instruct \
            --max-model-len 4096 --batch-size 2 --gpu-memory 0.85
        ;;
    *) echo "Expected action: install, system, cuda, python, build, check, test, record, status, smoke, serve" >&2; exit 2 ;;
esac
