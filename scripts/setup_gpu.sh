#!/usr/bin/env bash
set -euo pipefail
repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
source "$repo_root/scripts/repro_env.sh"
bootstrap_python="${JITSERVE_BOOTSTRAP_PYTHON:-python3}"
venv_path="${JITSERVE_VENV:-$repo_root/.venv}"

for tool in gcc g++ nvcc git; do
    if ! command -v "$tool" >/dev/null 2>&1; then
        echo "Missing $tool. Run in a CUDA development environment; see REPRODUCE.md." >&2
        exit 1
    fi
done

if [[ ! -x "$venv_path/bin/python" ]]; then
    "$bootstrap_python" -m venv "$venv_path"
fi
python_bin="$venv_path/bin/python"
"$python_bin" -c 'import sys; assert sys.version_info[:2] == (3, 11), "The pinned reproduction environment requires Python 3.11"'
cuda_flavor="$("$python_bin" -c 'import re, subprocess
version = re.search(r"release (\d+\.\d+)", subprocess.check_output(["nvcc", "--version"], text=True)).group(1)
supported = {"11.8": "cu118", "12.1": "cu121", "12.4": "cu124"}
if version not in supported:
    raise SystemExit("Use CUDA toolkit 11.8, 12.1 or 12.4 for the vendored PyTorch 2.5.0 engine; found " + version)
print(supported[version])')"
"$python_bin" -m pip install 'pip<25' wheel 'setuptools>=61,<76' 'setuptools-scm>=8,<9'
"$python_bin" -m pip install torch==2.5.0 torchvision==0.20.0 \
    --index-url "https://download.pytorch.org/whl/$cuda_flavor"
"$python_bin" -c 'import torch; assert torch.cuda.is_available(), "No CUDA GPU visible"; print(torch.cuda.get_device_name(0))'
"$python_bin" -m pip install -r "$repo_root/third_party/vllm/requirements-build.txt" \
    -r "$repo_root/requirements-reproduce.txt"
"$python_bin" -m pip install -e "$repo_root"
# Use the installed Torch and build tools instead of downloading them again
# into an isolated temporary build environment.
export MAX_JOBS="${MAX_JOBS:-4}"
export NVCC_THREADS="${NVCC_THREADS:-2}"
export SETUPTOOLS_SCM_PRETEND_VERSION_FOR_VLLM=0.6.3
"$python_bin" -m pip install --no-build-isolation \
    -c "$repo_root/requirements-reproduce.txt" -e "$repo_root/third_party/vllm"
"$python_bin" -m pip check
"$python_bin" "$repo_root/scripts/reproduce.py" check
