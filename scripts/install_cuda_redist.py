"""Install NVIDIA's CUDA 12.4.1 redistributables into the project, without drivers."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import tarfile
import time
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / ".deps/cuda-12.4"
CACHE = ROOT / ".deps/cuda-downloads"
BASE = "https://developer.download.nvidia.com/compute/cuda/redist/"
COMPONENTS = ("cuda_nvcc", "cuda_cudart", "cuda_cccl", "cuda_nvrtc", "cuda_cupti",
              "cuda_nvtx", "cuda_profiler_api", "libcublas", "libcusparse", "libcusolver",
              "libcurand", "libcufft", "libnvjitlink")


def fetch(url, target, expected=None):
    temporary = target.with_suffix(target.suffix + ".partial")
    if target.is_file():
        if expected is None:
            return
        with target.open("rb") as stream:
            if hashlib.file_digest(stream, "sha256").hexdigest() == expected:
                return
        raise RuntimeError(f"Cached archive checksum mismatch: {target}")
    for attempt in range(10):
        try:
            offset = temporary.stat().st_size if temporary.exists() else 0
            request = urllib.request.Request(url, headers={"Range": f"bytes={offset}-"} if offset else {})
            with urllib.request.urlopen(request, timeout=120) as response:
                mode = "ab" if offset and response.status == 206 else "wb"
                with temporary.open(mode) as output:
                    shutil.copyfileobj(response, output, length=1024 * 1024)
            if expected:
                with temporary.open("rb") as stream:
                    actual = hashlib.file_digest(stream, "sha256").hexdigest()
                if actual != expected:
                    raise RuntimeError(f"Checksum mismatch for {url}: {actual}")
            temporary.replace(target)
            return
        except Exception as exc:
            print(f"Download attempt {attempt + 1}: {type(exc).__name__}: {exc}", flush=True)
            if attempt == 9:
                raise
            time.sleep(min(2 ** attempt, 15))


def main():
    CACHE.mkdir(parents=True, exist_ok=True)
    DEST.mkdir(parents=True, exist_ok=True)
    manifest = CACHE / "redistrib_12.4.1.json"
    fetch(BASE + manifest.name, manifest)
    data = json.loads(manifest.read_text())
    installed = {}
    for name in COMPONENTS:
        info = data[name]["linux-x86_64"]
        relative = info["relative_path"]
        archive = CACHE / Path(relative).name
        marker = DEST / ("." + name + ".sha256")
        if marker.is_file() and marker.read_text().strip() == info["sha256"]:
            print("Already installed", name, flush=True)
            installed[name] = info["sha256"]
            continue
        print("Installing", name, "bytes", info.get("size"), flush=True)
        fetch(BASE + relative, archive, info["sha256"])
        with tarfile.open(archive) as contents:
            top = contents.getmembers()[0].name.split("/")[0]
            for member in contents:
                path = Path(member.name)
                if path.parts[0] != top or ".." in path.parts or path.is_absolute():
                    raise RuntimeError(f"Unsafe archive path: {member.name}")
                member.name = str(Path(*path.parts[1:]))
                if member.name == ".":
                    continue
                contents.extract(member, DEST, filter="data")
        marker.write_text(info["sha256"] + "\n")
        installed[name] = info["sha256"]
    # NVIDIA's redistributables use lib/, while nvcc's Linux profile links
    # against lib64/ by default, including the static CUDA runtime.
    lib64 = DEST / "lib64"
    if not lib64.exists():
        lib64.symlink_to("lib", target_is_directory=True)
    (DEST / "installed-components.json").write_text(json.dumps(installed, indent=2) + "\n")
    print("CUDA installed at", DEST, flush=True)


if __name__ == "__main__":
    main()
