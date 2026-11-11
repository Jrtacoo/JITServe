#!/usr/bin/env python3
"""Prepare resources and run small, health-checked JITServe experiments."""
import argparse
import contextlib
import importlib.util
import json
import os
from pathlib import Path
import pickle
import signal
import socket
import subprocess
import sys
import time
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
QRF_REPO = "En-2863/jitserve-qrf-length-predictor"
QRF_REVISION = "1e587124a9671f54f4c659f97bccbbe890b3d8af"


def qrf_paths(family):
    filename = f"0_qrf_lmsys_chat_{family}.pkl"
    return (ROOT / "assets/qrf/qrf_model" / filename,
            ROOT / "assets/qrf/qrf_vectorizer" / filename)


def require_environment():
    missing = [name for name in ("torch", "vllm", "jitserve")
               if importlib.util.find_spec(name) is None]
    if missing:
        raise RuntimeError("Missing packages: " + ", ".join(missing) +
                           ". See REPRODUCE.md for the GPU environment setup.")
    spec = importlib.util.find_spec("vllm")
    expected = ROOT / "third_party/vllm/vllm"
    if not Path(spec.origin).resolve().is_relative_to(expected):
        raise RuntimeError(f"vLLM must be installed from {expected.parent}; found {spec.origin}")
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is unavailable in this process. Expose the GPU to the container first.")
    print(f"PyTorch {torch.__version__}; CUDA {torch.version.cuda}", flush=True)
    for index in range(torch.cuda.device_count()):
        prop = torch.cuda.get_device_properties(index)
        print(f"GPU {index}: {prop.name}, {prop.total_memory / 2**30:.1f} GiB", flush=True)


def ensure_free_port(port):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        # A completed experiment can leave accepted connections in TIME_WAIT.
        # Match the HTTP server's reuse behavior while still rejecting listeners.
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind(("127.0.0.1", port))


def wait_ready(process, probe, timeout):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"Service exited with code {process.returncode}; inspect its log")
        try:
            if probe():
                if process.poll() is not None:
                    raise RuntimeError("Service exited during readiness check")
                return
        except (OSError, ValueError, EOFError, pickle.UnpicklingError):
            pass
        time.sleep(0.5)
    raise TimeoutError(f"Service was not ready after {timeout}s; inspect its log")


def http_json(url, payload=None, timeout=10):
    # Local health and inference requests must bypass external proxies.
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    data = None if payload is None else json.dumps(payload).encode()
    request = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    with opener.open(request, timeout=timeout) as response:
        body = response.read()
        return json.loads(body) if body else {}


def qrf_probe():
    from jitserve.request_analyzer.prediction_protocol import send_message, receive_message
    with socket.create_connection(("localhost", 65433), timeout=5) as sock:
        sock.settimeout(10)
        send_message(sock, {"prompt": ["Explain why the sky is blue."]})
        result = receive_message(sock)
        return int(result["output_len"][0]) > 0


@contextlib.contextmanager
def service(command, logfile):
    with open(logfile, "w") as output:
        process = subprocess.Popen(command, cwd=ROOT, stdout=output,
                                   stderr=subprocess.STDOUT, start_new_session=True)
        try:
            yield process
        finally:
            # Terminate the engine workers along with their parent process.
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                process.wait(timeout=20)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()


def download(args):
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    from huggingface_hub import HfApi, snapshot_download
    model, vectorizer = qrf_paths(args.qrf_family)
    snapshot_download(QRF_REPO, revision=QRF_REVISION,
                      allow_patterns=[f"qrf_model/{model.name}",
                                      f"qrf_vectorizer/{vectorizer.name}", "metadata.json", "README.md"],
                      local_dir=ROOT / "assets/qrf", max_workers=2)
    manifest = {"qrf_repo": QRF_REPO, "qrf_revision": QRF_REVISION,
                "qrf_family": args.qrf_family}
    if not args.assets_only:
        revision = HfApi().model_info(args.model).sha
        target = ROOT.parent / "models" / args.model.split("/")[-1]
        snapshot_download(args.model, revision=revision, local_dir=target, max_workers=2,
                          allow_patterns=["*.json", "*.safetensors", "*.txt", "*.model", "*.tiktoken"])
        manifest.update(model=args.model, model_revision=revision, model_path=str(target))
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "downloads.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2), flush=True)


def predictor_command(args):
    model, vectorizer = qrf_paths(args.qrf_family)
    for path in (model, vectorizer):
        if not path.is_file():
            raise FileNotFoundError(f"Missing {path}; run the download action first")
    return [sys.executable, "-u", "-m", "jitserve.request_analyzer.prediction",
            "--model-path", str(model), "--tokenizer-path", str(vectorizer)]


def server_command(args, policy):
    command = [sys.executable, "-u", "-m", "jitserve.server", "--host", "127.0.0.1",
               "--port", str(args.port), "--model", args.model,
               "--scheduling-policy", policy, "--max-model-len", str(args.max_model_len),
               "--max-num-seqs", str(args.batch_size), "--gpu-memory-utilization", str(args.gpu_memory),
               "--enable-chunked-prefill", "True", "--penalty-factor", "100"]
    if policy == "fcfs":
        command += ["--disable-prediction", "--disable-graph-matching", "--graph-matching-mode", "none"]
    else:
        model, vectorizer = qrf_paths(args.qrf_family)
        command += ["--prediction-model-path", str(model), "--prediction-tokenizer-path", str(vectorizer)]
    return command


def smoke_request(args):
    prompt = "Explain why the sky is blue in one sentence."
    payload = {"client_id": 0, "stream": False, "slo_constraint": [60, 10, 120],
               "sampling_params": {"max_tokens": 32, "temperature": 0.0},
               "request_info": {"prompt": prompt, "request_type": 0, "collection_id": 0,
                                "deadline": 120, "input_len": 0, "output_len": 32}}
    result = http_json(f"http://127.0.0.1:{args.port}/generate", payload, timeout=180)
    if not result.get("text") or not any(text.removeprefix(prompt).strip() for text in result["text"]):
        raise RuntimeError(f"Generation returned no text: {result}")
    return result


def prepare_trace(count, output):
    if count < 3:
        raise ValueError("Mixed benchmark requires at least three prompts")
    source = json.loads((ROOT / "traces/lmsys.json").read_text())
    buckets = {kind: [item for item in source if item["request_type"] == kind]
               for kind in range(3)}
    if any(not bucket for bucket in buckets.values()):
        raise ValueError("Source trace must contain latency, throughput and collective requests")
    trace = [buckets[index % 3][(index // 3) % len(buckets[index % 3])]
             for index in range(count)]
    path = output / "mixed-trace.json"
    path.write_text(json.dumps(trace) + "\n")
    return path


def run_experiments(args):
    require_environment()
    args.output.mkdir(parents=True, exist_ok=True)
    policies = args.policies.split(",")
    if args.action == "serve" and len(policies) != 1:
        raise ValueError("serve requires exactly one policy; use --policies jitserve or --policies fcfs")
    if any(policy not in ("fcfs", "jitserve") for policy in policies):
        raise ValueError("Supported reproduction policies: fcfs,jitserve")
    trace_path = prepare_trace(args.num_prompts, args.output) if args.action == "benchmark" else None
    for policy in policies:
        ensure_free_port(args.port)
        with contextlib.ExitStack() as stack:
            if policy == "jitserve":
                ensure_free_port(65433)
                predictor = stack.enter_context(service(predictor_command(args), args.output / "qrf.log"))
                wait_ready(predictor, qrf_probe, args.startup_timeout)
            command = server_command(args, policy)
            (args.output / f"{policy}-command.json").write_text(json.dumps(command, indent=2) + "\n")
            server = stack.enter_context(service(command, args.output / f"{policy}-server.log"))
            wait_ready(server, lambda: http_json(f"http://127.0.0.1:{args.port}/health") == {},
                       args.startup_timeout)
            result = smoke_request(args)
            (args.output / f"{policy}-smoke.json").write_text(json.dumps(result, indent=2) + "\n")
            print(f"{policy}: smoke request passed", flush=True)
            if args.action == "serve":
                print(f"JITServe is ready at http://127.0.0.1:{args.port}; press Ctrl-C to stop", flush=True)
                while server.poll() is None:
                    if policy == "jitserve" and predictor.poll() is not None:
                        raise RuntimeError("QRF predictor exited; inspect qrf.log")
                    time.sleep(1)
                raise RuntimeError(f"Server exited with code {server.returncode}")
            if args.action == "benchmark":
                command = [sys.executable, "benchmark/benchmark_scheduler.py", "--model", args.model,
                           "--host", "127.0.0.1", "--port", str(args.port), "--policy", policy,
                           "--num-prompts", str(args.num_prompts), "--arrival-rate", str(args.arrival_rate),
                           "--batch-size", str(args.batch_size), "--trace-path", str(trace_path),
                           "--max-output-len", "128", "--slo-constraint", "2,0.1,20", "--penalty-factor", "100",
                           "--result-dir", str(args.output), "--result-filename", f"{policy}-metrics.json"]
                with open(args.output / f"{policy}-benchmark.log", "w") as output:
                    subprocess.run(command, cwd=ROOT, stdout=output, stderr=subprocess.STDOUT,
                                   check=True, timeout=args.benchmark_timeout)
                result = json.loads((args.output / f"{policy}-metrics.json").read_text())
                completed = result["requests"]["request_completed"][3]
                total = result["requests"]["request_num"][3]
                if completed == 0:
                    raise RuntimeError(f"{policy}: benchmark completed no requests")
                print(f"{policy}: benchmark completed {completed}/{total} requests; see saved metrics", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["check", "download", "qrf", "smoke", "benchmark", "serve"])
    parser.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--qrf-family", choices=["qwen25_7b", "llama3_8b"], default="qwen25_7b")
    parser.add_argument("--assets-only", action="store_true")
    parser.add_argument("--policies", default="fcfs,jitserve")
    parser.add_argument("--port", type=int, default=8100)
    parser.add_argument("--max-model-len", type=int, default=8192)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--gpu-memory", type=float, default=0.85)
    parser.add_argument("--num-prompts", type=int, default=6)
    parser.add_argument("--arrival-rate", type=float, default=0.5)
    parser.add_argument("--startup-timeout", type=float, default=600)
    parser.add_argument("--benchmark-timeout", type=float, default=1800)
    parser.add_argument("--output", type=Path, default=ROOT / "reproduction_results")
    args = parser.parse_args()
    if args.action == "check":
        require_environment()
    elif args.action == "download":
        download(args)
    elif args.action == "qrf":
        ensure_free_port(65433)
        raise SystemExit(subprocess.call(predictor_command(args), cwd=ROOT))
    else:
        run_experiments(args)


if __name__ == "__main__":
    def stop_service(signum, frame):
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, stop_service)
    try:
        main()
    except KeyboardInterrupt:
        print("JITServe stopped", file=sys.stderr)
        raise SystemExit(0)
    except (RuntimeError, OSError, TimeoutError, ValueError, subprocess.SubprocessError) as exc:
        print(f"Reproduction failed: {exc}", file=sys.stderr)
        raise SystemExit(1)
