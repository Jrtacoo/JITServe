"""Exercise reproduction lifecycle without requiring CUDA or loading vLLM."""
import importlib.util
import json
from pathlib import Path
import socket
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import SimpleNamespace

import pytest

SPEC = importlib.util.spec_from_file_location(
    "reproduce", Path(__file__).resolve().parents[1] / "scripts/reproduce.py")
reproduce = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(reproduce)


@pytest.fixture
def http_server():
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(200)
            self.end_headers()

        def do_POST(self):
            payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            self.send_response(200)
            self.end_headers()
            # vLLM includes the prompt in the returned text. Echoing only the
            # prompt must not count as a successful generation.
            self.wfile.write(json.dumps({"text": [payload["request_info"]["prompt"]]}).encode())

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield server
    server.shutdown()
    server.server_close()
    thread.join(timeout=5)


def test_local_health_bypasses_environment_proxy(http_server, monkeypatch):
    for key in ("http_proxy", "HTTP_PROXY", "https_proxy", "HTTPS_PROXY"):
        monkeypatch.setenv(key, "http://127.0.0.1:1")
    monkeypatch.setenv("NO_PROXY", "")
    monkeypatch.setenv("no_proxy", "")
    port = http_server.server_address[1]
    assert reproduce.http_json(f"http://127.0.0.1:{port}/health") == {}


def test_smoke_rejects_prompt_without_generated_text(http_server):
    args = SimpleNamespace(port=http_server.server_address[1])
    with pytest.raises(RuntimeError, match="no text"):
        reproduce.smoke_request(args)


def test_occupied_port_fails_before_starting_service(http_server):
    with pytest.raises(OSError):
        reproduce.ensure_free_port(http_server.server_address[1])


def test_completed_connection_does_not_block_service_restart():
    with socket.socket() as listener:
        listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
        listener.listen(1)
        with socket.create_connection(("127.0.0.1", port), timeout=5) as client:
            connection, _ = listener.accept()
            with connection:
                connection.shutdown(socket.SHUT_WR)
                assert client.recv(1) == b""
    reproduce.ensure_free_port(port)


def test_readiness_fails_immediately_for_exited_service():
    process = subprocess.Popen([sys.executable, "-c", "raise SystemExit(7)"])
    process.wait(timeout=10)
    with pytest.raises(RuntimeError, match="code 7"):
        reproduce.wait_ready(process, lambda: True, timeout=10)


def test_readiness_timeout_cleans_up_child(tmp_path):
    with reproduce.service([sys.executable, "-c", "import time; time.sleep(60)"],
                           tmp_path / "service.log") as process:
        with pytest.raises(TimeoutError):
            reproduce.wait_ready(process, lambda: False, timeout=0.1)
    assert process.poll() is not None


def test_mixed_trace_has_all_request_types_and_is_repeatable(tmp_path):
    path = reproduce.prepare_trace(6, tmp_path)
    first = path.read_bytes()
    trace = json.loads(first)
    assert len(trace) == 6
    assert [item["request_type"] for item in trace] == [0, 1, 2, 0, 1, 2]
    assert reproduce.prepare_trace(6, tmp_path).read_bytes() == first


def test_missing_qrf_resources_fail_before_launch(tmp_path, monkeypatch):
    monkeypatch.setattr(reproduce, "ROOT", tmp_path)
    with pytest.raises(FileNotFoundError, match="download"):
        reproduce.predictor_command(SimpleNamespace(qrf_family="qwen25_7b"))
