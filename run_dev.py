"""Start the FastAPI service and Flask web client for local development."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path
from urllib.error import URLError
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parent


def wait_for_api(process: subprocess.Popen, health_url: str, timeout: float = 120) -> None:
    deadline = time.monotonic() + timeout
    last_error: Exception | None = None
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"API đã dừng với mã thoát {process.returncode}.")
        try:
            with urlopen(health_url, timeout=1) as response:
                if response.status == 200:
                    return
        except (TimeoutError, URLError, OSError) as exc:
            last_error = exc
        time.sleep(0.4)
    raise TimeoutError(f"API không sẵn sàng trong {timeout:.0f} giây: {last_error}")


def stop_processes(processes: list[subprocess.Popen]) -> None:
    for process in processes:
        if process.poll() is None:
            process.terminate()
    for process in processes:
        if process.poll() is None:
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--api-host", default="127.0.0.1")
    parser.add_argument("--api-port", type=int, default=8000)
    parser.add_argument("--web-host", default="127.0.0.1")
    parser.add_argument("--web-port", type=int, default=5000)
    args = parser.parse_args()

    environment = os.environ.copy()
    environment.setdefault("PYTHONUTF8", "1")
    environment.setdefault("API_URL", f"http://{args.api_host}:{args.api_port}")
    api_command = [
        sys.executable,
        "-m",
        "uvicorn",
        "api.main:app",
        "--host",
        args.api_host,
        "--port",
        str(args.api_port),
    ]
    web_command = [
        sys.executable,
        str(ROOT / "web" / "app.py"),
    ]
    processes: list[subprocess.Popen] = []
    try:
        print(f"Starting API at {environment['API_URL']} ...", flush=True)
        processes.append(
            subprocess.Popen(api_command, cwd=ROOT, env=environment)
        )
        wait_for_api(
            processes[0],
            f"http://{args.api_host}:{args.api_port}/health",
        )
        print(f"Starting web at http://{args.web_host}:{args.web_port} ...", flush=True)
        web_environment = environment.copy()
        web_environment["FLASK_HOST"] = args.web_host
        web_environment["FLASK_PORT"] = str(args.web_port)
        processes.append(
            subprocess.Popen(web_command, cwd=ROOT, env=web_environment)
        )
        while all(process.poll() is None for process in processes):
            time.sleep(0.5)
        return next(
            (process.returncode for process in processes if process.returncode),
            0,
        )
    except KeyboardInterrupt:
        return 0
    except (OSError, RuntimeError, TimeoutError) as exc:
        print(f"Không khởi động được OutfitAI: {exc}", file=sys.stderr)
        return 1
    finally:
        stop_processes(processes)


if __name__ == "__main__":
    raise SystemExit(main())
