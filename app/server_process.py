"""llama-server 子プロセスの起動・停止。"""

from __future__ import annotations

import os
import shlex
import shutil
import subprocess
import sys
import threading
import time
from collections import deque
from pathlib import Path
from typing import Any

from app.install_server import find_binary
from app.llama_client import LlamaClient


class ServerProcess:
    """llama-server を 1 つだけ管理する。"""

    def __init__(self) -> None:
        """停止状態で初期化する。"""
        self.proc: subprocess.Popen[str] | None = None
        self.status: str = "stopped"
        self.error: str = ""
        self.base_url: str = "http://127.0.0.1:8080"
        self.model_label: str = ""
        self.log_lines: deque[str] = deque(maxlen=800)
        self._lock = threading.Lock()
        self._reader: threading.Thread | None = None

    def logs(self) -> str:
        """標準出力ログを結合して返す。"""
        return "\n".join(self.log_lines)

    def is_ready(self) -> bool:
        """プロセスが ready かつ /health が通るか。"""
        return self.status == "ready" and LlamaClient(self.base_url).health()

    def status_line(self) -> str:
        """ヘッダ表示用の1行テキストを返す。"""
        label = self.model_label or "(no model)"
        extra = f" · {self.base_url}" if self.status in {"ready", "starting"} else ""
        err = f" — {self.error}" if self.error and self.status == "error" else ""
        return f"{self.status} · {label}{extra}{err}"

    def spawn(self, settings: dict[str, Any]) -> str | None:
        """llama-server を起動する。/health は待たず、失敗時はエラー文を返す。"""
        with self._lock:
            if self.proc is not None and self.proc.poll() is None:
                return "llama-server is already running. Stop it first."
            try:
                argv = build_argv(settings)
            except ValueError as exc:
                self.status = "error"
                self.error = str(exc)
                return str(exc)

            host = str(settings.get("host") or "127.0.0.1")
            port = int(settings.get("port") or 8080)
            self.base_url = f"http://{host}:{port}"
            self.model_label = _model_label(settings)
            self.error = ""
            self.status = "starting"
            self.log_lines.clear()
            self.log_lines.append(" ".join(_quote(part) for part in argv))

            bin_dir = str(Path(argv[0]).resolve().parent)
            env = os.environ.copy()
            env["PATH"] = bin_dir + os.pathsep + env.get("PATH", "")
            kwargs: dict[str, Any] = {
                "stdout": subprocess.PIPE,
                "stderr": subprocess.STDOUT,
                "text": True,
                "bufsize": 1,
                "encoding": "utf-8",
                "errors": "replace",
                "cwd": bin_dir,
                "env": env,
            }
            if sys.platform == "win32":
                kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
            else:
                kwargs["start_new_session"] = True

            try:
                self.proc = subprocess.Popen(argv, **kwargs)
            except OSError as exc:
                self.status = "error"
                self.error = str(exc)
                self.proc = None
                return f"Failed to start llama-server: {exc}"

            self._reader = threading.Thread(
                target=self._pump,
                args=(self.proc.stdout,),
                daemon=True,
            )
            self._reader.start()
        return None

    def poll_ready(self) -> str:
        """プロセスと /health を見て starting / ready / error / stopped を返す。"""
        if self.status in {"stopped", "error"} and self.proc is None:
            return self.status
        if self.proc is not None and self.proc.poll() is not None:
            self.status = "error"
            self.error = "llama-server exited before becoming ready"
            self.proc = None
            return "error"
        client = LlamaClient(self.base_url)
        if client.health():
            self.status = "ready"
            self.error = ""
            names = client.models()
            if names:
                self.model_label = names[0]
            return "ready"
        return self.status or "starting"

    def start(self, settings: dict[str, Any], timeout: float = 300.0) -> str:
        """起動して /health 成功まで待つ（CLI / テスト用）。"""
        err = self.spawn(settings)
        if err:
            return err
        deadline = time.time() + timeout
        while time.time() < deadline:
            state = self.poll_ready()
            if state == "ready":
                return f"Ready · {self.model_label} · {self.base_url}"
            if state == "error":
                return f"{self.error}. Last log:\n{self.logs()[-2000:]}"
            time.sleep(0.4)
        self.status = "error"
        self.error = f"Timed out waiting for /health ({timeout:.0f}s)"
        return f"{self.error}\n{self.logs()[-2000:]}"

    def stop(self) -> str:
        """子プロセスを terminate / kill する。"""
        with self._lock:
            proc = self.proc
            self.proc = None
        if proc is None or proc.poll() is not None:
            self.status = "stopped"
            self.error = ""
            return "Already stopped."
        _terminate(proc)
        self.status = "stopped"
        self.error = ""
        self.model_label = ""
        return "Stopped."

    def _pump(self, stream: Any) -> None:
        """stdout を行単位でログへ蓄積する（バックグラウンドスレッド）。"""
        if stream is None:
            return
        try:
            for line in iter(stream.readline, ""):
                self.log_lines.append(line.rstrip("\n"))
        except Exception:
            pass
        try:
            stream.close()
        except Exception:
            pass
        if self.proc is not None and self.proc.poll() is not None and self.status == "ready":
            self.status = "error"
            self.error = "llama-server process exited"


def resolve_llama_server(configured: str | None, install_dir: str | None = None) -> str | None:
    """設定パス → 環境変数 → PATH → 導入ディレクトリの順でバイナリを探す。"""
    candidates: list[str] = []
    if configured and configured.strip():
        candidates.append(configured.strip())
    env = os.environ.get("LLAMA_SERVER") or os.environ.get("LLAMA_SERVER_PATH")
    if env:
        candidates.append(env)
    for raw in candidates:
        path = Path(raw).expanduser()
        if path.is_file():
            return str(path)
    found = shutil.which("llama-server") or shutil.which("llama-server.exe")
    if found:
        return found
    if install_dir:
        binary = find_binary(Path(install_dir))
        if binary:
            return str(binary)
    return None


def build_argv(settings: dict[str, Any]) -> list[str]:
    """設定から llama-server のコマンドライン引数を組み立てる。"""
    binary = resolve_llama_server(
        str(settings.get("llama_server_path") or ""),
        install_dir=str(settings.get("llama_install_dir") or "") or None,
    )
    if not binary:
        raise ValueError(
            "llama-server binary not found. Use Models → Install llama-server, "
            "set the path, or set LLAMA_SERVER."
        )

    host = str(settings.get("host") or "127.0.0.1")
    port = int(settings.get("port") or 8080)
    argv = [
        binary,
        "--host",
        host,
        "--port",
        str(port),
        "--jinja",
        "--no-webui",
        "--reasoning-format",
        "auto",
    ]

    model_path = str(settings.get("model_path") or "").strip()
    hf_repo = str(settings.get("hf_repo") or "").strip()
    if model_path:
        argv += ["-m", model_path]
    elif hf_repo:
        argv += ["-hf", hf_repo]
    else:
        raise ValueError("Set a local GGUF path or a Hugging Face repo (-hf).")

    mmproj = str(settings.get("mmproj_path") or "").strip()
    if mmproj:
        argv += ["--mmproj", mmproj]

    mtp = str(settings.get("mtp_path") or "").strip()
    extra = str(settings.get("extra_args") or "").strip()
    if mtp:
        argv += ["--model-draft", mtp]
        extra_l = extra.lower()
        if "--spec-type" not in extra_l:
            argv += ["--spec-type", "draft-mtp"]
        if "--spec-draft-n-max" not in extra_l:
            argv += ["--spec-draft-n-max", "4"]

    ctx = int(settings.get("ctx_size") or 0)
    if ctx > 0:
        argv += ["-c", str(ctx)]

    ngl = str(settings.get("n_gpu_layers") or "").strip()
    if ngl:
        argv += ["-ngl", ngl]

    alias = _model_label(settings)
    if alias:
        argv += ["--alias", alias]

    if extra:
        posix = os.name != "nt"
        argv += shlex.split(extra, posix=posix)

    return argv


def _model_label(settings: dict[str, Any]) -> str:
    """--alias と状態表示に使う短いモデル名を返す。"""
    model_path = str(settings.get("model_path") or "").strip()
    if model_path:
        return Path(model_path).stem
    hf_repo = str(settings.get("hf_repo") or "").strip()
    if hf_repo:
        return hf_repo.split("/")[-1]
    return ""


def _quote(part: str) -> str:
    """ログ表示用に空白を含む引数を引用符で囲む。"""
    if any(ch in part for ch in ' \t"'):
        return f'"{part}"'
    return part


def _terminate(proc: subprocess.Popen[str]) -> None:
    """プロセスを terminate し、応答が無ければ kill する。"""
    try:
        proc.terminate()
    except OSError:
        pass
    try:
        proc.wait(timeout=8)
        return
    except subprocess.TimeoutExpired:
        pass
    try:
        proc.kill()
        proc.wait(timeout=5)
    except (OSError, subprocess.TimeoutExpired):
        pass
