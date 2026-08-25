"""UI 全体で共有する状態。"""

from __future__ import annotations

import os
from pathlib import Path

from app.conversations import ConversationStore
from app.llama_client import LlamaClient
from app.server_process import ServerProcess, resolve_llama_server
from app.settings import Settings


class AppContext:
    """設定・会話・プロセスなど、UI が共有する状態。"""

    def __init__(self, data_dir: Path) -> None:
        """data 配下の設定と会話ストアを開く。空の models / conversations / uploads は作らない。"""
        self.data_dir = data_dir
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.settings = Settings(data_dir / "settings.json")
        self.store = ConversationStore(data_dir / "conversations")
        self.uploads = data_dir / "uploads"
        self.process = ServerProcess()
        env_server = os.environ.get("LLAMA_SERVER", "").strip()
        if env_server:
            self.settings.data["llama_server_path"] = env_server
        elif not str(self.settings.get("llama_server_path") or "").strip():
            found = resolve_llama_server("", str(data_dir / "llama-server"))
            if found:
                self.settings.data["llama_server_path"] = found


def llama_client(ctx: AppContext) -> LlamaClient:
    """現在の llama-server URL 向けクライアントを返す。"""
    return LlamaClient(ctx.process.base_url)
