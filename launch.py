#!/usr/bin/env python3
"""Gradio UI の起動エントリ。"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
# Notebook / uv の PYTHONPATH や PYTHONSAFEPATH だと、スクリプトのある場所が
# import パスに入らず、別の不完全な app が先に見つかることがある。
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _require_app_package() -> None:
    """app.settings が import できる状態か確認する。"""
    needed = ("__init__.py", "settings.py", "ui.py")
    missing = [name for name in needed if not (ROOT / "app" / name).is_file()]
    if missing:
        raise SystemExit(
            "リポジトリの app/ が不完全です: "
            + ", ".join(missing)
            + f"\nROOT={ROOT}\n"
            "ノートだけ手動アップロードした場合は、残りのファイルを git clone するか、"
            "app/ 一式をこの ROOT 配下へコピーしてください。"
        )


def _ensure_llama_server(data_dir: Path, *, backend: str, tag: str, force: bool, skip: bool) -> str | None:
    """llama-server が無ければ公式リリースから導入し、パスを返す。"""
    from app.install_server import InstallError, install_blocking
    from app.server_process import resolve_llama_server
    from app.settings import Settings

    settings = Settings(data_dir / "settings.json")
    install_dir = data_dir / "llama-server"
    found = resolve_llama_server(str(settings.get("llama_server_path") or ""), str(install_dir))
    if found and not force:
        if not str(settings.get("llama_server_path") or "").strip():
            settings.data["llama_server_path"] = found
        print(f"Using llama-server: {found}")
        return found
    if skip:
        if found:
            print(f"Using llama-server: {found}")
            return found
        print("llama-server not found and --skip-install was set.")
        return None

    print("Installing llama-server from GitHub Releases (first run or --install-server)...")
    try:
        binary = install_blocking(data_dir, backend=backend, tag=tag)
    except (InstallError, Exception) as exc:
        raise SystemExit(f"llama-server install failed: {exc}") from exc
    settings.update({"llama_server_path": binary})
    print(f"llama_server_path={binary}")
    return binary


def main() -> None:
    """引数を解釈し、必要なら llama-server を入れて UI を起動する。"""
    parser = argparse.ArgumentParser(description="llamachat-webui - Gradio frontend for llama-server")
    parser.add_argument("--server-name", default="127.0.0.1", help="Gradio bind address")
    parser.add_argument("--server-port", type=int, default=7862, help="Gradio bind port")
    parser.add_argument("--share", action="store_true", help="Create a Gradio public URL")
    parser.add_argument(
        "--data-dir",
        default=str(ROOT / "data"),
        help="Directory for settings, conversations, and uploads",
    )
    parser.add_argument(
        "--install-server",
        action="store_true",
        help="Re-download llama-server even if a binary is already present",
    )
    parser.add_argument(
        "--skip-install",
        action="store_true",
        help="Do not download llama-server; fail later if none is found",
    )
    parser.add_argument(
        "--llama-backend",
        default=os.environ.get("LLAMA_BACKEND", "auto"),
        help="Install backend: auto, cpu, cuda, cuda-12.4, cuda-13.3, vulkan, rocm",
    )
    parser.add_argument(
        "--llama-tag",
        default=os.environ.get("LLAMA_TAG", ""),
        help="llama.cpp release tag (default: latest)",
    )
    args = parser.parse_args()
    _require_app_package()

    data_dir = Path(args.data_dir).resolve()
    data_dir.mkdir(parents=True, exist_ok=True)

    skip = args.skip_install or os.environ.get("LLAMA_SKIP_INSTALL", "").strip() in {"1", "true", "yes"}
    _ensure_llama_server(
        data_dir,
        backend=args.llama_backend,
        tag=args.llama_tag,
        force=args.install_server,
        skip=skip,
    )

    from app.ui import CSS, build_app

    demo = build_app(data_dir)
    demo.queue()
    demo.launch(
        server_name=args.server_name,
        server_port=args.server_port,
        share=args.share,
        css=CSS,
        allowed_paths=[str(data_dir)],
    )


if __name__ == "__main__":
    main()
