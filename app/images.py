"""Vision 送信用の画像処理。"""

from __future__ import annotations

import base64
import shutil
import uuid
from io import BytesIO
from pathlib import Path
from typing import Any

from PIL import Image


def file_path(item: Any) -> Path | None:
    """Gradio のファイル値から実在するパスを取り出す。"""
    if item is None:
        return None
    if isinstance(item, Path):
        return item if item.is_file() else None
    if isinstance(item, str):
        path = Path(item)
        return path if path.is_file() else None
    if isinstance(item, dict):
        for key in ("path", "name", "orig_name"):
            raw = item.get(key)
            if raw:
                path = Path(str(raw))
                if path.is_file():
                    return path
    return None


def persist_images(files: list[Any], dest_dir: Path) -> list[str]:
    """アップロード画像を会話用ディレクトリへコピーし、保存パス一覧を返す。"""
    dest_dir.mkdir(parents=True, exist_ok=True)
    saved: list[str] = []
    for item in files or []:
        src = file_path(item)
        if src is None:
            continue
        suffix = src.suffix.lower() if src.suffix else ".png"
        dest = dest_dir / f"{uuid.uuid4().hex}{suffix}"
        shutil.copy2(src, dest)
        saved.append(str(dest))
    return saved


def image_to_data_url(path: str | Path, max_side: int = 1024) -> str:
    """画像を長辺制限つきで JPEG 化し、data URL にする。"""
    src = Path(path)
    image = Image.open(src)
    if image.mode not in ("RGB", "L"):
        image = image.convert("RGB")
    elif image.mode == "L":
        image = image.convert("RGB")

    if max_side and max(image.size) > max_side:
        image.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)

    buf = BytesIO()
    image.save(buf, format="JPEG", quality=85)
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{b64}"
