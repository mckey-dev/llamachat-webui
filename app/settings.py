"""アプリ設定 (settings.json) の永続化。"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any


DEFAULTS: dict[str, Any] = {
    "llama_server_path": "",
    "host": "127.0.0.1",
    "port": 8080,
    "model_path": "",
    "hf_repo": "",
    "mmproj_path": "",
    "mtp_path": "",
    "models_dir": "",
    "extra_models_dirs": "",
    "ctx_size": 0,
    "n_gpu_layers": "auto",
    "extra_args": "",
    "model_catalogs": {},
    "system_prompt": "",
    # llama-server 推奨（tools/server/README.md）
    "temperature": 0.8,
    "top_k": 40,
    "top_p": 0.95,
    "min_p": 0.05,
    "repeat_penalty": 1.0,
    "presence_penalty": 0.0,
    "frequency_penalty": 0.0,
    "max_tokens": None,
    "seed": None,
    "image_max_side": 1024,
}

SAMPLING_KEYS = (
    "temperature",
    "top_k",
    "top_p",
    "min_p",
    "repeat_penalty",
    "presence_penalty",
    "frequency_penalty",
    "max_tokens",
    "seed",
)

# Models タブのうち、カタログ ID ごとに残す項目。
CATALOG_KEYS = (
    "model_path",
    "hf_repo",
    "mmproj_path",
    "mtp_path",
    "ctx_size",
    "n_gpu_layers",
    "extra_args",
)


class Settings:
    """settings.json の読み書きと既定値の補完。"""

    def __init__(self, path: Path) -> None:
        """設定ファイルを開き、無ければ既定値で作成する。"""
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.data: dict[str, Any] = dict(DEFAULTS)
        self.load()

    def load(self) -> dict[str, Any]:
        """ディスクから読み込み、既知キーだけを既定値へマージする。"""
        if self.path.is_file():
            try:
                raw = json.loads(self.path.read_text(encoding="utf-8"))
                if isinstance(raw, dict):
                    merged = dict(DEFAULTS)
                    merged.update({k: v for k, v in raw.items() if k in DEFAULTS})
                    self.data = merged
                else:
                    self.data = dict(DEFAULTS)
            except (OSError, json.JSONDecodeError):
                self.data = dict(DEFAULTS)
        else:
            self.data = dict(DEFAULTS)
        if _fill_llama_server_sampling(self.data) and self.path.is_file():
            self.save()
        return self.data

    def save(self) -> None:
        """現在の設定を JSON として保存する。"""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(
            json.dumps(self.data, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

    def update(self, values: dict[str, Any]) -> dict[str, Any]:
        """既知キーだけ更新して保存する。"""
        for key, value in values.items():
            if key in DEFAULTS:
                self.data[key] = value
        _fill_llama_server_sampling(self.data)
        self.save()
        return self.data

    def get(self, key: str, default: Any = None) -> Any:
        """設定値を取得する。"""
        return self.data.get(key, default)

    def get_catalog(self, catalog_id: str) -> dict[str, Any]:
        """カタログ ID に保存した Models 欄を返す。無ければ空。"""
        catalogs = self.data.get("model_catalogs")
        if not catalog_id or not isinstance(catalogs, dict):
            return {}
        entry = catalogs.get(catalog_id)
        return dict(entry) if isinstance(entry, dict) else {}

    def upsert_catalog(self, catalog_id: str, values: dict[str, Any]) -> dict[str, Any]:
        """現在の Models 欄を書き、カタログ ID があればそこにも残す。"""
        current = {key: values[key] for key in CATALOG_KEYS if key in values}
        if catalog_id.strip():
            catalogs = dict(self.data.get("model_catalogs") or {})
            if not isinstance(self.data.get("model_catalogs"), dict):
                catalogs = {}
            entry = dict(catalogs.get(catalog_id) or {})
            entry.update(current)
            catalogs[catalog_id] = entry
            current["model_catalogs"] = catalogs
        return self.update(current)


def _blank_or_non_positive(value: Any) -> bool:
    """空欄、または Gradio が空欄を 0 にしたものか。"""
    if value is None:
        return True
    if isinstance(value, str) and not value.strip():
        return True
    number = optional_number(value)
    if number is None:
        return True
    return float(number) <= 0


def _fill_llama_server_sampling(data: dict[str, Any]) -> bool:
    """未設定や 0 のサンプリングを llama-server 推奨値で埋める。"""
    changed = False
    for key in ("temperature", "top_k", "top_p", "min_p", "repeat_penalty"):
        if _blank_or_non_positive(data.get(key)):
            data[key] = DEFAULTS[key]
            changed = True
    for key in ("presence_penalty", "frequency_penalty"):
        if data.get(key) is None:
            data[key] = DEFAULTS[key]
            changed = True
    for key in ("max_tokens", "seed"):
        if _blank_or_non_positive(data.get(key)):
            if data.get(key) is not None:
                changed = True
            data[key] = None
    return changed


def optional_number(value: Any) -> float | int | None:
    """空欄や不正値は None、それ以外は数値にする。"""
    if value is None:
        return None
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        try:
            if "." in text:
                return float(text)
            return int(text)
        except ValueError:
            return None
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        if isinstance(value, float) and not math.isfinite(value):
            return None
        return value
    return None


def sampling_payload(
    *,
    temperature: Any = None,
    top_k: Any = None,
    top_p: Any = None,
    min_p: Any = None,
    repeat_penalty: Any = None,
    presence_penalty: Any = None,
    frequency_penalty: Any = None,
    max_tokens: Any = None,
    seed: Any = None,
) -> dict[str, Any]:
    """サンプリング指定を API 用 dict にする。空欄は載せない（サーバ既定）。"""
    mapping = {
        "temperature": optional_number(temperature),
        "top_k": optional_number(top_k),
        "top_p": optional_number(top_p),
        "min_p": optional_number(min_p),
        "repeat_penalty": optional_number(repeat_penalty),
        "presence_penalty": optional_number(presence_penalty),
        "frequency_penalty": optional_number(frequency_penalty),
        "max_tokens": optional_number(max_tokens),
        "seed": optional_number(seed),
    }
    payload: dict[str, Any] = {}
    for key, value in mapping.items():
        if value is None:
            continue
        number = float(value)
        if not math.isfinite(number):
            continue
        # Gradio の空 Number は 0 になりがち。llama-server は penalty_repeat > 0 が必須。
        if key == "repeat_penalty" and number <= 0:
            continue
        if key == "max_tokens" and int(value) <= 0:
            continue
        if key in {"top_k", "max_tokens", "seed"}:
            payload[key] = int(value)
        else:
            payload[key] = number
    return payload
