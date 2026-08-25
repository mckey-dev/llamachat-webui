"""settings.json と Gradio フォームの対応。"""

from __future__ import annotations

from typing import Any

from app.settings import DEFAULTS, optional_number, sampling_payload

# Gradio Number の既定 step=1 だと 0.8 / 0.95 が HTML の step mismatch で 0 に見える。
STEP_ANY: Any = "any"

# collect_form / load / demo.load で同じ並びにする。
FORM_KEYS = (
    "llama_server_path",
    "host",
    "port",
    "model_path",
    "hf_repo",
    "mmproj_path",
    "mtp_path",
    "ctx_size",
    "n_gpu_layers",
    "extra_args",
    "models_dir",
    "extra_models_dirs",
    "system_prompt",
    "temperature",
    "top_k",
    "top_p",
    "min_p",
    "repeat_penalty",
    "presence_penalty",
    "frequency_penalty",
    "max_tokens",
    "seed",
    "image_max_side",
)
SAMPLING_FORM_KEYS = FORM_KEYS[FORM_KEYS.index("system_prompt") :]


def setting_num(data: dict[str, Any], key: str, fallback: Any = None) -> Any:
    """settings.json の数値を Gradio Number 用にする。空欄は fallback。"""
    value = data.get(key, fallback)
    if value is None or value == "":
        return fallback
    parsed = optional_number(value)
    return fallback if parsed is None else parsed


def setting_int(data: dict[str, Any], key: str, fallback: int | None = None) -> int | None:
    """settings.json の数値を Gradio Number 用の int にする。"""
    value = setting_num(data, key, fallback)
    if value is None:
        return fallback
    try:
        return int(value)
    except (TypeError, ValueError):
        return fallback


def nullable_text(data: dict[str, Any], key: str) -> str:
    """null の数値設定を空の Textbox にする。Gradio Number は空を 0 にしてしまう。"""
    value = data.get(key)
    if value is None or value == "":
        return ""
    return str(value)


def form_value(data: dict[str, Any], key: str) -> Any:
    """FORM_KEYS 1 件を UI の初期値 / load 出力にする。"""
    if key in {"max_tokens", "seed"}:
        return nullable_text(data, key)
    if key == "port":
        return setting_int(data, key, 8080)
    if key == "ctx_size":
        return setting_int(data, key, 0)
    if key == "image_max_side":
        return setting_int(data, key, 1024) or 1024
    if key == "n_gpu_layers":
        return str(data.get(key) or "auto")
    if key in DEFAULTS and not isinstance(DEFAULTS[key], str):
        return setting_num(data, key, DEFAULTS[key])
    return data.get(key) or ""


def collect_form(*values: Any) -> dict[str, Any]:
    """フォーム入力を settings.json 用 dict にする。"""
    raw = dict(zip(FORM_KEYS, values, strict=True))
    rpt = optional_number(raw["repeat_penalty"])
    return {
        "llama_server_path": raw["llama_server_path"] or "",
        "host": raw["host"] or "127.0.0.1",
        "port": int(raw["port"] or 8080),
        "model_path": raw["model_path"] or "",
        "hf_repo": raw["hf_repo"] or "",
        "mmproj_path": raw["mmproj_path"] or "",
        "mtp_path": raw["mtp_path"] or "",
        "ctx_size": int(raw["ctx_size"] or 0),
        "n_gpu_layers": str(raw["n_gpu_layers"] or "").strip() or "auto",
        "extra_args": raw["extra_args"] or "",
        "models_dir": raw["models_dir"] or "",
        "extra_models_dirs": raw["extra_models_dirs"] or "",
        "system_prompt": raw["system_prompt"] or "",
        "temperature": optional_number(raw["temperature"]),
        "top_k": optional_number(raw["top_k"]),
        "top_p": optional_number(raw["top_p"]),
        "min_p": optional_number(raw["min_p"]),
        "repeat_penalty": None if rpt is None or float(rpt) <= 0 else float(rpt),
        "presence_penalty": optional_number(raw["presence_penalty"]),
        "frequency_penalty": optional_number(raw["frequency_penalty"]),
        "max_tokens": optional_number(raw["max_tokens"]),
        "seed": optional_number(raw["seed"]),
        "image_max_side": int(raw["image_max_side"] or 1024),
    }


def parse_sampling(*values: Any) -> tuple[dict[str, Any], dict[str, Any]]:
    """Settings タブのサンプリング欄を API 用 payload と生 dict にする。"""
    sample = dict(zip(SAMPLING_FORM_KEYS, values, strict=True))
    payload = sampling_payload(
        temperature=sample["temperature"],
        top_k=sample["top_k"],
        top_p=sample["top_p"],
        min_p=sample["min_p"],
        repeat_penalty=sample["repeat_penalty"],
        presence_penalty=sample["presence_penalty"],
        frequency_penalty=sample["frequency_penalty"],
        max_tokens=sample["max_tokens"],
        seed=sample["seed"],
    )
    return payload, sample
