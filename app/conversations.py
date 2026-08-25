"""会話 JSON の保存と、Gradio / llama-server 形式への変換。"""

from __future__ import annotations

import json
import re
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from app.images import image_to_data_url


ISO = "%Y-%m-%dT%H:%M:%S.%fZ"


def _now() -> str:
    """現在時刻を UTC の ISO 文字列で返す。"""
    return datetime.now(timezone.utc).strftime(ISO)


def _title_from_text(text: str) -> str:
    """ユーザー発言から会話タイトルを短く作る。"""
    cleaned = re.sub(r"\s+", " ", (text or "").strip())
    if not cleaned:
        return "New chat"
    if len(cleaned) > 48:
        return cleaned[:45] + "..."
    return cleaned


class ConversationStore:
    """会話を data/conversations/*.json として管理する。"""

    def __init__(self, directory: Path) -> None:
        """保存先ディレクトリを用意する。"""
        self.directory = directory

    def _ensure_dir(self) -> None:
        """保存先が無ければ作る。"""
        self.directory.mkdir(parents=True, exist_ok=True)

    def _path(self, conv_id: str) -> Path:
        """会話 ID に対応する JSON パスを返す。"""
        return self.directory / f"{conv_id}.json"

    def list_summaries(self) -> list[dict[str, str]]:
        """更新日時の新しい順で、id / title / updated_at の一覧を返す。"""
        items: list[dict[str, str]] = []
        if not self.directory.is_dir():
            return items
        for path in self.directory.glob("*.json"):
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if not isinstance(data, dict) or not data.get("id"):
                continue
            items.append(
                {
                    "id": str(data["id"]),
                    "title": str(data.get("title") or "New chat"),
                    "updated_at": str(data.get("updated_at") or ""),
                }
            )
        items.sort(key=lambda row: row.get("updated_at") or "", reverse=True)
        return items

    def choices(self) -> list[tuple[str, str]]:
        """Dropdown 用の (タイトル, id) リストを返す。"""
        return [(row["title"], row["id"]) for row in self.list_summaries()]

    def get(self, conv_id: str | None) -> dict[str, Any] | None:
        """会話を 1 件読み込む。無ければ None。"""
        if not conv_id:
            return None
        path = self._path(conv_id)
        if not path.is_file():
            return None
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        return data if isinstance(data, dict) else None

    def save(self, conv: dict[str, Any]) -> dict[str, Any]:
        """会話を保存し、updated_at を更新する。"""
        conv["updated_at"] = _now()
        if not conv.get("created_at"):
            conv["created_at"] = conv["updated_at"]
        self._ensure_dir()
        path = self._path(str(conv["id"]))
        path.write_text(json.dumps(conv, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        return conv

    def create(self, title: str = "New chat") -> dict[str, Any]:
        """空の会話を新規作成する。"""
        conv = {
            "id": uuid.uuid4().hex,
            "title": title,
            "created_at": _now(),
            "updated_at": _now(),
            "messages": [],
        }
        return self.save(conv)

    def delete(self, conv_id: str) -> None:
        """会話 JSON を削除する。"""
        path = self._path(conv_id)
        if path.is_file():
            path.unlink()

    def append_user(self, conv_id: str, text: str, images: list[str]) -> dict[str, Any]:
        """ユーザー発言を追加し、必要ならタイトルを付ける。"""
        conv = self.get(conv_id) or self.create()
        conv["id"] = conv_id or conv["id"]
        conv.setdefault("messages", []).append(
            {"role": "user", "content": text, "images": images}
        )
        if conv.get("title") in {"", "New chat"} and text.strip():
            conv["title"] = _title_from_text(text)
        return self.save(conv)

    def set_assistant(
        self,
        conv_id: str,
        content: str,
        reasoning: str = "",
        stats: str = "",
    ) -> dict[str, Any]:
        """アシスタント応答（本文・推論・統計）を追加する。"""
        conv = self.get(conv_id)
        if conv is None:
            raise KeyError(conv_id)
        conv.setdefault("messages", []).append(
            {
                "role": "assistant",
                "content": content,
                "reasoning": reasoning,
                "stats": stats,
            }
        )
        return self.save(conv)

    def export_all(self) -> dict[str, Any]:
        """全会話をエクスポート用 dict にまとめる。"""
        conversations: list[dict[str, Any]] = []
        for row in self.list_summaries():
            item = self.get(row["id"])
            if item:
                conversations.append(item)
        return {"conversations": conversations}

    def import_all(self, payload: dict[str, Any] | list[Any]) -> int:
        """JSON から会話を取り込み、新規 ID で保存した件数を返す。"""
        if isinstance(payload, list):
            items = payload
        elif isinstance(payload, dict):
            items = payload.get("conversations") or payload.get("items") or []
        else:
            items = []
        count = 0
        for item in items:
            if not isinstance(item, dict):
                continue
            conv = {
                "id": uuid.uuid4().hex,
                "title": str(item.get("title") or "Imported chat"),
                "created_at": item.get("created_at") or _now(),
                "updated_at": item.get("updated_at") or _now(),
                "messages": _normalize_messages(item.get("messages") or []),
            }
            self.save(conv)
            count += 1
        return count


def _normalize_messages(raw: list[Any]) -> list[dict[str, Any]]:
    """インポートした messages を内部形式へ正規化する。"""
    out: list[dict[str, Any]] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        role = item.get("role")
        if role not in {"user", "assistant"}:
            continue
        content = item.get("content")
        text = content if isinstance(content, str) else ""
        images = item.get("images") if isinstance(item.get("images"), list) else []
        if isinstance(content, list):
            texts: list[str] = []
            for part in content:
                if isinstance(part, str):
                    texts.append(part)
                elif isinstance(part, dict):
                    if part.get("type") == "text":
                        texts.append(str(part.get("text") or ""))
                    elif part.get("type") == "image" and part.get("image"):
                        images.append(str(part["image"]))
            text = "\n".join(t for t in texts if t)
        msg: dict[str, Any] = {"role": role, "content": text, "images": [str(p) for p in images]}
        if role == "assistant":
            msg["reasoning"] = str(item.get("reasoning") or "")
            msg["stats"] = str(item.get("stats") or "")
        out.append(msg)
    return out


def to_chatbot_messages(conv: dict[str, Any] | None) -> list[dict[str, Any]]:
    """保存済み会話を Gradio Chatbot の messages 形式にする。"""
    if not conv:
        return []
    history: list[dict[str, Any]] = []
    for msg in conv.get("messages") or []:
        role = msg.get("role")
        if role == "user":
            history.append(_user_chatbot_message(msg))
        elif role == "assistant":
            reasoning = str(msg.get("reasoning") or "")
            if reasoning.strip():
                history.append(
                    {
                        "role": "assistant",
                        "content": reasoning,
                        "metadata": {"title": "Thinking"},
                    }
                )
            text = str(msg.get("content") or "")
            stats = str(msg.get("stats") or "")
            if stats:
                text = f"{text}\n\n*{stats}*" if text else f"*{stats}*"
            history.append({"role": "assistant", "content": text})
    return history


def _user_chatbot_message(msg: dict[str, Any]) -> dict[str, Any]:
    """ユーザー発言をテキスト＋画像つき Chatbot メッセージにする。"""
    text = str(msg.get("content") or "")
    images = [p for p in (msg.get("images") or []) if p]
    if not images:
        return {"role": "user", "content": text}
    content: list[Any] = []
    if text.strip():
        content.append({"type": "text", "text": text})
    for path in images:
        content.append({"path": path})
    return {"role": "user", "content": content}


def to_api_messages(
    conv: dict[str, Any] | None,
    system_prompt: str,
    max_side: int = 1024,
    encode_image: Callable[[str, int], str] | None = None,
) -> list[dict[str, Any]]:
    """会話を llama-server の chat completions 用 messages にする。"""
    encode = encode_image or image_to_data_url
    messages: list[dict[str, Any]] = []
    prompt = (system_prompt or "").strip()
    if prompt:
        messages.append({"role": "system", "content": prompt})
    if not conv:
        return messages
    for msg in conv.get("messages") or []:
        role = msg.get("role")
        if role == "user":
            text = str(msg.get("content") or "")
            images = [p for p in (msg.get("images") or []) if p and Path(str(p)).is_file()]
            if images:
                parts: list[dict[str, Any]] = []
                if text.strip():
                    parts.append({"type": "text", "text": text})
                elif not text.strip():
                    parts.append({"type": "text", "text": "Describe the image."})
                for path in images:
                    parts.append(
                        {
                            "type": "image_url",
                            "image_url": {"url": encode(str(path), max_side)},
                        }
                    )
                messages.append({"role": "user", "content": parts})
            else:
                messages.append({"role": "user", "content": text})
        elif role == "assistant":
            item: dict[str, Any] = {"role": "assistant", "content": str(msg.get("content") or "")}
            reasoning = str(msg.get("reasoning") or "")
            if reasoning:
                item["reasoning_content"] = reasoning
            messages.append(item)
    return messages


def streaming_chatbot(
    conv: dict[str, Any] | None,
    reasoning: str,
    content: str,
    stats: str = "",
    error: str = "",
) -> list[dict[str, Any]]:
    """保存済み履歴に、生成中の reasoning / 本文 / エラーを足して表示用にする。"""
    history = to_chatbot_messages(conv)
    if error:
        history.append({"role": "assistant", "content": f"**Error:** {error}"})
        return history
    if reasoning.strip():
        history.append(
            {
                "role": "assistant",
                "content": reasoning,
                "metadata": {"title": "Thinking"},
            }
        )
    text = content
    if stats:
        text = f"{text}\n\n*{stats}*" if text else f"*{stats}*"
    history.append({"role": "assistant", "content": text})
    return history
