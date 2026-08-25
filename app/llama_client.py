"""llama-server 向け HTTP クライアント（chat / completion）。"""

from __future__ import annotations

import json
import threading
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

import httpx


@dataclass
class ChatDelta:
    """チャットストリーム 1 チャンク分の差分。"""
    content: str = ""
    reasoning: str = ""
    timings: dict[str, Any] | None = None
    usage: dict[str, Any] | None = None
    error: str | None = None
    done: bool = False


@dataclass
class CompletionDelta:
    """Notebook 用 /completion ストリーム 1 チャンク分の差分。"""
    content: str = ""
    timings: dict[str, Any] | None = None
    error: str | None = None
    done: bool = False


def format_timings(timings: dict[str, Any] | None, usage: dict[str, Any] | None = None) -> str:
    """timings / usage から tok/s などの短い統計文字列を作る。"""
    if not timings and not usage:
        return ""
    parts: list[str] = []
    predicted = None
    prompt = None
    pred_speed = None
    prompt_speed = None
    if timings:
        predicted = timings.get("predicted_n")
        prompt = timings.get("prompt_n")
        pred_speed = timings.get("predicted_per_second")
        prompt_speed = timings.get("prompt_per_second")
    if usage:
        predicted = predicted or usage.get("completion_tokens")
        prompt = prompt or usage.get("prompt_tokens")
    if predicted is not None:
        parts.append(f"{int(predicted)} tokens")
    if pred_speed:
        parts.append(f"{float(pred_speed):.1f} tok/s")
    if prompt is not None:
        extra = f"{int(prompt)} prompt"
        if prompt_speed:
            extra += f" · {float(prompt_speed):.1f} tok/s"
        parts.append(extra)
    return " · ".join(parts)


class LlamaClient:
    """llama-server の OpenAI 互換 API と native completion を呼ぶ。"""

    def __init__(self, base_url: str, timeout: float | None = None) -> None:
        """接続先 URL を保持する。"""
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout if timeout is not None else httpx.Timeout(None, connect=5.0)

    def _client(self, timeout: float | httpx.Timeout | None = None) -> httpx.Client:
        """httpx クライアントを作る。"""
        return httpx.Client(base_url=self.base_url, timeout=timeout or self.timeout)

    def health(self) -> bool:
        """GET /health が ready なら True。"""
        try:
            with self._client(timeout=2.0) as client:
                response = client.get("/health")
                if response.status_code != 200:
                    return False
                try:
                    data = response.json()
                except json.JSONDecodeError:
                    return True
                if not isinstance(data, dict):
                    return True
                if "status" in data:
                    return str(data.get("status") or "").lower() in {"ok", "healthy"}
                return "error" not in data
        except httpx.HTTPError:
            return False

    def props(self) -> dict[str, Any]:
        """GET /props でサーバ情報を取得する。"""
        try:
            with self._client(timeout=5.0) as client:
                response = client.get("/props")
                response.raise_for_status()
                data = response.json()
                return data if isinstance(data, dict) else {}
        except httpx.HTTPError:
            return {}

    def models(self) -> list[str]:
        """GET /v1/models からモデル ID 一覧を返す。"""
        try:
            with self._client(timeout=5.0) as client:
                response = client.get("/v1/models")
                response.raise_for_status()
                data = response.json()
                items = data.get("data") if isinstance(data, dict) else None
                if not isinstance(items, list):
                    return []
                names: list[str] = []
                for item in items:
                    if isinstance(item, dict) and item.get("id"):
                        names.append(str(item["id"]))
                return names
        except httpx.HTTPError:
            return []

    def chat_stream(
        self,
        messages: list[dict[str, Any]],
        sampling: dict[str, Any] | None = None,
        abort: threading.Event | None = None,
    ) -> Iterator[ChatDelta]:
        """POST /v1/chat/completions をストリーム受信する。"""
        payload: dict[str, Any] = {
            "messages": messages,
            "stream": True,
            "cache_prompt": True,
        }
        if sampling:
            payload.update(sampling)
        yield from self._stream_chat(payload, abort)

    def completion_stream(
        self,
        prompt: str,
        sampling: dict[str, Any] | None = None,
        abort: threading.Event | None = None,
    ) -> Iterator[CompletionDelta]:
        """POST /completion をストリーム受信する（Notebook 用）。"""
        payload: dict[str, Any] = {
            "prompt": prompt,
            "stream": True,
            "cache_prompt": True,
        }
        if sampling:
            mapped = dict(sampling)
            if "max_tokens" in mapped:
                mapped["n_predict"] = mapped.pop("max_tokens")
            payload.update(mapped)
        try:
            with self._client() as client:
                with client.stream("POST", "/completion", json=payload) as response:
                    if response.status_code >= 400:
                        body = response.read().decode("utf-8", errors="replace")
                        yield CompletionDelta(error=f"HTTP {response.status_code}: {body[:500]}", done=True)
                        return
                    for event in _iter_sse(response, abort):
                        if abort is not None and abort.is_set():
                            yield CompletionDelta(done=True)
                            return
                        if event is None:
                            continue
                        if isinstance(event, dict) and event.get("error"):
                            err = event["error"]
                            text = err if isinstance(err, str) else json.dumps(err)
                            yield CompletionDelta(error=text, done=True)
                            return
                        content = ""
                        if isinstance(event, dict):
                            content = event.get("content") or ""
                            timings = event.get("timings")
                            done = bool(event.get("stop"))
                        else:
                            timings = None
                            done = False
                        yield CompletionDelta(
                            content=str(content),
                            timings=timings if isinstance(timings, dict) else None,
                            done=done,
                        )
                        if done:
                            return
        except httpx.HTTPError as exc:
            yield CompletionDelta(error=str(exc), done=True)

    def _stream_chat(
        self,
        payload: dict[str, Any],
        abort: threading.Event | None,
    ) -> Iterator[ChatDelta]:
        """chat completions の SSE を ChatDelta に分解する。"""
        try:
            with self._client() as client:
                with client.stream("POST", "/v1/chat/completions", json=payload) as response:
                    if response.status_code >= 400:
                        body = response.read().decode("utf-8", errors="replace")
                        yield ChatDelta(error=f"HTTP {response.status_code}: {body[:500]}", done=True)
                        return
                    last_timings: dict[str, Any] | None = None
                    last_usage: dict[str, Any] | None = None
                    for event in _iter_sse(response, abort):
                        if abort is not None and abort.is_set():
                            yield ChatDelta(timings=last_timings, usage=last_usage, done=True)
                            return
                        if event is None:
                            continue
                        if isinstance(event, dict) and event.get("error"):
                            err = event["error"]
                            text = err if isinstance(err, str) else json.dumps(err)
                            yield ChatDelta(error=text, done=True)
                            return
                        delta = ChatDelta()
                        if isinstance(event, dict):
                            if isinstance(event.get("timings"), dict):
                                last_timings = event["timings"]
                                delta.timings = last_timings
                            if isinstance(event.get("usage"), dict):
                                last_usage = event["usage"]
                                delta.usage = last_usage
                            choices = event.get("choices") or []
                            if choices and isinstance(choices[0], dict):
                                piece = choices[0].get("delta") or choices[0].get("message") or {}
                                if isinstance(piece, dict):
                                    delta.content = piece.get("content") or ""
                                    delta.reasoning = (
                                        piece.get("reasoning_content")
                                        or piece.get("reasoning")
                                        or ""
                                    )
                        yield delta
                    yield ChatDelta(timings=last_timings, usage=last_usage, done=True)
        except httpx.HTTPError as exc:
            yield ChatDelta(error=str(exc), done=True)


def _iter_sse(response: httpx.Response, abort: threading.Event | None) -> Iterator[dict[str, Any] | None]:
    """SSE の data: 行を JSON dict として yield する。"""
    for line in response.iter_lines():
        if abort is not None and abort.is_set():
            try:
                response.close()
            except Exception:
                pass
            return
        if not line:
            continue
        if line.startswith(":"):
            continue
        if not line.startswith("data:"):
            continue
        data = line[5:].strip()
        if not data or data == "[DONE]":
            if data == "[DONE]":
                return
            continue
        try:
            parsed = json.loads(data)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            yield parsed
        else:
            yield None
