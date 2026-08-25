"""ヘッダの起動・停止と、llama-server 状態の更新。"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import gradio as gr

from app.ui_context import AppContext
from app.ui_form import collect_form


@dataclass
class HeaderBar:
    """画面上部のタイトル・状態・起動停止。"""

    status: Any
    start_btn: Any
    stop_btn: Any


def render_header(ctx: AppContext) -> HeaderBar:
    """タイトル行を組み立てる。"""
    with gr.Row(elem_classes=["header-row"]):
        gr.Markdown("# llamachat-webui", elem_classes=["app-title"])
        status = gr.Textbox(
            value=ctx.process.status_line(),
            show_label=False,
            lines=1,
            max_lines=1,
            interactive=False,
            container=False,
            autoscroll=False,
            elem_classes=["header-status"],
        )
        start_btn = gr.Button("Start", variant="primary", scale=0, min_width=110)
        stop_btn = gr.Button("Stop", variant="stop", scale=0, min_width=110)
    return HeaderBar(status=status, start_btn=start_btn, stop_btn=stop_btn)


def _status_and_logs(ctx: AppContext) -> tuple[str, str]:
    """ヘッダ状態とログ文字列を返す。"""
    return ctx.process.status_line(), ctx.process.logs()


def start_server(ctx: AppContext, *values: Any):
    """設定を保存し、llama-server を起動して ready まで待つ。"""
    data = collect_form(*values)
    ctx.settings.update(data)
    payload = dict(ctx.settings.data)
    payload["llama_install_dir"] = str(ctx.data_dir / "llama-server")
    err = ctx.process.spawn(payload)
    yield ctx.process.status_line(), ctx.process.logs() or (err or "Starting…")
    if err:
        return
    deadline = time.time() + 300
    while time.time() < deadline:
        state = ctx.process.poll_ready()
        yield ctx.process.status_line(), ctx.process.logs()
        if state in {"ready", "error"}:
            return
        time.sleep(0.4)
    ctx.process.status = "error"
    ctx.process.error = "Timed out waiting for /health (300s)"
    yield ctx.process.status_line(), ctx.process.logs()


def stop_server(ctx: AppContext):
    """llama-server を停止する。"""
    ctx.process.stop()
    return _status_and_logs(ctx)


def tick_status(ctx: AppContext, cache: dict[str, str]) -> tuple[Any, Any]:
    """定期的にサーバ状態とログを更新する。変化がなければ gr.skip()。"""
    status, logs = _status_and_logs(ctx)
    if status != cache.get("status"):
        cache["status"] = status
        status_out: Any = status
    else:
        status_out = gr.skip()
    if logs != cache.get("logs"):
        cache["logs"] = logs
        logs_out: Any = logs
    else:
        logs_out = gr.skip()
    return status_out, logs_out


def bind_server(
    ctx: AppContext,
    header: HeaderBar,
    models: Any,
    form_inputs: list[Any],
    timer: Any,
    demo: gr.Blocks,
) -> None:
    """起動・停止・状態更新のイベントを接続する。"""
    status_log = [header.status, models.server_log]
    tick_cache: dict[str, str] = {}

    def on_start(*values: Any):
        for status, logs in start_server(ctx, *values):
            tick_cache["status"] = status
            tick_cache["logs"] = logs if isinstance(logs, str) else str(logs)
            yield status, logs

    def on_stop():
        status, logs = stop_server(ctx)
        tick_cache["status"] = status
        tick_cache["logs"] = logs
        return status, logs

    def on_tick():
        return tick_status(ctx, tick_cache)

    start_event = header.start_btn.click(on_start, inputs=form_inputs, outputs=status_log)
    models.model_start.click(on_start, inputs=form_inputs, outputs=status_log)
    header.stop_btn.click(on_stop, outputs=status_log, cancels=[start_event])
    models.model_stop.click(on_stop, outputs=status_log)
    timer.tick(on_tick, outputs=status_log)
    demo.load(on_tick, outputs=status_log)
