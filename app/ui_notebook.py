"""Notebook タブ。"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Any

import gradio as gr

from app.llama_client import format_timings
from app.ui_context import AppContext, llama_client
from app.ui_form import parse_sampling


@dataclass
class NotebookTab:
    """Notebook タブの部品。"""

    text: Any
    generate_btn: Any
    stop_btn: Any
    undo_btn: Any
    stats: Any


def render_notebook() -> NotebookTab:
    """Notebook タブのレイアウトを組み立てる。"""
    with gr.Tab("Notebook"):
        gr.Markdown(
            "Raw completion via `/completion` (no chat template). "
            "The whole editor is the prompt; generated text is appended."
        )
        nb_text = gr.Textbox(
            lines=22,
            max_lines=40,
            show_label=False,
            placeholder="Write a prompt, then Generate.",
        )
        with gr.Row():
            nb_gen = gr.Button("Generate", variant="primary")
            nb_stop = gr.Button("Stop", variant="stop")
            nb_undo = gr.Button("Undo last generation")
        nb_stats = gr.Markdown("")
    return NotebookTab(
        text=nb_text,
        generate_btn=nb_gen,
        stop_btn=nb_stop,
        undo_btn=nb_undo,
        stats=nb_stats,
    )


def bind_notebook(
    ctx: AppContext,
    tab: NotebookTab,
    undo_state: Any,
    sampling_inputs: list[Any],
) -> None:
    """Notebook タブのイベントを接続する。"""

    def notebook_generate(text: str, undo: str, *sample_values: Any):
        """Notebook の全文を prompt として /completion する。"""
        sampling, _sample = parse_sampling(*sample_values)
        prompt = text or ""
        if not ctx.process.is_ready():
            yield prompt, prompt, "llama-server is not running. Click Start."
            return
        abort = threading.Event()
        acc = prompt
        stats = ""
        try:
            for delta in llama_client(ctx).completion_stream(prompt, sampling, abort=abort):
                if delta.error:
                    yield acc, prompt, delta.error
                    return
                if delta.content:
                    acc += delta.content
                if delta.timings:
                    stats = format_timings(delta.timings, None)
                yield acc, prompt, stats
                if delta.done:
                    break
        except GeneratorExit:
            raise
        finally:
            abort.set()
        yield acc, prompt, stats

    def notebook_undo(undo: str):
        """直近の completion を取り消して、生成前のテキストに戻す。"""
        return undo or "", ""

    nb_event = tab.generate_btn.click(
        notebook_generate,
        inputs=[tab.text, undo_state, *sampling_inputs],
        outputs=[tab.text, undo_state, tab.stats],
    )
    tab.stop_btn.click(fn=None, cancels=[nb_event])
    tab.undo_btn.click(notebook_undo, inputs=[undo_state], outputs=[tab.text, tab.stats])
