"""Gradio Blocks UI（Chat / Notebook / Models / Settings）。"""

from __future__ import annotations

from pathlib import Path

import gradio as gr

from app.ui_chat import bind_chat, render_chat
from app.ui_context import AppContext
from app.ui_form import FORM_KEYS
from app.ui_models import bind_models, render_models
from app.ui_notebook import bind_notebook, render_notebook
from app.ui_server import bind_server, render_header
from app.ui_settings import bind_settings, render_settings

CSS = """
.header-row { align-items: center; }
.app-title {
  overflow: hidden !important;
  flex-shrink: 0;
}
.app-title h1 {
  margin: 0;
  font-size: 1.35rem;
  font-weight: 650;
  letter-spacing: -0.02em;
  white-space: nowrap;
  line-height: 1.2;
  overflow: hidden;
}
.header-status textarea {
  font-size: 1.02rem;
  line-height: 1.2;
  background: transparent !important;
  border: none !important;
  box-shadow: none !important;
  resize: none !important;
  overflow: hidden !important;
  padding: 0 !important;
  min-height: 0 !important;
}
"""


def build_app(data_dir: Path) -> gr.Blocks:
    """Gradio Blocks を組み立てて返す。"""
    ctx = AppContext(Path(data_dir))

    with gr.Blocks(title="llamachat-webui", fill_height=True) as demo:
        conv_id_state = gr.State("")
        notebook_undo = gr.State("")
        header = render_header(ctx)
        with gr.Tabs():
            chat = render_chat(ctx)
            notebook = render_notebook()
            models = render_models(ctx)
            settings = render_settings(ctx)

        form_inputs = models.inputs() + settings.path_inputs() + settings.inputs()
        sampling_inputs = settings.inputs()
        if len(form_inputs) != len(FORM_KEYS):
            raise RuntimeError("FORM_KEYS とフォーム部品の並びが一致していません。")

        timer = gr.Timer(2.0)
        bind_server(ctx, header, models, form_inputs, timer, demo)
        bind_chat(ctx, chat, conv_id_state, sampling_inputs)
        bind_notebook(ctx, notebook, notebook_undo, sampling_inputs)
        bind_models(ctx, models, settings.models_dir, settings.extra_models_dirs)
        bind_settings(ctx, settings, models, form_inputs, chat.conv_dd, demo)

    return demo
