"""Chat タブ。"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Any

import gradio as gr

from app.conversations import (
    ConversationStore,
    streaming_chatbot,
    to_api_messages,
    to_chatbot_messages,
)
from app.images import persist_images
from app.llama_client import format_timings
from app.ui_context import AppContext, llama_client
from app.ui_form import parse_sampling


@dataclass
class ChatTab:
    """Chat タブの部品。"""

    conv_dd: Any
    new_btn: Any
    del_btn: Any
    chatbot: Any
    chat_stats: Any
    chat_input: Any
    chat_stop: Any


def dropdown_update(store: ConversationStore, selected: str | None) -> dict[str, Any]:
    """会話 Dropdown の choices と選択値を更新する。"""
    choices = store.choices()
    ids = {item[1] for item in choices}
    value = selected if selected in ids else (choices[0][1] if choices else None)
    return gr.update(choices=choices, value=value)


def empty_multimodal() -> dict[str, Any]:
    """MultimodalTextbox を空にする値。"""
    return {"text": "", "files": []}


def render_chat(ctx: AppContext) -> ChatTab:
    """Chat タブのレイアウトを組み立てる。"""
    with gr.Tab("Chat"):
        with gr.Row(equal_height=False):
            with gr.Column(scale=1, min_width=240):
                conv_dd = gr.Dropdown(
                    label="Conversations",
                    choices=ctx.store.choices(),
                    value=None,
                    interactive=True,
                )
                with gr.Row():
                    new_btn = gr.Button("New chat")
                    del_btn = gr.Button("Delete")
            with gr.Column(scale=4):
                chatbot = gr.Chatbot(
                    value=[],
                    height=540,
                    placeholder="Start a conversation",
                    latex_delimiters=[
                        {"left": "$$", "right": "$$", "display": True},
                        {"left": "$", "right": "$", "display": False},
                    ],
                    buttons=["copy", "copy_all"],
                    reasoning_tags=[("<think>", "</think>")],
                )
                chat_stats = gr.Markdown("")
                chat_input = gr.MultimodalTextbox(
                    file_count="multiple",
                    file_types=["image"],
                    sources=["upload"],
                    placeholder="Message — images optional. Enter to send.",
                    show_label=False,
                    submit_btn=True,
                    max_plain_text_length=4096,
                    lines=5,
                    max_lines=5,
                )
                chat_stop = gr.Button("Stop generation", variant="stop")
    return ChatTab(
        conv_dd=conv_dd,
        new_btn=new_btn,
        del_btn=del_btn,
        chatbot=chatbot,
        chat_stats=chat_stats,
        chat_input=chat_input,
        chat_stop=chat_stop,
    )


def bind_chat(
    ctx: AppContext,
    tab: ChatTab,
    conv_id_state: Any,
    sampling_inputs: list[Any],
) -> None:
    """Chat タブのイベントを接続する。"""

    def on_new_chat():
        """空の会話を作り、Chatbot をクリアする。"""
        conv = ctx.store.create()
        return (
            conv["id"],
            [],
            "",
            dropdown_update(ctx.store, conv["id"]),
        )

    def on_delete(conv_id: str):
        """選択中の会話を削除し、別の会話へ切り替える。"""
        if conv_id:
            ctx.store.delete(conv_id)
        remaining = ctx.store.choices()
        if remaining:
            new_id = remaining[0][1]
            conv = ctx.store.get(new_id)
            return new_id, to_chatbot_messages(conv), "", dropdown_update(ctx.store, new_id)
        conv = ctx.store.create()
        return conv["id"], [], "", dropdown_update(ctx.store, conv["id"])

    def on_select_conv(evt: gr.SelectData):
        """Dropdown で選んだ会話を Chatbot に読み込む。"""
        selected = str(evt.value) if evt is not None and evt.value is not None else ""
        conv = ctx.store.get(selected)
        return selected, to_chatbot_messages(conv), ""

    def add_user(message: dict[str, Any] | None, conv_id: str):
        """ユーザー入力を会話へ追加し、入力欄を空にする。"""
        message = message or {}
        text = str(message.get("text") or "")
        files = message.get("files") or []
        if not text.strip() and not files:
            raise gr.Error("Enter a message or attach an image.")
        created = False
        if not conv_id or ctx.store.get(conv_id) is None:
            conv = ctx.store.create()
            conv_id = conv["id"]
            created = True
        images = persist_images(files, ctx.uploads / conv_id)
        conv = ctx.store.append_user(conv_id, text.strip(), images)
        dd = dropdown_update(ctx.store, conv_id) if created else gr.skip()
        return (
            empty_multimodal(),
            to_chatbot_messages(conv),
            conv_id,
            dd,
            "",
        )

    def stream_bot(history, conv_id: str, *sample_values: Any):
        """llama-server へ chat を投げ、応答をストリーム表示する。"""
        sampling, sample = parse_sampling(*sample_values)
        conv = ctx.store.get(conv_id)
        if conv is None:
            yield history or [], "No conversation.", gr.skip()
            return
        if not ctx.process.is_ready():
            yield (
                streaming_chatbot(
                    conv, "", "", error="llama-server is not running. Click Start."
                ),
                "",
                gr.skip(),
            )
            return

        max_side = int(sample["image_max_side"] or 1024)
        api_messages = to_api_messages(
            conv, str(sample["system_prompt"] or ""), max_side=max_side
        )
        abort = threading.Event()
        reasoning = ""
        content = ""
        stats = ""
        try:
            for delta in llama_client(ctx).chat_stream(api_messages, sampling, abort=abort):
                if delta.error:
                    yield streaming_chatbot(conv, reasoning, content, error=delta.error), delta.error, gr.skip()
                    return
                if delta.reasoning:
                    reasoning += delta.reasoning
                if delta.content:
                    content += delta.content
                if delta.timings or delta.usage:
                    stats = format_timings(delta.timings, delta.usage)
                if delta.reasoning or delta.content or delta.done:
                    yield streaming_chatbot(conv, reasoning, content, stats=stats), stats, gr.skip()
                if delta.done and not delta.content and not delta.reasoning:
                    break
            if content or reasoning:
                ctx.store.set_assistant(conv_id, content, reasoning, stats)
            yield to_chatbot_messages(ctx.store.get(conv_id)), stats, dropdown_update(ctx.store, conv_id)
        except GeneratorExit:
            if content or reasoning:
                ctx.store.set_assistant(conv_id, content, reasoning, stats)
            raise
        finally:
            abort.set()

    tab.new_btn.click(
        on_new_chat,
        outputs=[conv_id_state, tab.chatbot, tab.chat_stats, tab.conv_dd],
    )
    tab.del_btn.click(
        on_delete,
        inputs=[conv_id_state],
        outputs=[conv_id_state, tab.chatbot, tab.chat_stats, tab.conv_dd],
    )
    tab.conv_dd.select(
        on_select_conv,
        outputs=[conv_id_state, tab.chatbot, tab.chat_stats],
    )
    user_event = tab.chat_input.submit(
        add_user,
        inputs=[tab.chat_input, conv_id_state],
        outputs=[tab.chat_input, tab.chatbot, conv_id_state, tab.conv_dd, tab.chat_stats],
    )
    bot_event = user_event.then(
        stream_bot,
        inputs=[tab.chatbot, conv_id_state, *sampling_inputs],
        outputs=[tab.chatbot, tab.chat_stats, tab.conv_dd],
    )
    tab.chat_stop.click(fn=None, cancels=[bot_event])
