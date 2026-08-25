"""Settings タブ。"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import gradio as gr

from app.models import bundle_choices
from app.ui_chat import dropdown_update
from app.ui_context import AppContext
from app.ui_form import FORM_KEYS, STEP_ANY, collect_form, form_value
from app.ui_models import ModelsTab


@dataclass
class SettingsTab:
    """Settings タブの部品。"""

    settings_file_md: Any
    models_dir: Any
    extra_models_dirs: Any
    system_prompt: Any
    temperature: Any
    top_k: Any
    top_p: Any
    min_p: Any
    repeat_penalty: Any
    presence_penalty: Any
    frequency_penalty: Any
    max_tokens: Any
    seed: Any
    image_max_side: Any
    save_btn: Any
    settings_status: Any
    export_btn: Any
    import_file: Any
    export_file: Any
    import_status: Any

    def path_inputs(self) -> list[Any]:
        """settings.json のモデル検索パスに対応する入力欄。"""
        return [self.models_dir, self.extra_models_dirs]

    def inputs(self) -> list[Any]:
        """settings.json のサンプリング側キーに対応する入力欄。"""
        return [
            self.system_prompt,
            self.temperature,
            self.top_k,
            self.top_p,
            self.min_p,
            self.repeat_penalty,
            self.presence_penalty,
            self.frequency_penalty,
            self.max_tokens,
            self.seed,
            self.image_max_side,
        ]


def render_settings(ctx: AppContext) -> SettingsTab:
    """Settings タブのレイアウトを組み立てる。"""
    s = ctx.settings.data
    with gr.Tab("Settings"):
        settings_file_md = gr.Markdown(f"Loaded from `{ctx.settings.path}`")
        models_dir = gr.Textbox(
            label="Models directory",
            value=form_value(s, "models_dir"),
        )
        extra_models_dirs = gr.Textbox(
            label="Additional model directories",
            value=form_value(s, "extra_models_dirs"),
            lines=3,
            placeholder="./storage/llm/",
            info="Optional. One path per line. Relative paths are from the workspace "
            "(./storage/llm), not from the git clone. Each folder should contain ID directories.",
        )
        system_prompt = gr.Textbox(
            label="System prompt",
            lines=5,
            value=s.get("system_prompt") or "",
            placeholder="Optional. Empty uses the model's default behavior.",
        )
        gr.Markdown(
            "Sampling defaults match **llama-server** "
            "(temp 0.8, top_k 40, top_p 0.95, min_p 0.05, repeat 1.0). "
            "Max tokens / seed empty means unlimited / random."
        )
        with gr.Row():
            temperature = gr.Number(
                label="Temperature",
                value=form_value(s, "temperature"),
                step=STEP_ANY,
            )
            top_k = gr.Number(
                label="Top K",
                value=form_value(s, "top_k"),
                precision=0,
                step=1,
            )
            top_p = gr.Number(
                label="Top P",
                value=form_value(s, "top_p"),
                step=STEP_ANY,
            )
            min_p = gr.Number(
                label="Min P",
                value=form_value(s, "min_p"),
                step=STEP_ANY,
            )
        with gr.Row():
            repeat_penalty = gr.Number(
                label="Repeat penalty",
                value=form_value(s, "repeat_penalty"),
                step=STEP_ANY,
                info="llama-server default is 1.0 (no extra penalty). Must be greater than 0.",
            )
            presence_penalty = gr.Number(
                label="Presence penalty",
                value=form_value(s, "presence_penalty"),
                step=STEP_ANY,
            )
            frequency_penalty = gr.Number(
                label="Frequency penalty",
                value=form_value(s, "frequency_penalty"),
                step=STEP_ANY,
            )
            # Number は空欄を 0 にするので、unlimited / random は Textbox。
            max_tokens = gr.Textbox(
                label="Max tokens",
                value=form_value(s, "max_tokens"),
                placeholder="unlimited",
            )
        seed = gr.Textbox(
            label="Seed",
            value=form_value(s, "seed"),
            placeholder="random",
        )
        image_max_side = gr.Number(
            label="Vision image max side (px)",
            value=form_value(s, "image_max_side"),
            precision=0,
            step=1,
        )
        save_btn = gr.Button("Save settings", variant="primary")
        settings_status = gr.Markdown("")
        gr.Markdown("### Conversations")
        with gr.Row():
            export_btn = gr.Button("Export JSON")
            import_file = gr.File(label="Import JSON", file_types=[".json"])
        export_file = gr.File(label="Exported file")
        import_status = gr.Markdown("")
    return SettingsTab(
        settings_file_md=settings_file_md,
        models_dir=models_dir,
        extra_models_dirs=extra_models_dirs,
        system_prompt=system_prompt,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        min_p=min_p,
        repeat_penalty=repeat_penalty,
        presence_penalty=presence_penalty,
        frequency_penalty=frequency_penalty,
        max_tokens=max_tokens,
        seed=seed,
        image_max_side=image_max_side,
        save_btn=save_btn,
        settings_status=settings_status,
        export_btn=export_btn,
        import_file=import_file,
        export_file=export_file,
        import_status=import_status,
    )


def load_settings_into_ui(ctx: AppContext):
    """settings.json を読み直し、フォームへ載せる。"""
    ctx.settings.load()
    data = ctx.settings.data
    dest = data.get("models_dir") or str(ctx.data_dir / "models")
    extra = data.get("extra_models_dirs") or ""
    choices = bundle_choices(dest, extra, anchor=ctx.data_dir)
    current = str(data.get("model_path") or "")
    ids = {item[1] for item in choices}
    catalog = current if current in ids else (choices[0][1] if choices else None)
    return (
        *[form_value(data, key) for key in FORM_KEYS],
        f"Loaded from `{ctx.settings.path}`",
        gr.update(choices=choices, value=catalog),
    )


def bind_settings(
    ctx: AppContext,
    tab: SettingsTab,
    models: ModelsTab,
    form_inputs: list[Any],
    conv_dd: Any,
    demo: gr.Blocks,
) -> None:
    """Settings タブのイベントを接続する。"""

    def on_save(*values: Any):
        """Settings タブの内容を settings.json に保存する。"""
        ctx.settings.update(collect_form(*values))
        return "Settings saved."

    def on_export():
        """全会話を JSON ファイルとして書き出す。"""
        payload = ctx.store.export_all()
        out = ctx.data_dir / "conversations-export.json"
        out.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        return str(out)

    def on_import(file_obj):
        """会話 JSON を取り込み、一覧を更新する。"""
        if file_obj is None:
            return "Choose a JSON file.", dropdown_update(ctx.store, None)
        path = file_obj if isinstance(file_obj, str) else getattr(file_obj, "name", None)
        if not path:
            return "Could not read file.", dropdown_update(ctx.store, None)
        try:
            payload = json.loads(Path(path).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            return f"Invalid JSON: {exc}", dropdown_update(ctx.store, None)
        count = ctx.store.import_all(payload)
        return f"Imported {count} conversation(s).", dropdown_update(ctx.store, None)

    def on_load():
        return load_settings_into_ui(ctx)

    tab.save_btn.click(on_save, inputs=form_inputs, outputs=[tab.settings_status])
    tab.export_btn.click(on_export, outputs=[tab.export_file])
    tab.import_file.upload(on_import, inputs=[tab.import_file], outputs=[tab.import_status, conv_dd])
    demo.load(
        on_load,
        outputs=[
            *form_inputs,
            tab.settings_file_md,
            models.local_gguf,
        ],
    )
