"""Models タブ。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import gradio as gr

from app.install_server import BACKEND_CHOICES, InstallError, install_llama_server
from app.models import (
    bundle_choices,
    bundle_dir_id,
    bundle_from_selection,
    catalog_id_of,
    download_hf_bundle,
    missing_model_roots,
    parse_model_roots,
)
from app.settings import CATALOG_KEYS
from app.ui_context import AppContext
from app.ui_form import form_value, setting_int


@dataclass
class ModelsTab:
    """Models タブの部品。"""

    llama_backend: Any
    llama_tag: Any
    install_btn: Any
    install_status: Any
    llama_server_path: Any
    host: Any
    port: Any
    model_path: Any
    hf_repo: Any
    mmproj_path: Any
    mtp_path: Any
    ctx_size: Any
    n_gpu_layers: Any
    extra_args: Any
    local_gguf: Any
    scan_btn: Any
    save_btn: Any
    scan_status: Any
    dl_id: Any
    dl_repo: Any
    dl_model: Any
    dl_vision: Any
    dl_mtp: Any
    dl_btn: Any
    dl_status: Any
    model_start: Any
    model_stop: Any
    server_log: Any

    def inputs(self) -> list[Any]:
        """settings.json のモデル側キーに対応する入力欄。"""
        return [
            self.llama_server_path,
            self.host,
            self.port,
            self.model_path,
            self.hf_repo,
            self.mmproj_path,
            self.mtp_path,
            self.ctx_size,
            self.n_gpu_layers,
            self.extra_args,
        ]


def render_models(ctx: AppContext) -> ModelsTab:
    """Models タブのレイアウトを組み立てる。"""
    s = ctx.settings.data
    with gr.Tab("Models"):
        with gr.Accordion("Install llama-server", open=True):
            gr.Markdown(
                "Download official [llama.cpp](https://github.com/ggml-org/llama.cpp/releases) "
                "binaries into `data/llama-server/`. CUDA packages are Windows-only; "
                "Linux GPU builds use Vulkan. Optional `GITHUB_TOKEN` avoids API rate limits."
            )
            with gr.Row():
                llama_backend = gr.Dropdown(
                    label="Backend",
                    choices=BACKEND_CHOICES,
                    value="auto",
                )
                llama_tag = gr.Textbox(
                    label="Release tag",
                    placeholder="latest (e.g. b10549)",
                    value="",
                )
            install_btn = gr.Button("Install / Update llama-server", variant="primary")
            install_status = gr.Markdown("")
        llama_server_path = gr.Textbox(
            label="llama-server path",
            value=form_value(s, "llama_server_path"),
            placeholder="Filled by Install, or set LLAMA_SERVER / PATH",
        )
        with gr.Row():
            host = gr.Textbox(label="Host", value=form_value(s, "host") or "127.0.0.1")
            port = gr.Number(
                label="Port",
                value=setting_int(s, "port", 8080),
                precision=0,
                step=1,
            )
        hf_repo = gr.Textbox(
            label="Hugging Face repo (-hf)",
            value=form_value(s, "hf_repo"),
            placeholder="ggml-org/Qwen2.5-3B-Instruct-GGUF:Q4_K_M",
        )
        model_path = gr.Textbox(
            label="Local GGUF path (-m)",
            value=form_value(s, "model_path"),
            placeholder=r"C:\models\model.gguf",
        )
        mmproj_path = gr.Textbox(
            label="mmproj path (optional)",
            value=form_value(s, "mmproj_path"),
        )
        mtp_path = gr.Textbox(
            label="MTP draft path (optional, --model-draft)",
            value=form_value(s, "mtp_path"),
        )
        with gr.Row():
            ctx_size = gr.Number(
                label="Context size (-c, 0 = model default)",
                value=setting_int(s, "ctx_size", 0),
                precision=0,
                step=1,
            )
            n_gpu_layers = gr.Textbox(
                label="GPU layers (-ngl)",
                value=form_value(s, "n_gpu_layers"),
            )
        extra_args = gr.Textbox(
            label="Extra llama-server args",
            value=form_value(s, "extra_args"),
            placeholder="--flash-attn on",
        )
        with gr.Row():
            local_gguf = gr.Dropdown(
                label="Model catalog (ID)",
                choices=bundle_choices(
                    s.get("models_dir") or str(ctx.data_dir / "models"),
                    s.get("extra_models_dirs") or "",
                    anchor=ctx.data_dir,
                ),
                value=s.get("model_path") or None,
                allow_custom_value=True,
            )
            scan_btn = gr.Button("更新", scale=0)
            save_btn = gr.Button("保存", variant="primary", scale=0)
        scan_status = gr.Markdown("")
        with gr.Accordion("Download from Hugging Face", open=True):
            dl_id = gr.Textbox(
                label="Model ID (directory name)",
                info="Empty uses the model filename without .gguf. Files go to models_dir/ID/",
            )
            dl_repo = gr.Textbox(label="Repository")
            dl_model = gr.Textbox(label="Model file")
            dl_vision = gr.Textbox(label="Vision file (mmproj, optional)")
            dl_mtp = gr.Textbox(label="MTP file (optional)")
            dl_btn = gr.Button("Download all specified files", variant="primary")
            dl_status = gr.Markdown("")
        with gr.Row():
            model_start = gr.Button("Start llama-server", variant="primary")
            model_stop = gr.Button("Stop", variant="stop")
        server_log = gr.Textbox(
            label="llama-server log",
            lines=14,
            max_lines=20,
            interactive=False,
            autoscroll=False,
            value=ctx.process.logs(),
        )
    return ModelsTab(
        llama_backend=llama_backend,
        llama_tag=llama_tag,
        install_btn=install_btn,
        install_status=install_status,
        llama_server_path=llama_server_path,
        host=host,
        port=port,
        model_path=model_path,
        hf_repo=hf_repo,
        mmproj_path=mmproj_path,
        mtp_path=mtp_path,
        ctx_size=ctx_size,
        n_gpu_layers=n_gpu_layers,
        extra_args=extra_args,
        local_gguf=local_gguf,
        scan_btn=scan_btn,
        save_btn=save_btn,
        scan_status=scan_status,
        dl_id=dl_id,
        dl_repo=dl_repo,
        dl_model=dl_model,
        dl_vision=dl_vision,
        dl_mtp=dl_mtp,
        dl_btn=dl_btn,
        dl_status=dl_status,
        model_start=model_start,
        model_stop=model_stop,
        server_log=server_log,
    )


def _catalog_fields(
    ctx: AppContext,
    selected: str,
    directory: str,
    extra_dirs: str,
) -> tuple[str, dict[str, Any]]:
    """選択中の ID について、保存済み Models 欄とディスク上のパスを合成する。"""
    dest = directory or ctx.settings.get("models_dir") or ""
    extra = extra_dirs if extra_dirs is not None else (ctx.settings.get("extra_models_dirs") or "")
    bundle = bundle_from_selection(selected, dest, extra, anchor=ctx.data_dir)
    catalog_id = catalog_id_of(selected, dest, extra, anchor=ctx.data_dir)
    preset = ctx.settings.get_catalog(catalog_id)
    fields = {
        "model_path": preset.get("model_path") or bundle.get("model") or "",
        "hf_repo": str(preset.get("hf_repo") or ""),
        "mmproj_path": preset["mmproj_path"] if "mmproj_path" in preset else (bundle.get("vision") or ""),
        "mtp_path": preset["mtp_path"] if "mtp_path" in preset else (bundle.get("mtp") or ""),
        "ctx_size": int(preset["ctx_size"]) if preset.get("ctx_size") not in (None, "") else 0,
        "n_gpu_layers": str(preset.get("n_gpu_layers") or "auto"),
        "extra_args": str(preset.get("extra_args") or ""),
    }
    if not preset:
        fields["ctx_size"] = 0
        fields["n_gpu_layers"] = "auto"
        fields["extra_args"] = ""
    return catalog_id, fields


def bind_models(
    ctx: AppContext,
    tab: ModelsTab,
    models_dir: Any,
    extra_models_dirs: Any,
) -> None:
    """Models タブのイベントを接続する。"""

    def on_scan(directory: str, extra_dirs: str):
        """モデル ID カタログを Dropdown に載せる。"""
        dest = directory or ctx.settings.get("models_dir") or ""
        extra = extra_dirs if extra_dirs is not None else (ctx.settings.get("extra_models_dirs") or "")
        ctx.settings.update({"models_dir": dest, "extra_models_dirs": extra or ""})
        roots = parse_model_roots(dest, extra, anchor=ctx.data_dir)
        missing = missing_model_roots(extra, anchor=ctx.data_dir)
        choices = bundle_choices(dest, extra, anchor=ctx.data_dir)
        current = str(ctx.settings.get("model_path") or "")
        ids = {item[1] for item in choices}
        value = current if current in ids else (choices[0][1] if choices else None)
        lines = [f"Found **{len(choices)}** model ID(s)."]
        if roots:
            lines.append("Using: " + ", ".join(f"`{path}`" for path in roots))
        if missing:
            lines.append(
                "Not found: "
                + ", ".join(f"`{item}`" for item in missing)
                + f" (resolved from data dir `{ctx.data_dir}` and its parents, not only the process cwd)."
            )
            gr.Warning("Additional model directory not found: " + "; ".join(missing))
        return gr.update(choices=choices, value=value), "\n\n".join(lines)

    def on_save_models(
        llama_path: str,
        host_v: str,
        port_v: Any,
        model_v: str,
        hf_v: str,
        mmproj_v: str,
        mtp_v: str,
        ctx_v: Any,
        ngl_v: str,
        extra_v: str,
        models_dir_v: str,
        extra_models_v: str,
        selected: str,
    ):
        """Models タブの入力を保存する。カタログ ID があれば ID ごとにも残す。"""
        ctx.settings.update(
            {
                "llama_server_path": llama_path or "",
                "host": host_v or "127.0.0.1",
                "port": int(port_v or 8080),
                "models_dir": models_dir_v or "",
                "extra_models_dirs": extra_models_v or "",
            }
        )
        catalog = {
            "model_path": model_v or "",
            "hf_repo": hf_v or "",
            "mmproj_path": mmproj_v or "",
            "mtp_path": mtp_v or "",
            "ctx_size": int(ctx_v or 0),
            "n_gpu_layers": str(ngl_v or "").strip() or "auto",
            "extra_args": extra_v or "",
        }
        catalog_id = catalog_id_of(
            selected or model_v or "",
            models_dir_v or "",
            extra_models_v or "",
            anchor=ctx.data_dir,
        )
        ctx.settings.upsert_catalog(catalog_id, catalog)
        if catalog_id:
            return f"Saved catalog **{catalog_id}**."
        return "Saved Models settings."

    def on_pick_gguf(selected: str, directory: str, extra_dirs: str):
        """カタログで選んだ ID のパスと、保存済み Models 欄を載せる。"""
        if not selected:
            skip = gr.skip()
            return skip, skip, skip, skip, skip, skip, skip
        _catalog_id, fields = _catalog_fields(ctx, selected, directory, extra_dirs)
        ctx.settings.update({key: fields[key] for key in CATALOG_KEYS})
        return (
            fields["model_path"],
            fields["hf_repo"],
            fields["mmproj_path"],
            fields["mtp_path"],
            fields["ctx_size"],
            fields["n_gpu_layers"],
            fields["extra_args"],
        )

    def on_download(
        repo: str,
        bundle_id: str,
        model_file: str,
        vision_file: str,
        mtp_file: str,
        directory: str,
        extra_dirs: str,
    ):
        """指定した本体 / Vision / MTP を ID ディレクトリへ保存し、パス欄へ反映する。"""
        dest = directory or ctx.settings.get("models_dir") or str(ctx.data_dir / "models")
        skip = gr.skip()
        message = ""
        paths: dict[str, str] = {}
        try:
            for message, paths in download_hf_bundle(
                repo,
                dest,
                model_file=model_file or "",
                vision_file=vision_file or "",
                mtp_file=mtp_file or "",
                bundle_id=bundle_id or "",
            ):
                yield (
                    message,
                    skip,
                    paths.get("model") or skip,
                    paths.get("vision") or skip,
                    paths.get("mtp") or skip,
                )
        except Exception as exc:
            yield f"**Download failed:** {exc}", skip, skip, skip, skip
            return
        choices = bundle_choices(dest, extra_dirs or "", anchor=ctx.data_dir)
        updates = {}
        if paths.get("model"):
            updates["model_path"] = paths["model"]
        if paths.get("vision"):
            updates["mmproj_path"] = paths["vision"]
        if paths.get("mtp"):
            updates["mtp_path"] = paths["mtp"]
        if updates:
            ctx.settings.update(updates)
        yield (
            message,
            gr.update(
                choices=choices,
                value=paths.get("model") or (choices[0][1] if choices else None),
            ),
            paths.get("model") or skip,
            paths.get("vision") or skip,
            paths.get("mtp") or skip,
        )

    def on_model_id_from_file(model_file: str):
        """Model file から ID（ディレクトリ名）を埋める。"""
        try:
            return bundle_dir_id(model_file=model_file or "")
        except ValueError:
            return gr.skip()

    def on_install(backend: str, tag: str):
        """公式リリースから llama-server を導入し、パスを保存する。"""
        binary = None
        try:
            for message, found in install_llama_server(
                ctx.data_dir,
                backend=backend or "auto",
                tag=tag or "",
            ):
                if found:
                    binary = found
                yield message, gr.skip()
        except InstallError as exc:
            yield f"**Install failed:** {exc}", gr.skip()
            return
        except Exception as exc:
            yield f"**Install failed:** {exc}", gr.skip()
            return
        if binary:
            ctx.settings.update({"llama_server_path": binary})
            yield f"Installed and saved path:\n`{binary}`", binary
        else:
            yield "**Install failed:** binary path missing.", gr.skip()

    tab.scan_btn.click(
        on_scan,
        inputs=[models_dir, extra_models_dirs],
        outputs=[tab.local_gguf, tab.scan_status],
    )
    tab.save_btn.click(
        on_save_models,
        inputs=[
            tab.llama_server_path,
            tab.host,
            tab.port,
            tab.model_path,
            tab.hf_repo,
            tab.mmproj_path,
            tab.mtp_path,
            tab.ctx_size,
            tab.n_gpu_layers,
            tab.extra_args,
            models_dir,
            extra_models_dirs,
            tab.local_gguf,
        ],
        outputs=[tab.scan_status],
    )
    tab.local_gguf.change(
        on_pick_gguf,
        inputs=[tab.local_gguf, models_dir, extra_models_dirs],
        outputs=[
            tab.model_path,
            tab.hf_repo,
            tab.mmproj_path,
            tab.mtp_path,
            tab.ctx_size,
            tab.n_gpu_layers,
            tab.extra_args,
        ],
    )
    tab.dl_model.change(on_model_id_from_file, inputs=[tab.dl_model], outputs=[tab.dl_id])
    tab.dl_btn.click(
        on_download,
        inputs=[
            tab.dl_repo,
            tab.dl_id,
            tab.dl_model,
            tab.dl_vision,
            tab.dl_mtp,
            models_dir,
            extra_models_dirs,
        ],
        outputs=[tab.dl_status, tab.local_gguf, tab.model_path, tab.mmproj_path, tab.mtp_path],
    )
    tab.install_btn.click(
        on_install,
        inputs=[tab.llama_backend, tab.llama_tag],
        outputs=[tab.install_status, tab.llama_server_path],
    )
