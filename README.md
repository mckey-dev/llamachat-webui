# llamachat-webui

[llama.cpp](https://github.com/ggml-org/llama.cpp) の `llama-server` 向け Gradio フロントエンドです。  
`llama-server` を子プロセスとして起動・停止し、HTTP（`/v1/chat/completions`、`/completion`）だけで会話します。公式 WebUI に近い操作感を目指しています。

MCP / ツール呼び出し / ルーターモードは対象外です。

- リポジトリ: https://github.com/mckey-dev/llamachat-webui
- **更新日:** 2026-08-25

## 必要環境

- Python **3.11 以上**
- GGUF モデル（ローカルファイル、または Hugging Face リポジトリ）
- 初回起動時のインターネット（`venv` 作成時のパッケージ取得、および `llama-server` 未導入の場合）。導入済みなら不要

## クイックスタート

### Windows

[`webui.bat`](webui.bat) をダブルクリックするか、次を実行します。

```bat
webui.bat
```

### Linux / macOS

```bash
chmod +x webui.sh
./webui.sh
```

**初回のみ:** `venv` を作成し、`requirements.txt` を `pip install` します。`llama-server` が未導入なら GitHub Releases からも取得します。

**2 回目以降:** 既存の `venv` を使ってすぐ起動します。`pip install` は再実行しません。依存パッケージを更新するときは `venv` を削除して起動し直すか、次を手動実行します。

```bat
venv\Scripts\python.exe -m pip install -r requirements.txt
```

```bash
venv/bin/python -m pip install -r requirements.txt
```

ブラウザで http://127.0.0.1:7860 を開きます。

起動前にバックエンドを指定する例:

```bat
set LLAMA_BACKEND=cuda
webui.bat
```

```bash
export LLAMA_BACKEND=cuda
./webui.sh
```

### CLI

`venv` を有効化したうえで:

```bash
python launch.py --server-name 127.0.0.1 --server-port 7860
```

| 引数 | 説明 |
|------|------|
| `--server-name` | Gradio の待受アドレス（既定 `127.0.0.1`） |
| `--server-port` | Gradio のポート（既定 `7860`） |
| `--share` | Gradio の公開 URL を発行する |
| `--data-dir` | 設定・会話・アップロードの保存先（既定 `data/`） |
| `--install-server` | 既存バイナリがあっても `llama-server` を再取得する |
| `--skip-install` | ダウンロードしない（既存バイナリのみ） |
| `--llama-backend` | `auto` / `cpu` / `cuda` / `cuda-12.4` / `cuda-13.3` / `vulkan` / `rocm`（環境変数 `LLAMA_BACKEND` でも可） |
| `--llama-tag` | llama.cpp のリリースタグ（例: `b10618`。未指定はバイナリ付きの最新 `bXXXX`。環境変数 `LLAMA_TAG` でも可） |

## llama-server の導入

公式バイナリは [llama.cpp の Releases](https://github.com/ggml-org/llama.cpp/releases) から取得します。

**自動:** `webui.bat` / `webui.sh` / `python launch.py` は、バイナリが見つからないときだけ導入します。導入済みなら上書きしません。強制更新は `--install-server`、スキップは `--skip-install` または `LLAMA_SKIP_INSTALL=1` です。

**既定タグ:** `--llama-tag` / `LLAMA_TAG` 未指定時は、GitHub の `/releases/latest` ではなく **バイナリ資産のある最新リリース**（通常は pre-release の `bXXXX`）を選びます。semver タグ（例: `v0.2.0`）は資産がほぼ無いことがあり、そのままでは導入に失敗するためです。

**UI:** Models タブ → **Install llama-server** → バックエンドを選ぶ → **Install / Update**。パスは設定に保存されます。

**CLI:**

```bash
python -m app.install_server --backend auto
python launch.py --install-server --llama-backend cuda
```

GPU 無し（CPU のみ）の例:

```bat
set LLAMA_BACKEND=cpu
webui.bat
```

| バックエンド | 内容 |
|--------------|------|
| `auto` | NVIDIA あり → Windows は CUDA、Linux は Vulkan。それ以外は CPU。macOS は Metal ビルド |
| `cpu` | CPU のみ |
| `cuda` / `cuda-12.4` / `cuda-13.3` | 公式リリースは **Windows のみ**。`cudart` も取得する |
| `vulkan` | Windows / Linux の GPU |
| `rocm` | そのリリースに ROCm/HIP 資産がある場合 |

保存先は `data/llama-server/<tag>-<backend>/` です。GitHub API のレート制限に当たったら `GITHUB_TOKEN` を設定してください。

自前ビルドのバイナリを使う場合は、UI の **llama-server path** を指定するか、環境変数 `LLAMA_SERVER` を設定します。

公式リリースに Linux 向け CUDA バイナリはありません。Linux で CUDA を使う場合は自分でビルドするか、Colab / Paperspace 用ノート（後述）を使ってください。

## モデルの使い方

次のいずれかが必要です。

- ローカル GGUF パス（`-m`）
- Hugging Face リポジトリ（`-hf`、例: `ggml-org/Qwen2.5-3B-Instruct-GGUF:Q4_K_M`）

任意: Vision 用 mmproj、MTP ドラフト（`--model-draft`）、コンテキスト長、GPU レイヤ数（`auto` / `all` / 数値）、追加引数。

Models タブでは、同一 Hugging Face リポジトリから本体 GGUF・Vision mmproj・MTP をまとめてダウンロードできます。MTP は `--model-draft` と `--spec-type draft-mtp` で渡します。

保存先は **モデル ID 名のディレクトリ** です。ID 未指定なら本体 GGUF のファイル名（拡張子なし）を使います。Models タブの **Model catalog (ID)** は、このディレクトリ単位で本体・mmproj・MTP をまとめて選択します。

Settings タブの **Additional model directories** に、別アプリで使っている保存先を追加できます（既定は空欄）。1 行に 1 パスで、ID フォルダの親を指定します。

```text
./storage/llm
```

このときカタログは `./storage/llm/gemma-4-31B-it-Q4_K_M/`（[ggml-org/gemma-4-31B-it-GGUF](https://huggingface.co/ggml-org/gemma-4-31B-it-GGUF)）や `./storage/llm/Qwen3.8-27B-Q4_K_M/`（[ggml-org/Qwen3.8-27B-GGUF](https://huggingface.co/ggml-org/Qwen3.8-27B-GGUF)）も列挙します。相対パスは作業ルート（`./storage/llamachat-webui` の親）からも探します。ダウンロード先はこれまでどおり **Models directory** です。

```text
models_dir/gemma-4-31B-it-Q4_K_M/
  gemma-4-31B-it-Q4_K_M.gguf
  mmproj-gemma-4-31B-it-bf16.gguf
```

**Start** で次のように起動します。

```text
llama-server --host … --port … --jinja --no-webui --reasoning-format auto …
```

UI は `GET /health` が成功するまで待ちます。**Stop** でプロセスを終了します。

## 機能

- **Chat** — ストリーミング、停止、会話一覧、画像添付（Vision）、推論（`reasoning_content` / `<think>`）、Markdown + LaTeX、tok/s。入力欄は固定高さ（超過分は欄内スクロール）
- **Notebook** — `/completion` による生テキスト生成、停止、直前生成の取り消し
- **Models** — llama-server の導入、起動/停止、ローカル GGUF スキャン、Hugging Face からのファイル取得
- **Settings** — モデル検索パス、システムプロンプト、サンプリング（空欄はサーバ既定）、会話 JSON の import / export

データは `data/` 以下です（`settings.json`、`conversations/`、`uploads/`、`models/`、`llama-server/`）。

## クラウド用 Notebook

`notebooks/` に Colab 用と Paperspace 用があります。**ノートだけ置いて Run All** すれば、残りのファイルは [GitHub](https://github.com/mckey-dev/llamachat-webui) から `git clone` し、`llama-server`・UI 起動まで進みます。毎回不要な処理（更新、停止、掃除）は、関数呼び出しのコメントを外して使います。

GPU ランタイムでは初回に `llama-server` をソースから CUDA ビルドし、実行に必要な成果物だけを永続ディレクトリへ残します。UI は背景起動するので、起動セルは URL を出して完了します。モデルの **Start** は Gradio 上で行います。

| ノート | 用途 |
|--------|------|
| [`notebooks/llamachat-webui-paperspace.ipynb`](notebooks/llamachat-webui-paperspace.ipynb) | Paperspace Gradient。`/notebooks` にノートだけ置く。起動後は `*.gradio.live`（Gradio Share）。venv は `/tmp` |
| [`notebooks/llamachat-webui-colab.ipynb`](notebooks/llamachat-webui-colab.ipynb) | Google Colab。ランタイムを GPU にする。起動後は `*.gradio.live`。**venv なし**（ランタイムへ pip） |

Paperspace の配置（`WORKSPACE` はありません）:

```
/notebooks                      # ノート。ブラウザに見えるのはここだけ
/notebooks/llamachat-webui      # git clone（settings.json、会話、uploads）
/notebooks/storage -> /storage  # ブラウザ用リンク
/notebooks/tmp -> /tmp          # ブラウザ用リンク
/tmp/llamachat-webui            # 一時（venv、モデル実体、ログ、PID）
/storage/llamachat-webui        # 永続（llama-server 本体と付属ライブラリ、残したいモデル）
```

Colab の配置:

```
/content/drive/MyDrive/llamachat-webui # 本体（git clone）+ 設定 + 会話 + llama-server
/content/tmp/llamachat-webui           # 一時（モデル実体、ログ、PID。venv は使わない）
```

## 注意

- モデル切り替えは `llama-server` の再起動が必要です（同時に 1 モデル）。Stop → GGUF または `-hf` を変更 → Start。
- サンプリングの既定は llama-server と同じです（temperature 0.8、top_k 40、top_p 0.95、min_p 0.05、repeat_penalty 1.0）。max_tokens / seed の空欄は無制限 / ランダムです。
- 子プロセス側の公式 Svelte WebUI は無効です（`--no-webui`）。
- `webui.bat` / `webui.sh` は `venv` 作成時だけ `pip install` します。`requirements.txt` を変えたあとは手動インストールか `venv` の作り直しが必要です。
