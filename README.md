# llamachat-webui

[llama.cpp](https://github.com/ggml-org/llama.cpp) の `llama-server` 向け Gradio フロントエンドです。  
`llama-server` を子プロセスとして起動・停止し、HTTP（`/v1/chat/completions`、`/completion`）だけで会話します。公式 WebUI に近い操作感を目指しています。

MCP / ツール呼び出し / ルーターモードは対象外です。

- リポジトリ: https://github.com/mckey-dev/llamachat-webui
- **更新日:** 2026-08-26

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

ブラウザで http://127.0.0.1:7862 を開きます。

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
python launch.py --server-name 127.0.0.1 --server-port 7862
```

| 引数 | 説明 |
|------|------|
| `--server-name` | Gradio の待受アドレス（既定 `127.0.0.1`） |
| `--server-port` | Gradio のポート（既定 `7862`） |
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

## 画面の使い方

ブラウザで UI を開いたら、次の順が基本です。

1. **Models** でモデル（と必要なら llama-server）を用意する  
2. ヘッダの **Start** で `llama-server` を起動する（状態が ready になるまで待つ）  
3. **Chat** または **Notebook** で使う  
4. 終わったらヘッダの **Stop**

### ヘッダ（全タブ共通）

画面上部にタイトル、状態表示、**Start** / **Stop** があります。

| 操作 | 内容 |
|------|------|
| **Start** | Models / Settings の内容を保存し、`llama-server` を子プロセスとして起動する。`GET /health` が成功するまで待つ |
| **Stop** | `llama-server` を終了する |
| 状態表示 | 起動中・ready・停止・エラーなどを表示する |

モデルや Extra args を変えたあとは、**Stop → Start** で再起動してください（同時に 1 モデルだけ）。

### Models

`llama-server` の導入と、起動に渡すモデル・引数を設定します。

1. **Install llama-server**（未導入のとき）  
   Backend（`auto` / `cpu` / `cuda` など）と任意の Release tag を選び、**Install / Update**。パスは **llama-server path** に入ります。自前バイナリならパスを直接指定しても構いません。
2. モデルを指定する（どちらか一方で可）  
   - **Hugging Face repo (-hf)** … 例: `ggml-org/Qwen2.5-3B-Instruct-GGUF:Q4_K_M`  
   - **Local GGUF path (-m)** … ローカルの `.gguf`  
   任意で **mmproj**（Vision）、**MTP draft**、および下の起動オプション。
3. **Model catalog (ID)**  
   Models directory 配下の ID フォルダ単位で、本体・mmproj・MTP をまとめて選べます。**Refresh** で再スキャン。
4. **Download from Hugging Face**（任意）  
   Repository とファイル名を指定して、Models directory の ID フォルダへ取得します。
5. ヘッダの **Start**  
   下の **llama-server log** に起動ログが出ます。

パスやカタログの詳細は「モデルの使い方」を参照してください。

#### Context size / GPU layers / Extra args

Models タブの次の項目は、**Start** 時に `llama-server` へ渡されます。変更後は Stop → Start が必要です。

| 項目 | llama-server | 説明 |
|------|--------------|------|
| **Context size (-c, 0 = model default)** | `-c` | コンテキスト長（トークン数）。会話・プロンプト・生成が収まる上限に近い値です。大きいほどメモリ（VRAM / RAM）を使います。**`0` のときは `-c` を付けず、モデル側の既定に任せます。** |
| **GPU layers (-ngl)** | `-ngl` | GPU に載せるレイヤ数。`auto`（既定）は llama-server に任せる、`all` は可能な限り全部、数値はレイヤ数の上限です。CPU のみのビルドでは効果が薄い／無視されることがあります。VRAM が足りないときは数値を下げてください。 |
| **Extra llama-server args** | （追加 argv） | 上記以外の起動引数を空白区切りで書きます。例: `--flash-attn on`、`--fit on --fit-target 24576`（単位 MiB）。シェルと同様に引用符で囲めます。MTP 利用時、ここに `--spec-type` / `--spec-draft-n-max` が無ければ UI 側で既定を足します。 |

A6000（48GB）で VRAM を約 24GB 空けたい例:

```text
--fit on --fit-target 24576
```

### Chat

会話形式のチャットです（`/v1/chat/completions`）。**Start** 済みである必要があります。

| 操作 | 内容 |
|------|------|
| **Conversations** | 保存済み会話の切り替え |
| **New chat** | 新しい会話を作る |
| **Delete** | 選択中の会話を削除する |
| 入力欄 | テキスト送信。画像を添付可（Vision モデル + mmproj 時）。Enter で送信 |
| **Stop generation** | 生成を中断する |

応答はストリーミング表示されます。推論タグ（`<think>` など）や Markdown / LaTeX、速度（tok/s）も表示されます。システムプロンプトとサンプリングは **Settings** の値が使われます。

### Notebook

チャットテンプレートを使わない生テキスト生成です（`/completion`）。エディタ全体がプロンプトで、生成文はその末尾に追記されます。

| 操作 | 内容 |
|------|------|
| **Generate** | 現在のテキストをプロンプトとして生成を開始する |
| **Stop** | 生成を中断する |
| **Undo last generation** | 直前の生成分だけ取り消す |

こちらも **Start** 済みである必要があり、サンプリングは **Settings** を使います。

### Settings

アプリ全体の設定です。変更は保存され、次の **Start** や生成に反映されます（モデル切替は再 Start が必要）。

| 項目 | 内容 |
|------|------|
| **Models directory** | ダウンロード先・カタログの基準ディレクトリ |
| **Additional model directories** | 別アプリのモデル置き場など。1 行に 1 パス |
| **System prompt** | Chat 用。空ならモデル既定 |
| サンプリング | Temperature / Top K / Top P / Min P / Repeat・Presence・Frequency penalty / Max tokens / Seed。空欄はサーバ既定または無制限・ランダム |
| **Vision image max side** | 添付画像の長辺上限（px） |
| **Save settings** | 設定をディスクへ保存 |
| **Import / Export conversations** | 会話 JSON の読み書き |

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
