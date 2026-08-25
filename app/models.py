"""ローカル GGUF の列挙と Hugging Face からの取得。"""

from __future__ import annotations

import shutil
import tempfile
from collections.abc import Iterator
from pathlib import Path

from huggingface_hub import hf_hub_download


def _search_bases(anchor: str | Path | None = None) -> list[Path]:
    """相対パスを解決する基準。cwd だけでなく data_dir とその親も見る。"""
    bases: list[Path] = [Path.cwd()]
    if anchor:
        try:
            data_dir = Path(anchor).expanduser().resolve()
        except OSError:
            data_dir = Path(anchor)
        bases.extend([data_dir, data_dir.parent, data_dir.parent.parent])
    for extra in (Path("/notebooks"), Path("/storage"), Path("/content")):
        if extra.is_dir():
            bases.append(extra)
    unique: list[Path] = []
    seen: set[Path] = set()
    for base in bases:
        try:
            resolved = base.resolve()
        except OSError:
            continue
        if resolved in seen:
            continue
        seen.add(resolved)
        unique.append(resolved)
    return unique


def _existing_dir(path: Path) -> Path | None:
    """存在するディレクトリなら resolve して返す。"""
    try:
        resolved = path.expanduser().resolve()
    except OSError:
        return None
    return resolved if resolved.is_dir() else None


def resolve_model_root(line: str, *, anchor: str | Path | None = None) -> Path | None:
    """1 行のモデル参照先を、存在するディレクトリへ解決する。"""
    text = line.strip()
    if not text or text.startswith("#"):
        return None
    path = Path(text).expanduser()
    if path.is_absolute():
        return _existing_dir(path)

    candidates: list[Path] = []
    for base in _search_bases(anchor):
        candidates.append(base / path)
        # `./storage/llm` を `.../storage/llamachat-webui` の隣 `.../storage/llm` としても試す
        if path.name and path.name not in {".", ".."}:
            candidates.append(base / path.name)
    seen: set[Path] = set()
    for candidate in candidates:
        key = candidate
        if key in seen:
            continue
        seen.add(key)
        found = _existing_dir(candidate)
        if found:
            return found
    return None


def parse_model_roots(
    *raw: str | Path | None,
    anchor: str | Path | None = None,
) -> list[Path]:
    """models_dir と追加参照先を、存在するディレクトリのリストにする。空欄は無視する。"""
    paths: list[Path] = []
    seen: set[Path] = set()
    chunks: list[str] = []
    for item in raw:
        if item is None:
            continue
        text = str(item).replace("\r\n", "\n").replace(";", "\n")
        chunks.extend(text.splitlines())
    for line in chunks:
        resolved = resolve_model_root(line, anchor=anchor)
        if resolved is None or resolved in seen:
            continue
        seen.add(resolved)
        paths.append(resolved)
    return paths


def missing_model_roots(
    *raw: str | Path | None,
    anchor: str | Path | None = None,
) -> list[str]:
    """解決できなかった参照先行を返す。"""
    missing: list[str] = []
    chunks: list[str] = []
    for item in raw:
        if item is None:
            continue
        text = str(item).replace("\r\n", "\n").replace(";", "\n")
        chunks.extend(text.splitlines())
    for line in chunks:
        text = line.strip()
        if not text or text.startswith("#"):
            continue
        if resolve_model_root(text, anchor=anchor) is None:
            missing.append(text)
    return missing


def list_gguf(models_dir: str | Path) -> list[str]:
    """指定ディレクトリ配下の GGUF パスを列挙する。"""
    root = Path(models_dir).expanduser() if models_dir else None
    if root is None or not root.is_dir():
        return []
    files = sorted(root.rglob("*.gguf"))
    return [str(path) for path in files]


def _gguf_role(path: Path) -> str:
    """ファイル名から本体 / Vision / MTP の役割を決める。"""
    name = path.name.lower()
    if name.startswith("mmproj") or "-mmproj-" in name:
        return "vision"
    if name.startswith("mtp-") or name.startswith("mtp_") or "-mtp-" in name:
        return "mtp"
    return "model"


def inspect_bundle_dir(folder: Path) -> dict[str, str]:
    """ID ディレクトリ内の本体・mmproj・MTP パスを返す。"""
    bundle = {"id": folder.name, "model": "", "vision": "", "mtp": ""}
    if not folder.is_dir():
        return bundle
    ggufs = sorted(folder.glob("*.gguf"))
    if not ggufs:
        ggufs = sorted(folder.glob("*/*.gguf"))
    for path in ggufs:
        role = _gguf_role(path)
        if not bundle[role]:
            bundle[role] = str(path.resolve())
    return bundle


def list_model_bundles(
    models_dir: str | Path = "",
    extra_dirs: str | Path = "",
    *,
    anchor: str | Path | None = None,
) -> list[dict[str, str]]:
    """models_dir と追加参照先から、ID ディレクトリをカタログとして列挙する。"""
    bundles: list[dict[str, str]] = []
    seen: set[str] = set()
    for root in parse_model_roots(str(models_dir or ""), str(extra_dirs or ""), anchor=anchor):
        found = _bundles_in_root(root)
        for item in found:
            key = item.get("model") or item.get("vision") or item.get("mtp") or ""
            if not key or key in seen:
                continue
            seen.add(key)
            bundles.append(item)
    return bundles


def _bundles_in_root(root: Path) -> list[dict[str, str]]:
    """1 つの参照先から ID バンドルを集める（直下の ID フォルダ、またはそのフォルダ自身）。"""
    items: list[dict[str, str]] = []
    for child in sorted(root.iterdir()):
        if not child.is_dir() or child.name.startswith("."):
            continue
        item = inspect_bundle_dir(child)
        if item["model"] or item["vision"] or item["mtp"]:
            item["root"] = str(root)
            items.append(item)
    if items:
        return items
    direct = inspect_bundle_dir(root)
    if direct["model"] or direct["vision"] or direct["mtp"]:
        direct["root"] = str(root.parent)
        items.append(direct)
    return items


def bundle_choices(
    models_dir: str | Path = "",
    extra_dirs: str | Path = "",
    *,
    anchor: str | Path | None = None,
) -> list[tuple[str, str]]:
    """Dropdown 用の (表示名, 本体パス) 一覧を返す。"""
    choices: list[tuple[str, str]] = []
    for item in list_model_bundles(models_dir, extra_dirs, anchor=anchor):
        value = item["model"] or item["vision"] or item["mtp"]
        label = item["id"]
        root = item.get("root") or ""
        if root:
            label = f"{item['id']} | {root}"
        choices.append((label, value))
    return choices


def catalog_id_of(
    selected: str,
    models_dir: str | Path = "",
    extra_dirs: str | Path = "",
    *,
    anchor: str | Path | None = None,
) -> str:
    """Dropdown の選択からカタログ ID（フォルダ名）を返す。"""
    bundle = bundle_from_selection(selected, models_dir, extra_dirs, anchor=anchor)
    if bundle.get("id"):
        return str(bundle["id"])
    path = Path((selected or "").strip())
    if path.is_file():
        return path.parent.name
    return path.name


def bundle_from_selection(
    selected: str,
    models_dir: str | Path = "",
    extra_dirs: str | Path = "",
    *,
    anchor: str | Path | None = None,
) -> dict[str, str]:
    """選択パスまたは ID から、同じディレクトリの本体・mmproj・MTP を返す。"""
    empty = {"id": "", "model": "", "vision": "", "mtp": ""}
    if not (selected or "").strip():
        return empty
    path = Path(selected)
    if path.is_dir():
        return inspect_bundle_dir(path)
    if path.is_file():
        return inspect_bundle_dir(path.parent)
    for root in parse_model_roots(str(models_dir or ""), str(extra_dirs or ""), anchor=anchor):
        candidate = root / selected
        if candidate.is_dir():
            return inspect_bundle_dir(candidate)
    return {**empty, "model": str(path)}


def bundle_dir_id(
    *,
    bundle_id: str = "",
    model_file: str = "",
    vision_file: str = "",
    mtp_file: str = "",
) -> str:
    """保存先ディレクトリ名（ID）を決める。未指定なら本体 GGUF の stem を使う。"""
    raw = (bundle_id or "").strip()
    if not raw:
        for name in (model_file, vision_file, mtp_file):
            stem = Path((name or "").strip().replace("\\", "/")).stem
            if stem:
                raw = stem
                break
    if not raw:
        raise ValueError("Set a model ID or at least one filename.")
    name = Path(raw.replace("\\", "/")).name
    if not name or name in {".", ".."}:
        raise ValueError("Invalid model ID.")
    return name


def download_hf_file(repo_id: str, filename: str, dest_dir: str | Path, token: str | None = None) -> str:
    """Hugging Face から 1 ファイルを取得し、dest_dir 直下にファイル名だけで保存する。"""
    dest = Path(dest_dir).expanduser()
    dest.mkdir(parents=True, exist_ok=True)
    remote = filename.strip().replace("\\", "/")
    basename = Path(remote).name
    if not basename:
        raise ValueError(f"Invalid filename: {filename}")
    with tempfile.TemporaryDirectory() as tmp:
        path = hf_hub_download(
            repo_id=repo_id.strip(),
            filename=remote,
            local_dir=tmp,
            token=token or None,
        )
        target = dest / basename
        shutil.copy2(path, target)
    return str(target.resolve())


def download_hf_bundle(
    repo_id: str,
    dest_dir: str | Path,
    *,
    model_file: str = "",
    vision_file: str = "",
    mtp_file: str = "",
    bundle_id: str = "",
    token: str | None = None,
) -> Iterator[tuple[str, dict[str, str]]]:
    """同一リポジトリから本体・Vision・MTP を models_dir/ID/ へ保存し、進捗とパスを yield する。"""
    repo = (repo_id or "").strip()
    files = {
        "model": (model_file or "").strip(),
        "vision": (vision_file or "").strip(),
        "mtp": (mtp_file or "").strip(),
    }
    pending = {key: name for key, name in files.items() if name}
    if not repo:
        raise ValueError("Set the Hugging Face repository.")
    if not pending:
        raise ValueError("Set at least one of: model file, Vision file, MTP file.")

    dir_id = bundle_dir_id(
        bundle_id=bundle_id,
        model_file=files["model"],
        vision_file=files["vision"],
        mtp_file=files["mtp"],
    )
    bundle_dir = Path(dest_dir).expanduser() / dir_id
    bundle_dir.mkdir(parents=True, exist_ok=True)

    paths: dict[str, str] = {}
    total = len(pending)
    yield f"Saving into `{bundle_dir}`", dict(paths)
    for index, (key, name) in enumerate(pending.items(), start=1):
        yield f"Downloading ({index}/{total}) `{name}` from `{repo}`…", dict(paths)
        paths[key] = download_hf_file(repo, name, bundle_dir, token=token)
        yield f"Saved `{paths[key]}`", dict(paths)
    yield _bundle_summary(repo, dir_id, paths), paths


def _bundle_summary(repo: str, dir_id: str, paths: dict[str, str]) -> str:
    """ダウンロード結果の Markdown 要約を作る。"""
    lines = [f"Downloaded from `{repo}` into `{dir_id}/`:"]
    labels = {"model": "Model", "vision": "Vision (mmproj)", "mtp": "MTP"}
    for key, label in labels.items():
        if key in paths:
            lines.append(f"- **{label}:** `{paths[key]}`")
    return "\n".join(lines)
