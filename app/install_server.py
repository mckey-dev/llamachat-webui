"""公式 GitHub Releases から llama-server を導入する。"""

from __future__ import annotations

import argparse
import os
import platform
import re
import shutil
import sys
import tarfile
import zipfile
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import httpx

REPO = "ggml-org/llama.cpp"
API_LATEST = f"https://api.github.com/repos/{REPO}/releases/latest"
API_RELEASES = f"https://api.github.com/repos/{REPO}/releases"
API_TAG = f"https://api.github.com/repos/{REPO}/releases/tags/{{tag}}"
USER_AGENT = "llamachat-webui"
BINARY_NAMES = ("llama-server.exe", "llama-server")
# llama.cpp の公式バイナリ資産。vX.Y.Z の semver リリースは資産がほぼ無い。
_BINARY_ASSET_RE = re.compile(r"^llama-.+-bin-.+\.(?:zip|tar\.gz|tgz)$", re.IGNORECASE)

BACKEND_CHOICES = (
    ("Auto (detect GPU)", "auto"),
    ("CPU", "cpu"),
    ("CUDA (Windows, latest)", "cuda"),
    ("CUDA 12.4 (Windows)", "cuda-12.4"),
    ("CUDA 13.3 (Windows)", "cuda-13.3"),
    ("Vulkan", "vulkan"),
    ("ROCm", "rocm"),
)


class InstallError(RuntimeError):
    """導入失敗（資産なし・非対応環境など）。"""


def detect_os_arch() -> tuple[str, str]:
    """実行環境を (win|macos|ubuntu, x64|arm64) で返す。"""
    system = sys.platform
    machine = platform.machine().lower()
    if machine in {"x86_64", "amd64"}:
        arch = "x64"
    elif machine in {"arm64", "aarch64"}:
        arch = "arm64"
    else:
        raise InstallError(f"Unsupported CPU architecture: {platform.machine()}")

    if system == "win32":
        return "win", arch
    if system == "darwin":
        return "macos", arch
    if system.startswith("linux"):
        return "ubuntu", arch
    raise InstallError(f"Unsupported OS: {system}")


def detect_backend(os_key: str) -> str:
    """GPU の有無から既定バックエンド (cpu / cuda / vulkan) を決める。"""
    if os_key == "macos":
        return "cpu"
    if _has_nvidia():
        return "cuda" if os_key == "win" else "vulkan"
    return "cpu"


def _has_nvidia() -> bool:
    """nvidia-smi が使えるなら NVIDIA GPU ありとみなす。"""
    if shutil.which("nvidia-smi") is None:
        return False
    try:
        result = __import__("subprocess").run(
            ["nvidia-smi", "-L"],
            capture_output=True,
            text=True,
            timeout=8,
        )
        return result.returncode == 0 and bool(result.stdout.strip())
    except Exception:
        return True


def _headers() -> dict[str, str]:
    """GitHub API 用ヘッダ。GITHUB_TOKEN があれば付ける。"""
    headers = {"User-Agent": USER_AGENT, "Accept": "application/vnd.github+json"}
    token = os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return headers


def fetch_release(tag: str = "") -> dict[str, Any]:
    """指定タグ、またはバイナリ付きの最新リリース JSON を取得する。

    GitHub の /releases/latest は pre-release を除外する。llama.cpp の
    日常ビルド (bXXXX) は pre-release のため、資産のない semver タグ
    (例: v0.2.0) が latest になることがある。タグ未指定時は一覧から
    バイナリ資産のある最新を選ぶ。
    """
    with httpx.Client(headers=_headers(), timeout=30.0, follow_redirects=True) as client:
        if tag.strip():
            return _get_release_by_tag(client, tag.strip())
        return _get_latest_binary_release(client)


def _raise_for_github(response: httpx.Response, *, missing: str) -> None:
    """GitHub API の代表的な失敗を InstallError に変換する。"""
    if response.status_code == 404:
        raise InstallError(f"Release not found: {missing}")
    if response.status_code == 403:
        raise InstallError("GitHub API rate limited. Set GITHUB_TOKEN and retry.")
    response.raise_for_status()


def _get_release_by_tag(client: httpx.Client, tag: str) -> dict[str, Any]:
    """タグ指定のリリース JSON を返す。"""
    response = client.get(API_TAG.format(tag=tag))
    _raise_for_github(response, missing=tag)
    data = response.json()
    if not isinstance(data, dict) or not data.get("tag_name"):
        raise InstallError("Unexpected GitHub release payload.")
    return data


def _release_has_binaries(release: dict[str, Any]) -> bool:
    """公式 llama-server バイナリ zip/tar が1つ以上あるか。"""
    for item in release.get("assets") or []:
        if not isinstance(item, dict):
            continue
        name = str(item.get("name") or "")
        if _BINARY_ASSET_RE.match(name):
            return True
    return False


def _get_latest_binary_release(client: httpx.Client) -> dict[str, Any]:
    """バイナリ資産付きの最新（非 draft）リリースを返す。"""
    # まず /latest を試し、資産があればそれを使う（将来 semver にバイナリが付いた場合）
    response = client.get(API_LATEST)
    if response.status_code not in {403, 404}:
        response.raise_for_status()
        data = response.json()
        if isinstance(data, dict) and data.get("tag_name") and _release_has_binaries(data):
            return data

    # pre-release の bXXXX など、一覧からバイナリ付きを探す
    response = client.get(API_RELEASES, params={"per_page": 30})
    _raise_for_github(response, missing="latest")
    releases = response.json()
    if not isinstance(releases, list):
        raise InstallError("Unexpected GitHub releases payload.")
    for item in releases:
        if not isinstance(item, dict) or item.get("draft"):
            continue
        if item.get("tag_name") and _release_has_binaries(item):
            return item
    raise InstallError(
        "No llama.cpp release with installable binaries was found. "
        "Set --llama-tag / LLAMA_TAG to a build such as b10618."
    )


def asset_names(release: dict[str, Any]) -> dict[str, str]:
    """リリース資産のファイル名 → ダウンロード URL の対応表を作る。"""
    mapping: dict[str, str] = {}
    for item in release.get("assets") or []:
        if not isinstance(item, dict):
            continue
        name = str(item.get("name") or "")
        url = str(item.get("browser_download_url") or "")
        if name and url:
            mapping[name] = url
    return mapping


def pick_assets(
    names: list[str],
    os_key: str,
    arch: str,
    backend: str,
    tag: str,
) -> tuple[str, str | None]:
    """OS / アーキ / バックエンドに合う本体アーカイブと、任意の cudart を選ぶ。"""
    backend = (backend or "auto").strip().lower()
    if backend == "auto":
        backend = detect_backend(os_key)

    if os_key == "macos":
        main = _must_exist(names, f"llama-{tag}-bin-macos-{arch}.tar.gz")
        return main, None

    if backend == "cpu":
        if os_key == "win":
            main = _must_exist(names, f"llama-{tag}-bin-win-cpu-{arch}.zip")
        else:
            main = _must_exist(names, f"llama-{tag}-bin-ubuntu-{arch}.tar.gz")
        return main, None

    if backend == "vulkan":
        if os_key == "win":
            main = _must_exist(names, f"llama-{tag}-bin-win-vulkan-{arch}.zip")
        else:
            main = _must_exist(names, f"llama-{tag}-bin-ubuntu-vulkan-{arch}.tar.gz")
        return main, None

    if backend == "rocm":
        hits = [n for n in names if _rocm_name(n, os_key, arch, tag)]
        if not hits:
            raise InstallError(
                f"No ROCm build for {os_key}/{arch} in {tag}. "
                "Pick CPU or Vulkan, or install llama-server yourself."
            )
        return sorted(hits)[-1], None

    if backend.startswith("cuda"):
        if os_key != "win":
            raise InstallError(
                "Official llama.cpp Linux/macOS releases do not include CUDA binaries. "
                "On Linux use Vulkan, or provide your own CUDA llama-server."
            )
        version = backend.split("-", 1)[1] if "-" in backend else None
        main = _pick_cuda(names, tag, arch, version)
        cudart = _pick_cudart(names, arch, _cuda_version_from_name(main))
        return main, cudart

    raise InstallError(f"Unknown backend: {backend}")


def _rocm_name(name: str, os_key: str, arch: str, tag: str) -> bool:
    """資産名がこの OS/arch 向けの ROCm/HIP ビルドか判定する。"""
    if not name.startswith(f"llama-{tag}-bin-{os_key}-"):
        return False
    if not (name.endswith(f"-{arch}.zip") or name.endswith(f"-{arch}.tar.gz")):
        return False
    return "rocm" in name or "hip" in name


def _cuda_version_from_name(name: str) -> str:
    """ファイル名から CUDA バージョン (例: 13.3) を取り出す。"""
    match = re.search(r"cuda-(\d+\.\d+)", name)
    if not match:
        raise InstallError(f"Could not parse CUDA version from {name}")
    return match.group(1)


def _pick_cuda(names: list[str], tag: str, arch: str, version: str | None) -> str:
    """Windows CUDA 資産を選ぶ。version 未指定なら最新を使う。"""
    prefix = f"llama-{tag}-bin-win-cuda-"
    suffix = f"-{arch}.zip"
    candidates: list[tuple[tuple[int, int], str]] = []
    for name in names:
        if not name.startswith(prefix) or not name.endswith(suffix):
            continue
        ver = _cuda_version_from_name(name)
        if version and ver != version:
            continue
        major, minor = (int(p) for p in ver.split("."))
        candidates.append(((major, minor), name))
    if not candidates:
        wanted = version or "any"
        raise InstallError(f"No Windows CUDA {wanted} build for {arch} in {tag}.")
    candidates.sort()
    return candidates[-1][1]


def _pick_cudart(names: list[str], arch: str, version: str) -> str | None:
    """対応する cudart zip 名を返す。無ければ None。"""
    expected = f"cudart-llama-bin-win-cuda-{version}-{arch}.zip"
    return expected if expected in names else None


def _must_exist(names: list[str], filename: str) -> str:
    """資産リストに無ければ InstallError。"""
    if filename not in names:
        raise InstallError(f"Asset not in this release: {filename}")
    return filename


def find_binary(root: Path) -> Path | None:
    """ディレクトリ配下の llama-server を、更新日時が新しいものから探す。"""
    if not root.is_dir():
        return None
    hits: list[Path] = []
    for name in BINARY_NAMES:
        hits.extend(p for p in root.rglob(name) if p.is_file())
    if not hits:
        return None
    hits.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return hits[0]


def install_dir(data_dir: Path, tag: str, backend: str) -> Path:
    """導入先 data/llama-server/<tag>-<backend>/ を返す。"""
    safe_backend = re.sub(r"[^a-z0-9.\-]+", "-", backend.lower())
    return Path(data_dir) / "llama-server" / f"{tag}-{safe_backend}"


def download_file(url: str, dest: Path) -> Iterator[str]:
    """URL からファイルを保存し、進捗メッセージを yield する。"""
    dest.parent.mkdir(parents=True, exist_ok=True)
    with httpx.Client(headers=_headers(), timeout=None, follow_redirects=True) as client:
        with client.stream("GET", url) as response:
            response.raise_for_status()
            total = int(response.headers.get("Content-Length") or 0)
            done = 0
            with dest.open("wb") as handle:
                for chunk in response.iter_bytes(1024 * 256):
                    handle.write(chunk)
                    done += len(chunk)
                    if total:
                        pct = min(100, int(done * 100 / total))
                        yield f"Downloading {dest.name}: {pct}% ({_mb(done)} / {_mb(total)} MB)"
                    else:
                        yield f"Downloading {dest.name}: {_mb(done)} MB"
    yield f"Downloaded {dest.name}"


def extract_archive(archive: Path, dest: Path) -> None:
    """zip / tar.gz を dest へ展開する。"""
    dest.mkdir(parents=True, exist_ok=True)
    name = archive.name.lower()
    if name.endswith(".zip"):
        with zipfile.ZipFile(archive) as zf:
            _safe_zip_extract(zf, dest)
        return
    if name.endswith(".tar.gz") or name.endswith(".tgz"):
        with tarfile.open(archive, "r:gz") as tf:
            kwargs: dict[str, Any] = {}
            if hasattr(tarfile, "data_filter"):
                kwargs["filter"] = "data"
            _safe_tar_extract(tf, dest, kwargs)
        return
    raise InstallError(f"Unsupported archive: {archive.name}")


def _safe_zip_extract(zf: zipfile.ZipFile, dest: Path) -> None:
    """zip slip を防ぎつつ zip を展開する。"""
    dest = dest.resolve()
    for info in zf.infolist():
        target = (dest / info.filename).resolve()
        if not _is_inside(dest, target):
            raise InstallError(f"Unsafe zip path: {info.filename}")
        zf.extract(info, dest)


def _safe_tar_extract(tf: tarfile.TarFile, dest: Path, extract_kwargs: dict[str, Any] | None = None) -> None:
    """パストラバーサルを防ぎつつ tar を展開する。"""
    dest = dest.resolve()
    for member in tf.getmembers():
        target = (dest / member.name).resolve()
        if not _is_inside(dest, target):
            raise InstallError(f"Unsafe tar path: {member.name}")
    tf.extractall(dest, **(extract_kwargs or {}))


def _is_inside(root: Path, target: Path) -> bool:
    """target が root 配下か判定する。"""
    try:
        target.relative_to(root)
        return True
    except ValueError:
        return False


def _mb(num: int) -> str:
    """バイト数を MB 表記（小数 1 桁）にする。"""
    return f"{num / (1024 * 1024):.1f}"


def _copy_dlls_beside_exe(exe: Path, search_root: Path) -> None:
    """展開した DLL を exe と同じフォルダへ集め、CUDA runtime を見つけやすくする。"""
    if exe.suffix.lower() != ".exe":
        return
    for dll in search_root.rglob("*.dll"):
        target = exe.parent / dll.name
        if dll.resolve() == target.resolve():
            continue
        if not target.exists():
            shutil.copy2(dll, target)


def install_llama_server(
    data_dir: Path,
    backend: str = "auto",
    tag: str = "",
) -> Iterator[tuple[str, str | None]]:
    """公式バイナリを導入する。進捗と、成功時は実行ファイルパスを yield する。"""
    os_key, arch = detect_os_arch()
    resolved_backend = (backend or "auto").strip() or "auto"
    if resolved_backend == "auto":
        resolved_backend = detect_backend(os_key)
        yield f"Detected backend: **{resolved_backend}** ({os_key}/{arch})", None
    else:
        yield f"Installing backend **{resolved_backend}** for {os_key}/{arch}", None

    yield "Fetching GitHub release metadata…", None
    release = fetch_release(tag)
    tag_name = str(release["tag_name"])
    urls = asset_names(release)
    main_name, cudart_name = pick_assets(list(urls), os_key, arch, resolved_backend, tag_name)
    dest = install_dir(data_dir, tag_name, resolved_backend)
    dest.mkdir(parents=True, exist_ok=True)
    tmp = dest / ".download"
    tmp.mkdir(parents=True, exist_ok=True)

    try:
        main_path = tmp / main_name
        for line in download_file(urls[main_name], main_path):
            yield line, None
        yield f"Extracting `{main_name}`…", None
        extract_archive(main_path, dest)

        if cudart_name and cudart_name in urls:
            cudart_path = tmp / cudart_name
            for line in download_file(urls[cudart_name], cudart_path):
                yield line, None
            yield f"Extracting `{cudart_name}` (CUDA runtime)…", None
            extract_archive(cudart_path, dest)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    binary = find_binary(dest)
    if binary is None:
        raise InstallError(f"llama-server was not found after extracting {main_name}")
    if os_key != "win":
        binary.chmod(binary.stat().st_mode | 0o111)
    _copy_dlls_beside_exe(binary, dest)
    yield f"Installed **{tag_name}** → `{binary}`", str(binary)


def install_blocking(data_dir: Path, backend: str = "auto", tag: str = "") -> str:
    """導入を完了まで実行し、llama-server のパスを返す。"""
    path = ""
    last = ""
    for message, found in install_llama_server(data_dir, backend=backend, tag=tag):
        last = message
        print(message)
        if found:
            path = found
    if not path:
        raise InstallError(last or "Install failed.")
    return path


def main() -> None:
    """python -m app.install_server の CLI。"""
    parser = argparse.ArgumentParser(description="Install llama-server from llama.cpp GitHub Releases")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--backend", default="auto", help="auto, cpu, cuda, cuda-12.4, cuda-13.3, vulkan, rocm")
    parser.add_argument("--tag", default="", help="Release tag such as b10549 (default: latest)")
    args = parser.parse_args()
    try:
        path = install_blocking(Path(args.data_dir), backend=args.backend, tag=args.tag)
    except (InstallError, httpx.HTTPError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
    print(path)


if __name__ == "__main__":
    main()
