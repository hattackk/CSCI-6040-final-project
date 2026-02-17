#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
import tempfile
import venv
from pathlib import Path


DEFAULT_MODELS = ["Qwen/Qwen2.5-3B-Instruct"]


def _default_venv_path() -> Path:
    env_path = os.environ.get("VENV_DIR")
    if env_path:
        return Path(env_path).expanduser()
    return Path(tempfile.gettempdir()) / "csci6040-final-venv"


def _venv_python(venv_path: Path) -> Path:
    if os.name == "nt":
        return venv_path / "Scripts" / "python.exe"
    return venv_path / "bin" / "python"


def _resolve_path(raw_path: str | None, fallback: Path) -> Path:
    if raw_path is None:
        return fallback.resolve()
    path = Path(raw_path).expanduser()
    if not path.is_absolute():
        path = (Path.cwd() / path).resolve()
    return path


def _format_cmd(parts: list[str]) -> str:
    return " ".join(shlex.quote(part) for part in parts)


def _run(cmd: list[str], dry_run: bool = False, env: dict[str, str] | None = None) -> None:
    print(f"[setup] $ {_format_cmd(cmd)}")
    if dry_run:
        return
    subprocess.run(cmd, check=True, env=env)


def _ensure_venv(venv_path: Path, dry_run: bool = False) -> Path:
    py_path = _venv_python(venv_path)
    if py_path.exists():
        print(f"[setup] using existing virtualenv: {venv_path}")
        return py_path

    print(f"[setup] creating virtualenv at: {venv_path}")
    if dry_run:
        return py_path

    venv_path.mkdir(parents=True, exist_ok=True)
    venv.EnvBuilder(with_pip=True).create(str(venv_path))
    return py_path


def _install_dependencies(
    venv_python: Path,
    project_root: Path,
    with_local: bool,
    dry_run: bool,
) -> None:
    _run([str(venv_python), "-m", "pip", "install", "--upgrade", "pip", "setuptools", "wheel"], dry_run=dry_run)
    _run([str(venv_python), "-m", "pip", "install", "-e", str(project_root)], dry_run=dry_run)
    if with_local:
        _run([str(venv_python), "-m", "pip", "install", "-e", f"{project_root}[local]"], dry_run=dry_run)


def _ensure_project_dirs(project_root: Path, dry_run: bool) -> None:
    for rel in ("results", "logs", "data/advbench"):
        target = project_root / rel
        if dry_run:
            print(f"[setup] mkdir -p {target}")
            continue
        target.mkdir(parents=True, exist_ok=True)


def _download_models(
    venv_python: Path,
    models: list[str],
    hf_home: Path | None,
    dry_run: bool,
) -> None:
    if not models:
        return

    _run([str(venv_python), "-m", "pip", "install", "huggingface_hub>=0.24.0"], dry_run=dry_run)

    env = os.environ.copy()
    if hf_home is not None:
        hf_home.mkdir(parents=True, exist_ok=True)
        env["HF_HOME"] = str(hf_home)
        print(f"[setup] HF_HOME={hf_home}")

    code = """
from huggingface_hub import snapshot_download
import os
import sys

cache = os.environ.get("HF_HOME") or None
models = sys.argv[1:]
print(f"[setup] downloading {len(models)} model(s)")
for model in models:
    print(f"[setup] snapshot_download -> {model}")
    path = snapshot_download(repo_id=model, cache_dir=cache)
    print(f"[setup] cached {model} at {path}")
"""
    _run([str(venv_python), "-c", code, *models], dry_run=dry_run, env=env)


def _activation_hint(venv_path: Path) -> str:
    if os.name == "nt":
        ps = f'& "{venv_path}\\Scripts\\Activate.ps1"'
        cmd = f'"{venv_path}\\Scripts\\activate.bat"'
        return (
            "PowerShell:\n"
            f"  {ps}\n"
            "Command Prompt:\n"
            f"  {cmd}"
        )
    return f'source "{venv_path}/bin/activate"'


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Cross-platform project installer: creates virtualenv, installs dependencies, "
            "and optionally pre-downloads local model weights."
        )
    )
    parser.add_argument(
        "--venv-path",
        default=None,
        help="Virtualenv path (default: VENV_DIR env var or OS temp dir).",
    )
    parser.add_argument(
        "--with-local",
        dest="with_local",
        action="store_true",
        default=True,
        help="Install local model dependencies (torch/transformers/accelerate). Default: enabled.",
    )
    parser.add_argument(
        "--no-local",
        dest="with_local",
        action="store_false",
        help="Skip local model dependencies.",
    )
    parser.add_argument(
        "--download-models",
        dest="download_models",
        action="store_true",
        default=True,
        help="Pre-download Hugging Face model(s). Default: enabled.",
    )
    parser.add_argument(
        "--skip-model-download",
        dest="download_models",
        action="store_false",
        help="Skip pre-downloading model weights.",
    )
    parser.add_argument(
        "--model",
        action="append",
        dest="models",
        default=None,
        help=(
            "HF model repo to pre-download. Repeat flag to add more than one model. "
            f"Default: {', '.join(DEFAULT_MODELS)}"
        ),
    )
    parser.add_argument(
        "--hf-home",
        default=None,
        help="Optional Hugging Face cache directory (sets HF_HOME while downloading models).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print actions without executing commands.",
    )
    return parser


def main() -> int:
    if sys.version_info < (3, 10):
        print("Python 3.10+ is required.", file=sys.stderr)
        return 1

    parser = build_parser()
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    venv_path = _resolve_path(args.venv_path, _default_venv_path())
    hf_home = _resolve_path(args.hf_home, Path()) if args.hf_home else None
    models = args.models if args.models else list(DEFAULT_MODELS)

    try:
        _ensure_project_dirs(project_root, dry_run=args.dry_run)
        venv_python = _ensure_venv(venv_path, dry_run=args.dry_run)
        _install_dependencies(
            venv_python=venv_python,
            project_root=project_root,
            with_local=bool(args.with_local),
            dry_run=args.dry_run,
        )
        if args.download_models:
            _download_models(
                venv_python=venv_python,
                models=models,
                hf_home=hf_home,
                dry_run=args.dry_run,
            )
        else:
            print("[setup] skipping model download.")
    except subprocess.CalledProcessError as exc:
        print(f"[setup] command failed with exit code {exc.returncode}", file=sys.stderr)
        return exc.returncode
    except Exception as exc:  # pragma: no cover
        print(f"[setup] failed: {exc}", file=sys.stderr)
        return 1

    print("")
    print("[setup] complete")
    print(f"[setup] project root: {project_root}")
    print(f"[setup] virtualenv: {venv_path}")
    print("[setup] activate:")
    print(_activation_hint(venv_path))
    print("[setup] next:")
    print("  python -m fitd_repro.dashboard --project-root .")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
