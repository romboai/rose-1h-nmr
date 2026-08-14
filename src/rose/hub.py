from __future__ import annotations

from pathlib import Path

DEFAULT_REPO_ID = "romboai/rose-1h-nmr"
DEFAULT_WEIGHTS = "best_model.pt"
DEFAULT_CONFIG = "rose.yaml"


def is_hub_id(value: str | Path) -> bool:
    s = str(value).strip()
    if not s or s.endswith((".pt", ".pth", ".bin", ".safetensors", ".yaml", ".yml")):
        return False
    p = Path(s)
    if p.exists():
        return False
    return "/" in s and not s.startswith((".", "/", "~"))


def resolve_hub_file(
    repo_id: str,
    *,
    filename: str,
    revision: str | None = None,
    cache_dir: str | Path | None = None,
) -> Path:
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as e:
        raise ImportError(
            "huggingface_hub required for Hub download. Install: pip install 'rose-1h-nmr[hub]'"
        ) from e
    path = hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        revision=revision,
        cache_dir=str(cache_dir) if cache_dir else None,
    )
    return Path(path)


def resolve_checkpoint(
    checkpoint: str | Path | None = None,
    *,
    revision: str | None = None,
    cache_dir: str | Path | None = None,
    weights_filename: str = DEFAULT_WEIGHTS,
) -> Path:
    if checkpoint is None:
        checkpoint = DEFAULT_REPO_ID
    if is_hub_id(checkpoint):
        return resolve_hub_file(
            str(checkpoint),
            filename=weights_filename,
            revision=revision,
            cache_dir=cache_dir,
        )
    path = Path(checkpoint).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"checkpoint not found: {path}")
    return path


def resolve_config(
    config: str | Path | None = None,
    *,
    repo_id: str | None = None,
    revision: str | None = None,
    cache_dir: str | Path | None = None,
) -> Path | None:
    if config is not None:
        if is_hub_id(config):
            return resolve_hub_file(
                str(config),
                filename=DEFAULT_CONFIG,
                revision=revision,
                cache_dir=cache_dir,
            )
        path = Path(config).expanduser()
        if path.exists():
            return path
        raise FileNotFoundError(f"config not found: {path}")
    if repo_id is not None and is_hub_id(repo_id):
        try:
            return resolve_hub_file(
                str(repo_id),
                filename=DEFAULT_CONFIG,
                revision=revision,
                cache_dir=cache_dir,
            )
        except (ImportError, OSError):
            return None
    return None
