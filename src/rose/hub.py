from __future__ import annotations

import hashlib
from pathlib import Path

DEFAULT_REPO_ID = "romboai/rose-1h-nmr"
DEFAULT_REVISION = "7ef29e44865bc515e180c76aa970531993ed5dde"
DEFAULT_WEIGHTS = "best_model.pt"
DEFAULT_CONFIG = "rose.yaml"
# SHA-256 of Hub LFS object best_model.pt at DEFAULT_REVISION.
DEFAULT_WEIGHTS_SHA256 = "f428ff3176de192ea2b4b2088fb5cdeafdb2ee867b6e1685a08db8d740d53a83"


def file_sha256(path: Path, *, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        while True:
            block = f.read(chunk)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def verify_weights_sha256(path: Path, expected: str | None = None) -> None:
    want = DEFAULT_WEIGHTS_SHA256 if expected is None else expected
    got = file_sha256(path)
    if got != want:
        raise ValueError(
            f"checksum mismatch for {Path(path).name}: expected {want}, got {got}. "
            "Refusing to load. Re-download from Hugging Face or pass a local checkpoint."
        )


def _pin_default_weights(repo_id: str, filename: str, revision: str | None) -> bool:
    return (
        repo_id == DEFAULT_REPO_ID
        and filename == DEFAULT_WEIGHTS
        and effective_revision(repo_id, revision) == DEFAULT_REVISION
    )


def is_hub_id(value: str | Path) -> bool:
    s = str(value).strip()
    if not s or s.endswith((".pt", ".pth", ".bin", ".safetensors", ".yaml", ".yml")):
        return False
    p = Path(s)
    if p.exists():
        return False
    return "/" in s and not s.startswith((".", "/", "~"))


def effective_revision(repo_id: str, revision: str | None) -> str | None:
    if revision is not None:
        return revision
    if repo_id == DEFAULT_REPO_ID:
        return DEFAULT_REVISION
    return None


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
    path = Path(
        hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            revision=effective_revision(repo_id, revision),
            cache_dir=str(cache_dir) if cache_dir else None,
        )
    )
    if _pin_default_weights(repo_id, filename, revision):
        verify_weights_sha256(path)
    return path


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
