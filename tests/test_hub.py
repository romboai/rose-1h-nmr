from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest
import torch

from rose import DEFAULT_REPO_ID, RoseModel, load, load_config
from rose.hub import is_hub_id, resolve_checkpoint


def test_is_hub_id():
    assert is_hub_id("romboai/rose-1h-nmr")
    assert not is_hub_id("best_model.pt")
    assert not is_hub_id("/tmp/best_model.pt")
    assert not is_hub_id("./ckpt.pt")


def test_resolve_local_checkpoint(tmp_path: Path):
    ckpt = tmp_path / "best_model.pt"
    cfg = load_config()
    model = RoseModel(cfg)
    torch.save({"model_state_dict": model.state_dict()}, ckpt)
    assert resolve_checkpoint(ckpt) == ckpt
    loaded = load(ckpt)
    assert isinstance(loaded, RoseModel)


def test_resolve_missing_local():
    with pytest.raises(FileNotFoundError):
        resolve_checkpoint("/tmp/does_not_exist_rose_ckpt.pt")


def test_hub_download_mocked(tmp_path: Path):
    cfg = load_config()
    model = RoseModel(cfg)
    fake = tmp_path / "best_model.pt"
    torch.save({"model_state_dict": model.state_dict()}, fake)

    def _fake_download(repo_id, filename, revision=None, cache_dir=None):
        assert repo_id == DEFAULT_REPO_ID
        if filename == "best_model.pt":
            return str(fake)
        raise FileNotFoundError(filename)

    with patch("huggingface_hub.hf_hub_download", _fake_download):
        path = resolve_checkpoint(DEFAULT_REPO_ID)
        assert path == fake
        loaded = load(DEFAULT_REPO_ID)
        assert isinstance(loaded, RoseModel)


def test_hub_missing_dep_message():
    import builtins

    from rose import hub

    real_import = builtins.__import__

    def _block(name, *args, **kwargs):
        if name == "huggingface_hub" or name.startswith("huggingface_hub."):
            raise ImportError("blocked")
        return real_import(name, *args, **kwargs)

    with patch("builtins.__import__", _block), pytest.raises(ImportError, match="huggingface_hub"):
        hub.resolve_hub_file("romboai/rose-1h-nmr", filename="best_model.pt")
