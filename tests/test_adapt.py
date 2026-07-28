from __future__ import annotations

import pytest

from rose import RoseModel, load_config
from rose.model import _TASK_HEAD_ATTR


@pytest.fixture
def model():
    return RoseModel(load_config())


def _any_trainable(module) -> bool:
    return any(p.requires_grad for p in module.parameters())


def test_adapt_freeze_encoder_unfreezes_task_head(model):
    out = model.adapt("freeze_encoder", task="peak")
    assert out is model
    assert not _any_trainable(model.spectral_encoder)
    assert not _any_trainable(model.structure_encoder)
    assert _any_trainable(model.peak_head)
    for name, attr in _TASK_HEAD_ATTR.items():
        if name == "peak":
            continue
        assert not _any_trainable(getattr(model, attr))


def test_adapt_unfreeze_encoder_and_task_head(model):
    model.adapt("unfreeze_encoder", task="retrieval")
    assert _any_trainable(model.spectral_encoder)
    assert _any_trainable(model.structure_encoder)
    assert _any_trainable(model.contrastive_head)
    assert not _any_trainable(model.peak_head)


def test_adapt_p1_p2_aliases(model):
    model.adapt("P1", task="peak")
    assert model._adapt_mode == "freeze_encoder"
    model.adapt("P2", task="peak")
    assert model._adapt_mode == "unfreeze_encoder"


def test_param_groups_freeze_encoder_single_head_group(model):
    model.adapt("freeze_encoder", task="peak")
    groups = model.param_groups(lr_head=1e-3, lr_encoder=1e-5)
    assert len(groups) == 1
    assert groups[0]["lr"] == 1e-3
    head_ids = {id(p) for p in model.peak_head.parameters()}
    assert {id(p) for p in groups[0]["params"]} == head_ids


def test_param_groups_unfreeze_encoder_two_lrs(model):
    model.adapt("unfreeze_encoder", task="peak")
    groups = model.param_groups(lr_head=1e-3, lr_encoder=1e-5)
    assert len(groups) == 2
    lrs = sorted(g["lr"] for g in groups)
    assert lrs == [1e-5, 1e-3]


def test_param_groups_before_adapt_raises(model):
    with pytest.raises(ValueError, match="adapt"):
        model.param_groups()


def test_adapt_invalid_mode(model):
    with pytest.raises(ValueError, match="adapt mode"):
        model.adapt("P3", task="peak")  # type: ignore[arg-type]


def test_adapt_unknown_task(model):
    with pytest.raises(ValueError, match="unknown task"):
        model.adapt("freeze_encoder", task="nope")  # type: ignore[arg-type]


def test_print_trainable_parameters(model, capsys):
    model.adapt("freeze_encoder", task="peak")
    model.print_trainable_parameters()
    captured = capsys.readouterr().out
    assert "trainable params:" in captured
    assert "all params:" in captured
    assert "trainable%:" in captured
