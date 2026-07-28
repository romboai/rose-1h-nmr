from __future__ import annotations

import numpy as np
import pytest
import torch

from rose import RoseModel, load_config, predict
from rose.grid import ppm_axis_tensor


@pytest.fixture
def cfg():
    return load_config()


@pytest.fixture
def model(cfg):
    m = RoseModel(cfg)
    m.eval()
    return m


@pytest.fixture
def spec(cfg):
    rng = np.random.default_rng(0)
    return rng.standard_normal(cfg.grid.num_points, dtype=np.float32)


@pytest.fixture
def spec_b(cfg):
    rng = np.random.default_rng(1)
    return rng.standard_normal(cfg.grid.num_points, dtype=np.float32)


def test_zero_shot_denoise(model, spec):
    out = predict(model, spec, task="denoise")
    assert out.shape == (1, model.cfg.grid.num_points)


def test_zero_shot_peak(model, spec):
    out = predict(model, spec, task="peak")
    assert out.shape == (1, model.cfg.grid.num_points)
    assert float(out.min()) >= 0.0 and float(out.max()) <= 1.0


def test_zero_shot_retrieval(model, spec):
    out = predict(model, spec, task="retrieval", smiles="CCO")
    assert out["logits_per_spec"].shape == (1, 1)
    assert out["logits_per_struct"].shape == (1, 1)


def test_zero_shot_retrieval_batch(model, cfg, spec, spec_b):
    batch = np.stack([spec, spec_b])
    out = predict(model, batch, task="retrieval", smiles=["CCO", "c1ccccc1"])
    assert out["logits_per_spec"].shape == (2, 2)


def test_zero_shot_forward(model, spec):
    out = predict(model, spec, task="forward", smiles="CCO")
    assert out["shifts_pred"].shape == (1, model.cfg.heads.forward_max_peaks)
    assert out["count_pred"].shape == (1,)


def test_zero_shot_pair(model, spec, spec_b):
    out = predict(model, spec, task="pair", spectrum_b=spec_b)
    assert out["similarity"].shape == (1,)
    assert out["logits"].shape == (1, 1)
    assert "loss" not in out


def test_zero_shot_pair_batch(model, cfg, spec, spec_b):
    batch_a = np.stack([spec, spec_b])
    batch_b = np.stack([spec_b, spec])
    out = predict(model, batch_a, task="pair", spectrum_b=batch_b)
    assert out["similarity"].shape == (2,)
    assert out["logits"].shape == (2, 2)


def test_pair_batch_size_mismatch(model, spec, spec_b):
    batch = np.stack([spec, spec_b])
    with pytest.raises(ValueError, match="same batch size"):
        predict(model, batch, task="pair", spectrum_b=spec_b)


def _freeze_encoder(model: RoseModel) -> None:
    for p in model.spectral_encoder.parameters():
        p.requires_grad = False
    for p in model.structure_encoder.parameters():
        p.requires_grad = False


def test_p1_frozen_encoder_peak_head_updates(model, cfg, spec):
    _freeze_encoder(model)
    task = "peak"
    head = model.head(task)
    head.train()
    opt = torch.optim.Adam(head.parameters(), lr=1e-2)
    before = [p.detach().clone() for p in head.parameters()]

    spec_t = torch.from_numpy(spec).unsqueeze(0)
    ppm = ppm_axis_tensor(1, device="cpu", cfg=cfg)
    out = model.run_task(task, spectrum=spec_t, ppm_axis=ppm)
    loss = out.pow(2).mean()
    loss.backward()
    opt.step()

    for p in model.spectral_encoder.parameters():
        assert p.grad is None
    for p in model.structure_encoder.parameters():
        assert p.grad is None
    assert any(not torch.equal(b, p) for b, p in zip(before, head.parameters()))


def test_p1_frozen_encoder_retrieval_head_updates(model, cfg, spec, spec_b):
    _freeze_encoder(model)
    task = "retrieval"
    head = model.head(task)
    head.train()
    opt = torch.optim.Adam(head.parameters(), lr=1e-2)

    batch = torch.from_numpy(np.stack([spec, spec_b]))
    ppm = ppm_axis_tensor(2, device="cpu", cfg=cfg)
    from rose.api import smiles_batch

    mol = smiles_batch(["CCO", "c1ccccc1"])
    out = model.run_task(task, spectrum=batch, ppm_axis=ppm, mol_batch=mol)
    loss = out["loss"]
    loss.backward()

    for p in model.spectral_encoder.parameters():
        assert p.grad is None
    for p in model.structure_encoder.parameters():
        assert p.grad is None
    head_grads = [p.grad for p in head.parameters() if p.grad is not None]
    assert head_grads
    assert any(g.abs().sum() > 0 for g in head_grads)
    opt.step()


def test_head_maps_tasks(model):
    assert model.head("peak") is model.peak_head
    assert model.head("retrieval") is model.contrastive_head
    assert model.head("pair") is model.pair_contrastive_head
