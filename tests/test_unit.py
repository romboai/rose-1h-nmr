from __future__ import annotations

import torch

from rose import RoseModel, load_config
from rose.api import encode, smiles_batch
from rose.grid import canonical_ppm_axis


def test_build_and_forward():
    cfg = load_config()
    model = RoseModel(cfg)
    model.eval()
    spec = torch.randn(2, cfg.grid.num_points)
    ppm = torch.from_numpy(canonical_ppm_axis(cfg)).unsqueeze(0).expand(2, -1)
    z = model.encode(spec, ppm)
    assert z.shape == (2, cfg.spectral_encoder.embed_dim)


def test_encode_numpy():
    cfg = load_config()
    model = RoseModel(cfg)
    model.eval()
    import numpy as np

    spec = np.random.randn(cfg.grid.num_points).astype(np.float32)
    z = encode(model, spec)
    assert z.shape == (1, cfg.spectral_encoder.embed_dim)


def test_smiles_batch():
    batch = smiles_batch(["CCO", "c1ccccc1"])
    assert batch.num_graphs == 2
