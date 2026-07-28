from __future__ import annotations

import numpy as np
import torch

from rose.config import GridConfig, RoseConfig, load_config


def canonical_ppm_axis(cfg: RoseConfig | GridConfig | None = None) -> np.ndarray:
    g = cfg.grid if isinstance(cfg, RoseConfig) else cfg or load_config().grid
    return np.linspace(g.ppm_min, g.ppm_max, g.num_points, dtype=np.float32)


def ppm_axis_tensor(
    batch_size: int = 1, *, device: torch.device | str = "cpu", cfg: RoseConfig | None = None
) -> torch.Tensor:
    axis = canonical_ppm_axis(cfg)
    return torch.from_numpy(axis).unsqueeze(0).expand(batch_size, -1).to(device)
