from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class PeakPickingHead(nn.Module):
    def __init__(self, embed_dim: int = 256, patch_size: int = 32, hidden_dim: int = 128):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, patch_size),
        )

    def forward(self, patch_embeds: torch.Tensor) -> torch.Tensor:
        B, N, _ = patch_embeds.shape
        out = self.proj(patch_embeds)
        return out.reshape(B, N * self.patch_size)


def peaks_to_heatmap(
    peaks_ppm: list[torch.Tensor],
    peaks_intensity: list[torch.Tensor],
    *,
    ppm_min: float,
    ppm_max: float,
    num_points: int,
    sigma_ppm: float = 0.02,
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    B = len(peaks_ppm)
    grid = torch.linspace(ppm_min, ppm_max, num_points, device=device)
    out = torch.zeros(B, num_points, device=device)
    sigma2_2 = 2.0 * sigma_ppm * sigma_ppm
    for i in range(B):
        ppms = peaks_ppm[i].to(device)
        ints = peaks_intensity[i].to(device)
        if ppms.numel() == 0:
            continue
        keep = (ppms >= ppm_min) & (ppms <= ppm_max)
        if not keep.any():
            continue
        ppms = ppms[keep]
        ints = ints[keep]
        diff = grid.unsqueeze(0) - ppms.unsqueeze(1)
        kernel = torch.exp(-(diff**2) / sigma2_2)
        weights = ints.clamp(min=0.0)
        if float(weights.max()) < 1e-09:
            weights = torch.ones_like(ints)
        heat = (weights.unsqueeze(1) * kernel).sum(dim=0)
        m = heat.max().clamp(min=1e-09)
        out[i] = (heat / m).clamp(0.0, 1.0)
    return out


def peak_picking_loss(
    logits: torch.Tensor,
    target_heatmap: torch.Tensor,
    *,
    pos_weight: float = 100.0,
    has_peaks: torch.Tensor | None = None,
    dice_weight: float = 1.0,
) -> torch.Tensor:
    if has_peaks is not None:
        keep = has_peaks.bool()
        if not keep.any():
            return logits.sum() * 0.0
        logits = logits[keep]
        target_heatmap = target_heatmap[keep]
    pw = torch.tensor([pos_weight], device=logits.device, dtype=logits.dtype)
    bce = F.binary_cross_entropy_with_logits(logits, target_heatmap, pos_weight=pw)
    if dice_weight <= 0:
        return bce
    prob = torch.sigmoid(logits)
    inter = (prob * target_heatmap).sum(dim=-1)
    denom = prob.sum(dim=-1) + target_heatmap.sum(dim=-1)
    dice = 1.0 - (2.0 * inter + 1.0) / (denom + 1.0)
    return bce + dice_weight * dice.mean()
