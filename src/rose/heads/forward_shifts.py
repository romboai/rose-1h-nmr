from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class ForwardShiftHead(nn.Module):
    def __init__(self, struct_dim: int = 256, max_peaks: int = 32, hidden_dim: int = 128):
        super().__init__()
        self.max_peaks = max_peaks
        self.shared = nn.Sequential(
            nn.LayerNorm(struct_dim), nn.Linear(struct_dim, hidden_dim), nn.GELU()
        )
        self.shift_head = nn.Linear(hidden_dim, max_peaks)
        self.count_head = nn.Linear(hidden_dim, 1)

    def forward(self, struct_embed: torch.Tensor) -> dict[str, torch.Tensor]:
        h = self.shared(struct_embed)
        return {"shifts_pred": self.shift_head(h), "count_pred": self.count_head(h).squeeze(-1)}


def chamfer_set_loss(
    pred_shifts: torch.Tensor,
    target_ppms_list: list[torch.Tensor],
    *,
    target_min: float = -2.0,
    target_max: float = 12.0,
    max_peaks: int = 32,
) -> torch.Tensor:
    B = pred_shifts.size(0)
    device = pred_shifts.device
    losses = []
    for i in range(B):
        gt = target_ppms_list[i].to(device)
        if gt.numel() == 0:
            continue
        keep = (gt >= target_min) & (gt <= target_max)
        if not keep.any():
            continue
        gt = gt[keep]
        gt_top = gt[:max_peaks]
        pred = pred_shifts[i, : gt_top.numel()]
        d_pg = (pred.unsqueeze(1) - gt_top.unsqueeze(0)).pow(2)
        loss_pg = d_pg.min(dim=1).values.mean()
        loss_gp = d_pg.min(dim=0).values.mean()
        losses.append(0.5 * (loss_pg + loss_gp))
    if not losses:
        return pred_shifts.sum() * 0.0
    return torch.stack(losses).mean()


def forward_loss(
    out: dict[str, torch.Tensor],
    target_peaks_ppm: list[torch.Tensor],
    *,
    target_min: float = -2.0,
    target_max: float = 12.0,
    max_peaks: int = 32,
    count_weight: float = 0.1,
) -> torch.Tensor:
    counts = torch.tensor(
        [int(t.numel()) for t in target_peaks_ppm],
        device=out["count_pred"].device,
        dtype=out["count_pred"].dtype,
    )
    l_chamfer = chamfer_set_loss(
        out["shifts_pred"],
        target_peaks_ppm,
        target_min=target_min,
        target_max=target_max,
        max_peaks=max_peaks,
    )
    l_count = F.smooth_l1_loss(out["count_pred"], counts.clamp(max=max_peaks).float())
    return l_chamfer + count_weight * l_count
