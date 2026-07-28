from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class PairContrastiveHead(nn.Module):
    def __init__(self, spec_dim: int, proj_dim: int = 128, temperature_init: float = 0.1):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(spec_dim, proj_dim), nn.GELU(), nn.Linear(proj_dim, proj_dim)
        )
        self.temperature = nn.Parameter(torch.tensor(temperature_init).log())

    def forward(self, z_a: torch.Tensor, z_b: torch.Tensor) -> dict[str, torch.Tensor]:
        za = F.normalize(self.proj(z_a), dim=-1)
        zb = F.normalize(self.proj(z_b), dim=-1)
        temp = self.temperature.exp().clamp(min=0.01, max=1.0)
        logits = za @ zb.T / temp
        labels = torch.arange(len(logits), device=logits.device)
        l_ab = F.cross_entropy(logits, labels)
        l_ba = F.cross_entropy(logits.T, labels)
        return {"loss": 0.5 * (l_ab + l_ba), "logits": logits}


class ContrastiveHead(nn.Module):
    def __init__(self, spec_dim: int, struct_dim: int, proj_dim: int = 128):
        super().__init__()
        self.spec_proj = nn.Sequential(
            nn.Linear(spec_dim, proj_dim), nn.GELU(), nn.Linear(proj_dim, proj_dim)
        )
        self.struct_proj = nn.Sequential(
            nn.Linear(struct_dim, proj_dim), nn.GELU(), nn.Linear(proj_dim, proj_dim)
        )
        self.temperature = nn.Parameter(torch.tensor(0.07).log())

    def forward(
        self, spec_embed: torch.Tensor, struct_embed: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        z_spec = F.normalize(self.spec_proj(spec_embed), dim=-1)
        z_struct = F.normalize(self.struct_proj(struct_embed), dim=-1)
        temp = self.temperature.exp().clamp(min=0.01, max=1.0)
        logits = z_spec @ z_struct.T / temp
        labels = torch.arange(len(logits), device=logits.device)
        loss_s2t = F.cross_entropy(logits, labels)
        loss_t2s = F.cross_entropy(logits.T, labels)
        loss = (loss_s2t + loss_t2s) / 2
        return {"loss": loss, "logits_per_spec": logits, "logits_per_struct": logits.T}
