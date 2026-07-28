from __future__ import annotations

import torch
from torch import nn


class DenoisingHead(nn.Module):
    def __init__(self, embed_dim: int = 256, patch_size: int = 64, num_patches: int = 128):
        super().__init__()
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.decoder = nn.Sequential(
            nn.Linear(embed_dim, embed_dim), nn.GELU(), nn.Linear(embed_dim, patch_size)
        )

    def forward(self, patch_embeds: torch.Tensor) -> torch.Tensor:
        B, N, _ = patch_embeds.shape
        decoded_patches = self.decoder(patch_embeds)
        return decoded_patches.reshape(B, N * self.patch_size)
