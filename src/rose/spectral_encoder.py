from __future__ import annotations

import torch
from torch import nn


class DropPath(nn.Module):
    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or self.drop_prob == 0.0:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor = torch.floor_(random_tensor + keep_prob)
        return x / keep_prob * random_tensor


class SinusoidalPpmEncoding(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        inv_freq = 1.0 / 10000 ** (torch.arange(0, d_model, 2).float() / d_model)
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, ppm_values: torch.Tensor) -> torch.Tensor:
        ppm = ppm_values.unsqueeze(-1)
        freqs = ppm * self.inv_freq.unsqueeze(0).unsqueeze(0)
        return torch.cat([freqs.sin(), freqs.cos()], dim=-1)


class ConvStem1D(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        patch_size: int,
        *,
        channels: int = 32,
        num_layers: int = 3,
        kernel_size: int = 7,
        dilations: list[int] | None = None,
    ):
        super().__init__()
        if dilations is None:
            dilations = [2**i for i in range(num_layers)]
        layers: list[nn.Module] = []
        in_ch = 1
        for d in dilations:
            pad = (kernel_size - 1) // 2 * d
            layers.append(nn.Conv1d(in_ch, channels, kernel_size, padding=pad, dilation=d))
            layers.append(nn.GELU())
            layers.append(nn.GroupNorm(num_groups=min(8, channels), num_channels=channels))
            in_ch = channels
        self.body = nn.Sequential(*layers)
        self.patch_pool = nn.AvgPool1d(kernel_size=patch_size, stride=patch_size)
        self.proj = nn.Linear(channels, embed_dim)

    def forward(self, spectrum: torch.Tensor) -> torch.Tensor:
        x = spectrum.unsqueeze(1)
        x = self.body(x)
        x = self.patch_pool(x).transpose(1, 2)
        return self.proj(x)


class PatchEmbedding1D(nn.Module):
    def __init__(self, patch_size: int, embed_dim: int):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Linear(patch_size, embed_dim)

    def forward(self, spectrum: torch.Tensor) -> torch.Tensor:
        b, length = spectrum.shape
        n_patches = length // self.patch_size
        x = spectrum[:, : n_patches * self.patch_size].reshape(b, n_patches, self.patch_size)
        return self.proj(x)


class SpectralTransformerEncoder(nn.Module):
    def __init__(
        self,
        spectrum_length: int = 4096,
        patch_size: int = 32,
        embed_dim: int = 256,
        num_layers: int = 8,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
        num_metadata_tokens: int = 4,
        drop_path_rate: float = 0.1,
        *,
        solvent_vocab_size: int = 32,
        conv_stem: dict | None = None,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.num_patches = spectrum_length // patch_size
        self.patch_embed = PatchEmbedding1D(patch_size, embed_dim)
        cs = conv_stem or {}
        if cs.get("enabled", False):
            self.conv_stem = ConvStem1D(
                embed_dim=embed_dim,
                patch_size=patch_size,
                channels=int(cs.get("channels", 32)),
                num_layers=int(cs.get("num_layers", 3)),
                kernel_size=int(cs.get("kernel_size", 7)),
                dilations=list(cs.get("dilations", [1, 2, 4])),
            )
        else:
            self.conv_stem = None
        self.ppm_encoding = SinusoidalPpmEncoding(embed_dim)
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)
        self.metadata_tokens = nn.Parameter(torch.randn(1, num_metadata_tokens, embed_dim) * 0.02)
        self.num_metadata_tokens = num_metadata_tokens
        self.field_strength_proj = nn.Linear(1, embed_dim)
        self.field_unknown_embed = nn.Parameter(torch.randn(embed_dim) * 0.02)
        self.solvent_embed = nn.Embedding(int(solvent_vocab_size), embed_dim)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, num_layers)]
        self.layers = nn.ModuleList()
        self.drop_paths = nn.ModuleList()
        for i in range(num_layers):
            self.layers.append(
                nn.TransformerEncoderLayer(
                    d_model=embed_dim,
                    nhead=num_heads,
                    dim_feedforward=int(embed_dim * mlp_ratio),
                    dropout=dropout,
                    activation="gelu",
                    batch_first=True,
                    norm_first=True,
                )
            )
            self.drop_paths.append(DropPath(dpr[i]))
        self.norm = nn.LayerNorm(embed_dim)

    def forward(
        self,
        spectrum: torch.Tensor,
        ppm_axis: torch.Tensor,
        field_strength: torch.Tensor | None = None,
        solvent_id: torch.Tensor | None = None,
        field_strength_known: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        b = spectrum.shape[0]
        patch_embeds = self.patch_embed(spectrum)
        if self.conv_stem is not None:
            patch_embeds = patch_embeds + self.conv_stem(spectrum)
        ppm_patches = ppm_axis[:, : self.num_patches * self.patch_size]
        ppm_patches = ppm_patches.reshape(b, self.num_patches, self.patch_size)
        ppm_centers = ppm_patches.mean(dim=-1)
        patch_embeds = patch_embeds + self.ppm_encoding(ppm_centers)
        cls = self.cls_token.expand(b, -1, -1)
        meta = self.metadata_tokens.expand(b, -1, -1).clone()
        if field_strength is not None:
            if field_strength_known is None:
                meta[:, 0, :] = meta[:, 0, :] + self.field_strength_proj(field_strength)
            else:
                k = field_strength_known.to(dtype=field_strength.dtype)
                if k.dim() == 1:
                    k = k.unsqueeze(-1)
                k = k.clamp(0.0, 1.0)
                proj = self.field_strength_proj(field_strength)
                unk = self.field_unknown_embed.unsqueeze(0).expand(b, -1)
                meta[:, 0, :] = meta[:, 0, :] + proj * k + unk * (1.0 - k)
        if solvent_id is not None:
            meta[:, 1, :] = meta[:, 1, :] + self.solvent_embed(solvent_id)
        tokens = torch.cat([cls, meta, patch_embeds], dim=1)
        for layer, drop_path in zip(self.layers, self.drop_paths):
            residual = tokens
            tokens = layer(tokens)
            tokens = residual + drop_path(tokens - residual)
        tokens = self.norm(tokens)
        offset = 1 + self.num_metadata_tokens
        return {
            "cls_embed": tokens[:, 0, :],
            "patch_embeds": tokens[:, offset:, :],
            "all_embeds": tokens,
        }

    def get_embed_dim(self) -> int:
        return self.embed_dim
