from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass
class GridConfig:
    ppm_min: float = 0.0
    ppm_max: float = 14.0
    num_points: int = 4096


@dataclass
class ConvStemConfig:
    enabled: bool = True
    channels: int = 32
    num_layers: int = 3
    kernel_size: int = 7
    dilations: list[int] = field(default_factory=lambda: [1, 2, 4])


@dataclass
class SpectralEncoderConfig:
    spectrum_length: int = 4096
    patch_size: int = 32
    embed_dim: int = 256
    num_layers: int = 8
    num_heads: int = 8
    mlp_ratio: float = 4.0
    dropout: float = 0.1
    drop_path_rate: float = 0.1
    num_metadata_tokens: int = 4
    solvent_vocab_size: int = 32
    conv_stem: ConvStemConfig = field(default_factory=ConvStemConfig)


@dataclass
class StructureEncoderConfig:
    type: str = "graph_transformer"
    hidden_dim: int = 256
    num_layers: int = 4
    num_heads: int = 4
    dropout: float = 0.1


@dataclass
class HeadsConfig:
    contrastive_proj_dim: int = 128
    pair_temperature: float = 0.1
    forward_max_peaks: int = 32


@dataclass
class RoseConfig:
    grid: GridConfig = field(default_factory=GridConfig)
    spectral_encoder: SpectralEncoderConfig = field(default_factory=SpectralEncoderConfig)
    structure_encoder: StructureEncoderConfig = field(default_factory=StructureEncoderConfig)
    heads: HeadsConfig = field(default_factory=HeadsConfig)


def _merge_dataclass(obj, data: dict) -> None:
    for key, val in data.items():
        if not hasattr(obj, key):
            continue
        cur = getattr(obj, key)
        if hasattr(cur, "__dataclass_fields__") and isinstance(val, dict):
            _merge_dataclass(cur, val)
        else:
            setattr(obj, key, val)


def load_config(path: str | Path | None = None) -> RoseConfig:
    if path is None:
        path = Path(__file__).resolve().parents[2] / "configs" / "rose.yaml"
    path = Path(path)
    with open(path) as f:
        raw = yaml.safe_load(f) or {}
    cfg = RoseConfig()
    _merge_dataclass(cfg, raw)
    return cfg
