from __future__ import annotations

from typing import Literal

import torch
from torch import nn
from torch_geometric.data import Batch as PyGBatch

from rose.config import RoseConfig
from rose.heads.contrastive import ContrastiveHead, PairContrastiveHead
from rose.heads.denoising import DenoisingHead
from rose.heads.forward_shifts import ForwardShiftHead
from rose.heads.peak_picking import PeakPickingHead
from rose.spectral_encoder import SpectralTransformerEncoder
from rose.structure_encoder import MolecularGNN, MolecularGraphTransformer

TaskName = Literal["denoise", "peak", "retrieval", "forward", "pair"]
AdaptMode = Literal["freeze_encoder", "unfreeze_encoder", "P1", "P2"]

_ADAPT_MODE_ALIASES: dict[str, Literal["freeze_encoder", "unfreeze_encoder"]] = {
    "freeze_encoder": "freeze_encoder",
    "unfreeze_encoder": "unfreeze_encoder",
    "P1": "freeze_encoder",
    "P2": "unfreeze_encoder",
}

_TASK_HEAD_ATTR: dict[TaskName, str] = {
    "denoise": "denoising_head",
    "peak": "peak_head",
    "retrieval": "contrastive_head",
    "forward": "forward_head",
    "pair": "pair_contrastive_head",
}


def build_structure_encoder(cfg: RoseConfig) -> nn.Module:
    st = cfg.structure_encoder
    if st.type == "graph_transformer":
        return MolecularGraphTransformer(
            hidden_dim=st.hidden_dim,
            num_layers=st.num_layers,
            num_heads=st.num_heads,
            dropout=st.dropout,
        )
    return MolecularGNN(hidden_dim=st.hidden_dim, num_layers=st.num_layers, dropout=st.dropout)


class RoseModel(nn.Module):
    def __init__(self, cfg: RoseConfig):
        super().__init__()
        self.cfg = cfg
        se = cfg.spectral_encoder
        heads = cfg.heads
        self.spectral_encoder = SpectralTransformerEncoder(
            spectrum_length=se.spectrum_length,
            patch_size=se.patch_size,
            embed_dim=se.embed_dim,
            num_layers=se.num_layers,
            num_heads=se.num_heads,
            mlp_ratio=se.mlp_ratio,
            dropout=se.dropout,
            num_metadata_tokens=se.num_metadata_tokens,
            drop_path_rate=se.drop_path_rate,
            solvent_vocab_size=se.solvent_vocab_size,
            conv_stem={
                "enabled": se.conv_stem.enabled,
                "channels": se.conv_stem.channels,
                "num_layers": se.conv_stem.num_layers,
                "kernel_size": se.conv_stem.kernel_size,
                "dilations": list(se.conv_stem.dilations),
            },
        )
        num_patches = se.spectrum_length // se.patch_size
        self.denoising_head = DenoisingHead(
            embed_dim=se.embed_dim, patch_size=se.patch_size, num_patches=num_patches
        )
        self.structure_encoder = build_structure_encoder(cfg)
        self.contrastive_head = ContrastiveHead(
            spec_dim=se.embed_dim,
            struct_dim=cfg.structure_encoder.hidden_dim,
            proj_dim=heads.contrastive_proj_dim,
        )
        self.pair_contrastive_head = PairContrastiveHead(
            spec_dim=se.embed_dim,
            proj_dim=heads.contrastive_proj_dim,
            temperature_init=heads.pair_temperature,
        )
        self.peak_head = PeakPickingHead(embed_dim=se.embed_dim, patch_size=se.patch_size)
        self.forward_head = ForwardShiftHead(
            struct_dim=cfg.structure_encoder.hidden_dim, max_peaks=heads.forward_max_peaks
        )

    def encode_spectrum(
        self,
        spectrum: torch.Tensor,
        ppm_axis: torch.Tensor,
        field_strength: torch.Tensor | None = None,
        solvent_id: torch.Tensor | None = None,
        field_strength_known: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        return self.spectral_encoder(
            spectrum,
            ppm_axis,
            field_strength,
            solvent_id,
            field_strength_known=field_strength_known,
        )

    def encode(
        self,
        spectrum: torch.Tensor,
        ppm_axis: torch.Tensor,
        field_strength: torch.Tensor | None = None,
        solvent_id: torch.Tensor | None = None,
        field_strength_known: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.encode_spectrum(
            spectrum,
            ppm_axis,
            field_strength,
            solvent_id,
            field_strength_known=field_strength_known,
        )["cls_embed"]

    def predict_denoise(
        self,
        noisy_spectrum: torch.Tensor,
        ppm_axis: torch.Tensor,
        field_strength: torch.Tensor | None = None,
        solvent_id: torch.Tensor | None = None,
        field_strength_known: torch.Tensor | None = None,
    ) -> torch.Tensor:
        enc = self.encode_spectrum(
            noisy_spectrum,
            ppm_axis,
            field_strength,
            solvent_id,
            field_strength_known=field_strength_known,
        )
        return self.denoising_head(enc["patch_embeds"])

    def predict_peaks(
        self,
        spectrum: torch.Tensor,
        ppm_axis: torch.Tensor,
        field_strength: torch.Tensor | None = None,
        solvent_id: torch.Tensor | None = None,
        field_strength_known: torch.Tensor | None = None,
    ) -> torch.Tensor:
        enc = self.encode_spectrum(
            spectrum,
            ppm_axis,
            field_strength,
            solvent_id,
            field_strength_known=field_strength_known,
        )
        return self.peak_head(enc["patch_embeds"])

    def predict_retrieval(
        self,
        spectrum: torch.Tensor,
        ppm_axis: torch.Tensor,
        mol_batch: PyGBatch,
        field_strength: torch.Tensor | None = None,
        solvent_id: torch.Tensor | None = None,
        field_strength_known: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        spec_cls = self.encode(
            spectrum,
            ppm_axis,
            field_strength,
            solvent_id,
            field_strength_known=field_strength_known,
        )
        dev = next(self.structure_encoder.parameters()).device
        struct_out = self.structure_encoder(mol_batch.to(dev))
        if isinstance(struct_out, dict):
            struct_out = struct_out["graph"]
        return self.contrastive_head(spec_cls, struct_out)

    def predict_forward(self, mol_batch: PyGBatch) -> dict[str, torch.Tensor]:
        dev = next(self.structure_encoder.parameters()).device
        graph_embed = self.structure_encoder(mol_batch.to(dev))
        if isinstance(graph_embed, dict):
            graph_embed = graph_embed["graph"]
        return self.forward_head(graph_embed)

    def predict_pair(self, cls_a: torch.Tensor, cls_b: torch.Tensor) -> dict[str, torch.Tensor]:
        return self.pair_contrastive_head(cls_a, cls_b)

    def head(self, task: TaskName) -> nn.Module:
        try:
            return getattr(self, _TASK_HEAD_ATTR[task])
        except KeyError as e:
            raise ValueError(f"unknown task: {task}") from e

    @staticmethod
    def _set_requires_grad(module: nn.Module, value: bool) -> None:
        for p in module.parameters():
            p.requires_grad = value

    def adapt(self, mode: AdaptMode, task: TaskName) -> RoseModel:
        """Configure freeze/unfreeze for adaptation. Does not run training.

        Modes:
        - ``freeze_encoder`` (paper P1): freeze encoders, train ``task`` head
        - ``unfreeze_encoder`` (paper P2): train encoders + ``task`` head
        """
        canonical = _ADAPT_MODE_ALIASES.get(mode)
        if canonical is None:
            raise ValueError(
                f"unknown adapt mode: {mode!r} "
                "(expected 'freeze_encoder', 'unfreeze_encoder', or aliases 'P1'/'P2')"
            )
        head = self.head(task)
        self.train()
        self._set_requires_grad(self, False)
        if canonical == "unfreeze_encoder":
            self._set_requires_grad(self.spectral_encoder, True)
            self._set_requires_grad(self.structure_encoder, True)
        self._set_requires_grad(head, True)
        head.train()
        self._adapt_mode = canonical
        self._adapt_task = task
        return self

    def param_groups(
        self, *, lr_head: float = 1e-3, lr_encoder: float = 1e-5
    ) -> list[dict]:
        mode = getattr(self, "_adapt_mode", None)
        task = getattr(self, "_adapt_task", None)
        if mode is None or task is None:
            raise ValueError("call adapt(...) before param_groups()")
        head = self.head(task)
        groups: list[dict] = []
        if mode == "unfreeze_encoder":
            enc_params = [
                *self.spectral_encoder.parameters(),
                *self.structure_encoder.parameters(),
            ]
            groups.append({"params": enc_params, "lr": lr_encoder})
        groups.append({"params": list(head.parameters()), "lr": lr_head})
        return groups

    def print_trainable_parameters(self) -> None:
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.parameters())
        pct = 100.0 * trainable / total if total else 0.0
        print(f"trainable params: {trainable} || all params: {total} || trainable%: {pct:.4f}")

    def run_task(
        self,
        task: TaskName,
        *,
        spectrum: torch.Tensor | None = None,
        ppm_axis: torch.Tensor | None = None,
        mol_batch: PyGBatch | None = None,
        spectrum_b: torch.Tensor | None = None,
        ppm_axis_b: torch.Tensor | None = None,
        field_strength: torch.Tensor | None = None,
        solvent_id: torch.Tensor | None = None,
        field_strength_known: torch.Tensor | None = None,
    ) -> torch.Tensor | dict[str, torch.Tensor]:
        meta = {
            "field_strength": field_strength,
            "solvent_id": solvent_id,
            "field_strength_known": field_strength_known,
        }
        if task == "denoise":
            if spectrum is None or ppm_axis is None:
                raise ValueError("task='denoise' requires spectrum= and ppm_axis=")
            return self.predict_denoise(spectrum, ppm_axis, **meta)
        if task == "peak":
            if spectrum is None or ppm_axis is None:
                raise ValueError("task='peak' requires spectrum= and ppm_axis=")
            return self.predict_peaks(spectrum, ppm_axis, **meta)
        if task == "retrieval":
            if spectrum is None or ppm_axis is None or mol_batch is None:
                raise ValueError("task='retrieval' requires spectrum=, ppm_axis=, mol_batch=")
            return self.predict_retrieval(spectrum, ppm_axis, mol_batch, **meta)
        if task == "forward":
            if mol_batch is None:
                raise ValueError("task='forward' requires mol_batch=")
            return self.predict_forward(mol_batch)
        if task == "pair":
            if spectrum is None or spectrum_b is None or ppm_axis is None or ppm_axis_b is None:
                raise ValueError("task='pair' requires spectrum=, spectrum_b=, ppm_axis=, ppm_axis_b=")
            cls_a = self.encode(spectrum, ppm_axis, **meta)
            cls_b = self.encode(spectrum_b, ppm_axis_b, **meta)
            return self.predict_pair(cls_a, cls_b)
        raise ValueError(f"unknown task: {task}")
