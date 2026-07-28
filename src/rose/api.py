from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch_geometric.data import Batch as PyGBatch

from rose.checkpoint import load_state_dict_relaxed
from rose.config import load_config
from rose.grid import ppm_axis_tensor
from rose.hub import DEFAULT_REPO_ID, is_hub_id, resolve_checkpoint, resolve_config
from rose.model import RoseModel, TaskName
from rose.structure_encoder import smiles_to_pyg_data


def _as_tensor(spectrum: np.ndarray | torch.Tensor, *, device: torch.device) -> torch.Tensor:
    if isinstance(spectrum, np.ndarray):
        t = torch.from_numpy(np.asarray(spectrum, dtype=np.float32))
    else:
        t = spectrum.to(dtype=torch.float32)
    if t.dim() == 1:
        t = t.unsqueeze(0)
    return t.to(device)


def _field_tensor(
    field_mhz: float | None, batch_size: int, device: torch.device
) -> torch.Tensor | None:
    if field_mhz is None:
        return None
    return torch.full((batch_size, 1), float(field_mhz), dtype=torch.float32, device=device)


def _solvent_tensor(
    solvent_id: int | None, batch_size: int, device: torch.device
) -> torch.Tensor | None:
    if solvent_id is None:
        return None
    return torch.full((batch_size,), int(solvent_id), dtype=torch.long, device=device)


def load(
    checkpoint: str | Path | None = None,
    *,
    config: str | Path | None = None,
    device: str | torch.device | None = None,
    eval_mode: bool = True,
    revision: str | None = None,
    cache_dir: str | Path | None = None,
) -> RoseModel:
    ckpt_path = resolve_checkpoint(checkpoint, revision=revision, cache_dir=cache_dir)
    hub_repo = (
        str(checkpoint)
        if checkpoint is not None and is_hub_id(checkpoint)
        else (DEFAULT_REPO_ID if checkpoint is None else None)
    )
    cfg_path = resolve_config(
        config,
        repo_id=hub_repo,
        revision=revision,
        cache_dir=cache_dir,
    )
    cfg = load_config(cfg_path)
    model = RoseModel(cfg)
    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    payload = torch.load(ckpt_path, map_location=dev, weights_only=False)
    state = (
        payload["model_state_dict"]
        if isinstance(payload, dict) and "model_state_dict" in payload
        else payload
    )
    load_state_dict_relaxed(model, state)
    model.to(dev)
    if eval_mode:
        model.eval()
    return model


def encode(
    model: RoseModel,
    spectrum: np.ndarray | torch.Tensor,
    *,
    field_mhz: float | None = None,
    solvent_id: int | None = None,
    field_known: bool = True,
) -> np.ndarray:
    device = next(model.parameters()).device
    spec = _as_tensor(spectrum, device=device)
    ppm = ppm_axis_tensor(spec.size(0), device=device, cfg=model.cfg)
    field = _field_tensor(field_mhz, spec.size(0), device)
    solvent = _solvent_tensor(solvent_id, spec.size(0), device)
    known = None
    if field is not None:
        known = torch.full((spec.size(0), 1), float(field_known), device=device)
    with torch.no_grad():
        z = model.encode(spec, ppm, field, solvent, field_strength_known=known)
    return z.cpu().numpy()


def smiles_batch(smiles: list[str]) -> PyGBatch:
    graphs = [smiles_to_pyg_data(s) for s in smiles]
    return PyGBatch.from_data_list(graphs)


def predict(
    model: RoseModel,
    spectrum: np.ndarray | torch.Tensor,
    task: TaskName,
    *,
    smiles: str | list[str] | None = None,
    spectrum_b: np.ndarray | torch.Tensor | None = None,
    field_mhz: float | None = None,
    solvent_id: int | None = None,
    field_known: bool = True,
) -> np.ndarray | dict:
    device = next(model.parameters()).device
    spec = _as_tensor(spectrum, device=device)
    ppm = ppm_axis_tensor(spec.size(0), device=device, cfg=model.cfg)
    field = _field_tensor(field_mhz, spec.size(0), device)
    solvent = _solvent_tensor(solvent_id, spec.size(0), device)
    known = None
    if field is not None:
        known = torch.full((spec.size(0), 1), float(field_known), device=device)
    with torch.no_grad():
        if task == "denoise":
            out = model.run_task(
                task,
                spectrum=spec,
                ppm_axis=ppm,
                field_strength=field,
                solvent_id=solvent,
                field_strength_known=known,
            )
            return out.cpu().numpy()
        if task == "peak":
            logits = model.run_task(
                task,
                spectrum=spec,
                ppm_axis=ppm,
                field_strength=field,
                solvent_id=solvent,
                field_strength_known=known,
            )
            return torch.sigmoid(logits).cpu().numpy()
        if task == "retrieval":
            if not smiles:
                raise ValueError("task='retrieval' requires smiles=")
            smi = [smiles] if isinstance(smiles, str) else smiles
            if len(smi) != spec.size(0):
                raise ValueError("len(smiles) must match batch size")
            mol = smiles_batch(smi).to(device)
            out = model.run_task(
                task,
                spectrum=spec,
                ppm_axis=ppm,
                mol_batch=mol,
                field_strength=field,
                solvent_id=solvent,
                field_strength_known=known,
            )
            return {k: (v.cpu().numpy() if torch.is_tensor(v) else v) for k, v in out.items()}
        if task == "forward":
            if not smiles:
                raise ValueError("task='forward' requires smiles=")
            smi = [smiles] if isinstance(smiles, str) else smiles
            mol = smiles_batch(smi).to(device)
            out = model.run_task(task, mol_batch=mol)
            return {k: v.cpu().numpy() for k, v in out.items()}
        if task == "pair":
            if spectrum_b is None:
                raise ValueError("task='pair' requires spectrum_b=")
            spec_b = _as_tensor(spectrum_b, device=device)
            if spec_b.size(0) != spec.size(0):
                raise ValueError("spectrum and spectrum_b must have the same batch size")
            ppm_b = ppm_axis_tensor(spec_b.size(0), device=device, cfg=model.cfg)
            out = model.run_task(
                task,
                spectrum=spec,
                ppm_axis=ppm,
                spectrum_b=spec_b,
                ppm_axis_b=ppm_b,
                field_strength=field,
                solvent_id=solvent,
                field_strength_known=known,
            )
            logits = out["logits"]
            return {
                "similarity": logits.diagonal().cpu().numpy(),
                "logits": logits.cpu().numpy(),
            }
    raise ValueError(f"unknown task: {task}")
