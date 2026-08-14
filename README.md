---
license: apache-2.0
library_name: rose-1h-nmr
tags:
  - nmr
  - chemistry
  - foundation-model
  - pytorch
pipeline_tag: feature-extraction
---

# ROSE-1H NMR

Pretrained **¹H NMR** foundation model — inference and paper adaptation protocols.

**Paper:** preprint forthcoming.  
**Weights:** [`romboai/rose-1h-nmr`](https://huggingface.co/romboai/rose-1h-nmr) (Hugging Face).  
**Code:** this repo.

## Installation

```bash
pip install -e .
# weights from Hugging Face:
pip install -e ".[hub]"
```

## Quick start

Spectra are `float32` arrays of shape `(4096,)` or `(B, 4096)` on a linear **0–14 ppm** grid.  
`field_mhz` and `solvent_id` are optional (`solvent_id` → `configs/solvent_vocab.json`).

```python
import numpy as np
from rose import load, encode, predict

model = load()                         # default: romboai/rose-1h-nmr
# model = load("path/to/best_model.pt")

spectrum = np.load("spectrum.npy")     # (4096,)
z = encode(model, spectrum, field_mhz=400.0, solvent_id=3)  # cdcl3 → (B, 256)
```

## Task heads

```python
import numpy as np
from rose import load, predict

model = load()
spectrum = np.load("spectrum.npy")
noisy = spectrum + 0.01 * np.random.randn(*spectrum.shape).astype(np.float32)

# denoise — clean spectrum from noisy input (zero-shot)
clean = predict(model, noisy, task="denoise")
# (B, 4096)

# peak — peak probability per grid point
probs = predict(model, spectrum, task="peak")
# (B, 4096) in [0, 1]

# retrieval — contrastive spectrum ↔ SMILES (one SMILES per spectrum)
out = predict(model, spectrum, task="retrieval", smiles="CCO")
# logits_per_spec (B, B), logits_per_struct (B, B), loss

# forward — ¹H shifts from structure (SMILES; spectrum sets batch size)
out = predict(model, spectrum, task="forward", smiles="CCO")
# shifts_pred (B, 32), count_pred (B,)

# pair — similarity of aligned spectrum pairs (spectrum[i] ↔ spectrum_b[i])
out = predict(model, spectrum, task="pair", spectrum_b=spectrum)
# similarity (B,), logits (B, B)
```

## Fine-tuning

Zero-shot uses pretrained weights as-is. For adaptation, load with `eval_mode=False`, then `adapt` + `param_groups`:

- `freeze_encoder` (alias `P1`) — train task head only; easy domains
- `unfreeze_encoder` (alias `P2`) — train encoders + head; hard domains

**P1 — peak (frozen encoder)**

```python
import numpy as np
import torch
import torch.nn.functional as F
from rose import load
from rose.grid import ppm_axis_tensor

model = load("best_model.pt", eval_mode=False)
model.adapt(mode="freeze_encoder", task="peak")
model.print_trainable_parameters()
opt = torch.optim.Adam(model.param_groups())  # lr_head=1e-3

device = next(model.parameters()).device
spectrum = torch.from_numpy(np.load("spectrum.npy")).unsqueeze(0).to(device)
ppm = ppm_axis_tensor(spectrum.size(0), device=device, cfg=model.cfg)
peak_labels = torch.zeros_like(spectrum)  # (1, 4096) in [0, 1]

logits = model.run_task("peak", spectrum=spectrum, ppm_axis=ppm)
loss = F.binary_cross_entropy_with_logits(logits, peak_labels)
loss.backward()
opt.step()
```

**P2 — retrieval (unfreeze encoder)**

```python
import numpy as np
import torch
from rose import load, smiles_batch
from rose.grid import ppm_axis_tensor

model = load("best_model.pt", eval_mode=False)
model.adapt(mode="unfreeze_encoder", task="retrieval")
model.print_trainable_parameters()
opt = torch.optim.Adam(model.param_groups())  # lr_head=1e-3, lr_encoder=1e-5

# contrastive loss needs batch size ≥ 2
device = next(model.parameters()).device
spectrum = torch.from_numpy(
    np.stack([np.load("spectrum_a.npy"), np.load("spectrum_b.npy")])
).to(device)
ppm = ppm_axis_tensor(spectrum.size(0), device=device, cfg=model.cfg)
mol = smiles_batch(["CCO", "c1ccccc1"]).to(device)

out = model.run_task("retrieval", spectrum=spectrum, ppm_axis=ppm, mol_batch=mol)
loss = out["loss"]
loss.backward()
opt.step()
```

## Results

Headline numbers from the ROSE paper (ROSE-Pretrain-L, 7.8M parameters, 3.2M pretrain spectra, InChIKey-14 holdout).

**Internal** — native heads as pretrained, identity-disjoint held-out test (same corpus, no target-domain adaptation):

| Task | Metric | ROSE |
|------|--------|------|
| Denoise | ΔSNR_peak @ 10 dB / cosine | 16.1 / 0.95 |
| Pair | Top-1 | 87.8% |
| Retrieval | Top-1 | 79.4% |
| Forward | Chamfer (+ count) ↓ | 1.89 |
| Peak | F1 @ 0.05 ppm | 0.80 |

Low-field slice ($B_0$ ≤ 100 MHz) is weaker on structure-linked heads (retrieval 45.2%, peak F1 0.20); denoise cosine stays high (0.99).

**External** — comparisons use the **same protocol** as ROSE, not a published stack under different retrieval machinery:

| Benchmark | Protocol | Metric | ROSE | Same-protocol baseline |
|-----------|----------|--------|------|------------------------|
| QIB edible oils (60 MHz) | frozen encoder + linear head | balanced accuracy | 98.8% | 98.9% PLS-DA (*accuracy*, not BA) |
| NMRNet structure→spectrum | frozen encoder + head | Chamfer ↓ | 0.79 | 1.29 Morgan+Ridge |
| NMRformer peak detection | frozen encoder + head | F1 | 69% | — |
| NMR-Solver retrieval | 5-epoch P2, gallery ≈30k | Top-1 / Top-10 | 37.1% ± 2.1% / 67.7% ± 3.6% | scratch 0% / 0% |

NMR-Solver literature (52.9% / 67.3% Top-1 / Top-10) uses a ≈10⁸ gallery plus FAISS HNSW and set-similarity rerank — not like-for-like; Top-10 at our 30k gallery already matches theirs.

Weights: Hugging Face [`romboai/rose-1h-nmr`](https://huggingface.co/romboai/rose-1h-nmr).

## Citation

```bibtex
@article{diiorio2026rose,
  title   = {{ROSE}: a Foundation Model for Reusable One-dimensional
             Spectrum Embeddings in $^1$H~{NMR}},
  author  = {Di Iorio, Mattia and Mattia, Carmine and Zanda, Andrea
             and Atzori, Maurizio},
  year    = {2026},
  note    = {Paper link forthcoming},
  url     = {https://github.com/romboai/rose-1h-nmr}
}
```

## Repository layout

| Path | Role |
|------|------|
| `src/rose/` | Library — model, encoders, API, task heads |
| `configs/` | `rose.yaml`, `solvent_vocab.json` |
| `hub/` | Hugging Face metadata (`config.json`) |
| `scripts/` | Maintainer utilities (HF upload staging) |
| `indices/pretrain/` | Pretrain split policy (`pretrain_l_splits.meta.json`; ID lists not shipped) |
| `indices/holdout/` | Paper test holdout (IK14), excluded from pretrain |
| `indices/benchmarks/` | Eval splits (NMRBank, NMR-Solver) and literature holdouts |

Indices hold **IDs only** (no spectra). Each `.meta.json` documents format and split policy.
