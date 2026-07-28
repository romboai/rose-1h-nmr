from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

try:
    from torch_geometric.data import Batch, Data
    from torch_geometric.nn import GINEConv, TransformerConv, global_mean_pool

    HAS_PYG = True
except ImportError:
    HAS_PYG = False
ATOM_FEATURES = {
    "atomic_num": list(range(1, 119)),
    "degree": [0, 1, 2, 3, 4, 5, 6],
    "formal_charge": [-3, -2, -1, 0, 1, 2, 3],
    "hybridization": [0, 1, 2, 3, 4, 5],
    "num_hs": [0, 1, 2, 3, 4],
    "is_aromatic": [False, True],
}
BOND_FEATURES = {
    "bond_type": [1, 2, 3, 12],
    "is_conjugated": [False, True],
    "is_in_ring": [False, True],
}
ATOM_FEAT_DIM = sum(len(v) for v in ATOM_FEATURES.values())
BOND_FEAT_DIM = sum(len(v) for v in BOND_FEATURES.values())


def _one_hot(value, choices: list) -> list[float]:
    enc = [0.0] * len(choices)
    try:
        enc[choices.index(value)] = 1.0
    except ValueError:
        pass
    return enc


def smiles_to_pyg_data(smiles: str) -> Data:
    if not HAS_PYG:
        raise ImportError("torch_geometric is required for the structure encoder")
    from rdkit import Chem

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"RDKit could not parse SMILES: {smiles}")
    atom_feats = []
    for atom in mol.GetAtoms():
        feat = (
            _one_hot(atom.GetAtomicNum(), ATOM_FEATURES["atomic_num"])
            + _one_hot(atom.GetDegree(), ATOM_FEATURES["degree"])
            + _one_hot(atom.GetFormalCharge(), ATOM_FEATURES["formal_charge"])
            + _one_hot(int(atom.GetHybridization()), ATOM_FEATURES["hybridization"])
            + _one_hot(atom.GetTotalNumHs(), ATOM_FEATURES["num_hs"])
            + _one_hot(atom.GetIsAromatic(), ATOM_FEATURES["is_aromatic"])
        )
        atom_feats.append(feat)
    x = torch.tensor(atom_feats, dtype=torch.float)
    edge_index = []
    edge_attr = []
    for bond in mol.GetBonds():
        i, j = (bond.GetBeginAtomIdx(), bond.GetEndAtomIdx())
        feat = (
            _one_hot(int(bond.GetBondTypeAsDouble()), BOND_FEATURES["bond_type"])
            + _one_hot(bond.GetIsConjugated(), BOND_FEATURES["is_conjugated"])
            + _one_hot(bond.IsInRing(), BOND_FEATURES["is_in_ring"])
        )
        edge_index.extend([[i, j], [j, i]])
        edge_attr.extend([feat, feat])
    if edge_index:
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_attr, dtype=torch.float)
    else:
        edge_index = torch.zeros((2, 0), dtype=torch.long)
        edge_attr = torch.zeros((0, BOND_FEAT_DIM), dtype=torch.float)
    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)


class MolecularGNN(nn.Module):
    def __init__(
        self,
        atom_feat_dim: int = ATOM_FEAT_DIM,
        bond_feat_dim: int = BOND_FEAT_DIM,
        hidden_dim: int = 256,
        num_layers: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        if not HAS_PYG:
            raise ImportError("torch_geometric is required for MolecularGNN")
        self.atom_proj = nn.Linear(atom_feat_dim, hidden_dim)
        self.bond_proj = nn.Linear(bond_feat_dim, hidden_dim)
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        for _ in range(num_layers):
            mlp = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, hidden_dim)
            )
            self.convs.append(GINEConv(mlp, edge_dim=hidden_dim))
            self.norms.append(nn.LayerNorm(hidden_dim))
        self.dropout = nn.Dropout(dropout)
        self.embed_dim = hidden_dim

    def forward(
        self, batch: Batch, return_nodes: bool = False
    ) -> torch.Tensor | dict[str, torch.Tensor]:
        x = self.atom_proj(batch.x)
        edge_attr = self.bond_proj(batch.edge_attr)
        for conv, norm in zip(self.convs, self.norms):
            h = conv(x, batch.edge_index, edge_attr)
            h = norm(h)
            h = F.gelu(h)
            h = self.dropout(h)
            x = x + h
        graph_embed = global_mean_pool(x, batch.batch)
        if not return_nodes:
            return graph_embed
        nodes_padded, mask = _pad_node_embeddings(x, batch.batch)
        return {"graph": graph_embed, "nodes": nodes_padded, "mask": mask}

    def get_embed_dim(self) -> int:
        return self.embed_dim


def _pad_node_embeddings(
    node_embeds: torch.Tensor, batch_idx: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    if node_embeds.numel() == 0:
        return (
            node_embeds.new_zeros(1, 1, node_embeds.size(-1)),
            torch.ones(1, 1, dtype=torch.bool, device=node_embeds.device),
        )
    B = int(batch_idx.max().item()) + 1
    counts = torch.bincount(batch_idx, minlength=B)
    max_atoms = int(counts.max().item())
    D = node_embeds.size(-1)
    out = node_embeds.new_zeros(B, max_atoms, D)
    mask = torch.ones(B, max_atoms, dtype=torch.bool, device=node_embeds.device)
    cum = 0
    for b in range(B):
        n = int(counts[b].item())
        if n == 0:
            continue
        out[b, :n] = node_embeds[cum : cum + n]
        mask[b, :n] = False
        cum += n
    return (out, mask)


class MolecularGraphTransformer(nn.Module):
    def __init__(
        self,
        atom_feat_dim: int = ATOM_FEAT_DIM,
        bond_feat_dim: int = BOND_FEAT_DIM,
        hidden_dim: int = 256,
        num_layers: int = 4,
        num_heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        if not HAS_PYG:
            raise ImportError("torch_geometric is required for MolecularGraphTransformer")
        self.atom_proj = nn.Linear(atom_feat_dim, hidden_dim)
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        head_dim = hidden_dim // num_heads
        for _ in range(num_layers):
            self.convs.append(
                TransformerConv(
                    in_channels=hidden_dim,
                    out_channels=head_dim,
                    heads=num_heads,
                    edge_dim=bond_feat_dim,
                    dropout=dropout,
                    concat=True,
                )
            )
            self.norms.append(nn.LayerNorm(hidden_dim))
        self.dropout = nn.Dropout(dropout)
        self.embed_dim = hidden_dim

    def forward(
        self, batch: Batch, return_nodes: bool = False
    ) -> torch.Tensor | dict[str, torch.Tensor]:
        x = self.atom_proj(batch.x)
        for conv, norm in zip(self.convs, self.norms):
            h = conv(x, batch.edge_index, batch.edge_attr)
            h = norm(h)
            h = F.gelu(h)
            h = self.dropout(h)
            x = x + h
        graph_embed = global_mean_pool(x, batch.batch)
        if not return_nodes:
            return graph_embed
        nodes_padded, mask = _pad_node_embeddings(x, batch.batch)
        return {"graph": graph_embed, "nodes": nodes_padded, "mask": mask}

    def get_embed_dim(self) -> int:
        return self.embed_dim
