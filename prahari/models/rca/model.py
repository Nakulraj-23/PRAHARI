"""Pure-PyTorch residual GCN for root-cause analysis (no torch-geometric)."""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

HIDDEN = 64


def normalised_adjacency(adj: np.ndarray) -> np.ndarray:
    """Return A_hat = D^-1/2 (A + I) D^-1/2 for a 0/1 symmetric adjacency matrix."""
    a = np.asarray(adj, dtype=np.float32) + np.eye(len(adj), dtype=np.float32)
    d = a.sum(axis=1)
    dinv = 1.0 / np.sqrt(d)
    return (dinv[:, None] * a * dinv[None, :]).astype(np.float32)


class RCAGNN(nn.Module):
    """3-layer residual GCN, node head (root logits) and pooled graph head (fault type).

    forward(x, a_hat): x (B, N, F) raw features, a_hat (N, N) -> (root_logits (B, N), fault (B, K)).
    Input standardisation (signed log1p, then mean/std) lives in buffers so ONNX takes raw X.
    ``use_graph=False`` skips message passing (MLP baseline: same parameters, no graph).
    """

    def __init__(
        self,
        n_features: int,
        n_faults: int = 5,
        hidden: int = HIDDEN,
        dropout: float = 0.2,
        use_graph: bool = True,
    ):
        super().__init__()
        self.use_graph = use_graph
        self.register_buffer("mu", torch.zeros(n_features))
        self.register_buffer("sd", torch.ones(n_features))
        self.inp = nn.Linear(n_features, hidden)
        self.layers = nn.ModuleList([nn.Linear(hidden, hidden) for _ in range(3)])
        self.drop = nn.Dropout(dropout)
        self.root_head = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, 1))
        self.fault_head = nn.Sequential(
            nn.Linear(2 * hidden, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, n_faults),
        )

    @staticmethod
    def transform(x: torch.Tensor) -> torch.Tensor:
        return torch.sign(x) * torch.log1p(torch.abs(x))

    def set_norm(self, x: np.ndarray) -> None:
        """Fit standardisation on (M, N, F) training features."""
        t = self.transform(torch.as_tensor(x, dtype=torch.float32)).reshape(-1, x.shape[-1])
        self.mu.copy_(t.mean(0))
        self.sd.copy_(t.std(0).clamp_min(1e-3))

    def forward(self, x: torch.Tensor, a_hat: torch.Tensor):
        h = (self.transform(x) - self.mu) / self.sd
        h = torch.relu(self.inp(h))
        for lin in self.layers:
            m = lin(h)
            if self.use_graph:
                m = torch.matmul(a_hat, m)
            h = h + torch.relu(m)  # residual
            h = self.drop(h)
        root = self.root_head(h).squeeze(-1)
        g = torch.cat([h.mean(dim=1), h.max(dim=1).values], dim=-1)
        return root, self.fault_head(g)


class OnnxWrapper(nn.Module):
    """Single-graph export view: (x (N, F), a_hat (N, N)) -> (root_logits (N,), fault_logits (K,))."""

    def __init__(self, m: RCAGNN):
        super().__init__()
        self.m = m

    def forward(self, x, a_hat):
        r, f = self.m(x.unsqueeze(0), a_hat)
        return r[0], f[0]
