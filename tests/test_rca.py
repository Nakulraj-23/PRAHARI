"""Unit tests for the RCA GNN (toy graphs; no database needed)."""

import numpy as np
import torch

from prahari.models.rca.model import RCAGNN, normalised_adjacency
from prahari.models.rca.train import export_onnx, onnx_parity
from prahari.pipeline.features import feature_names

F = len(feature_names())


def _toy():
    a = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=np.float32)  # path 0-1-2
    return a


def test_a_hat_symmetric_and_normalised():
    a = _toy()
    ah = normalised_adjacency(a)
    assert np.allclose(ah, ah.T)
    deg = np.array([2.0, 3.0, 2.0])  # degrees of A + I
    expect = (a + np.eye(3)) / np.sqrt(deg[:, None] * deg[None, :])
    assert np.allclose(ah, expect, atol=1e-6)
    assert np.isclose(ah[0, 0], 0.5) and np.isclose(ah[1, 1], 1 / 3)


def _fake_set(n=8, nodes=12, seed=0):
    rng = np.random.default_rng(seed)
    a = np.zeros((nodes, nodes), dtype=np.float32)
    for i in range(1, nodes):
        j = int(rng.integers(0, i))
        a[i, j] = a[j, i] = 1
    x = np.abs(rng.normal(size=(n, nodes, F))).astype(np.float32)
    y = rng.integers(0, nodes, n)
    f = rng.integers(0, 5, n)
    return x, a, y, f


def test_overfits_eight_incidents():
    x, a, y, f = _fake_set()
    torch.manual_seed(0)
    m = RCAGNN(F, 5, dropout=0.0)
    m.set_norm(x)
    ah = torch.as_tensor(normalised_adjacency(a))
    opt = torch.optim.Adam(m.parameters(), lr=5e-3)
    ce = torch.nn.CrossEntropyLoss()
    xt, yt, ft = torch.as_tensor(x), torch.as_tensor(y), torch.as_tensor(f)
    for _ in range(400):
        r, fl = m(xt, ah)
        loss = ce(r, yt) + 0.5 * ce(fl, ft)
        opt.zero_grad()
        loss.backward()
        opt.step()
    m.eval()
    r, fl = m(xt, ah)
    assert (r.argmax(1) == yt).float().mean() == 1.0
    assert (fl.argmax(1) == ft).float().mean() == 1.0


def test_deterministic_under_seed():
    x, a, _, _ = _fake_set()
    ah = torch.as_tensor(normalised_adjacency(a))
    outs = []
    for _ in range(2):
        torch.manual_seed(3)
        m = RCAGNN(F, 5).eval()
        outs.append(m(torch.as_tensor(x), ah)[0].detach().numpy())
    assert np.array_equal(outs[0], outs[1])


def test_onnx_parity_dynamic_nodes(tmp_path):
    x, a, _, _ = _fake_set(n=2, nodes=12)
    torch.manual_seed(1)
    m = RCAGNN(F, 5).eval()
    m.set_norm(x)
    p = tmp_path / "m.onnx"
    export_onnx(m, p, F)
    assert onnx_parity(m, p, x[0], a) < 1e-4
    x2, a2, _, _ = _fake_set(n=1, nodes=30, seed=5)  # different node count
    assert onnx_parity(m, p, x2[0], a2) < 1e-4
