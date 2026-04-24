import torch
from camelflow.models import LSTMRegressor


def test_forward_shape():
    m = LSTMRegressor(n_dyn=5, n_static=14, hidden=32, num_layers=1, dropout=0.0)
    x_dyn = torch.randn(4, 30, 5)
    x_static = torch.randn(4, 14)
    y = m(x_dyn, x_static)
    assert y.shape == (4,)


def test_backward():
    m = LSTMRegressor(n_dyn=5, n_static=14, hidden=16)
    x_dyn = torch.randn(2, 10, 5)
    x_static = torch.randn(2, 14)
    y = m(x_dyn, x_static).sum()
    y.backward()
    assert all(p.grad is not None for p in m.parameters())
