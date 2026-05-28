"""tests/test_model.py — model shape and forward pass tests."""
import pytest
import torch
from sdrl.model import PrimaryNet, DualNet, build_models


@pytest.fixture
def device():
    return "cuda" if torch.cuda.is_available() else "cpu"


@pytest.fixture
def models(device):
    return build_models(base_ch=16, device=device)   # small for CI speed


class TestPrimaryNet:
    def test_output_shape(self, models, device):
        P, _ = models
        x = torch.randn(2, 1, 50, 50, device=device)
        y = P(x)
        assert y.shape == (2, 1, 200, 200), f"Got {y.shape}"

    def test_skip_connection_alignment(self, models, device):
        """Encoder produces odd sizes (50→25→13→7); decoder must align."""
        P, _ = models
        x = torch.randn(1, 1, 50, 50, device=device)
        # should not raise any shape mismatch
        _ = P(x)

    def test_odd_batch_size(self, models, device):
        P, _ = models
        x = torch.randn(3, 1, 50, 50, device=device)
        assert P(x).shape == (3, 1, 200, 200)

    def test_no_nan_output(self, models, device):
        P, _ = models
        x = torch.randn(2, 1, 50, 50, device=device)
        y = P(x)
        assert not torch.isnan(y).any()

    def test_gradient_flows(self, models, device):
        P, _ = models
        x = torch.randn(2, 1, 50, 50, device=device, requires_grad=False)
        y = P(x)
        loss = y.mean()
        loss.backward()
        # check at least one parameter has a gradient
        assert any(p.grad is not None for p in P.parameters())


class TestDualNet:
    def test_output_shape(self, models, device):
        _, D = models
        y = torch.randn(2, 1, 200, 200, device=device)
        x = D(y)
        assert x.shape == (2, 1, 50, 50), f"Got {x.shape}"

    def test_no_nan_output(self, models, device):
        _, D = models
        y = torch.randn(2, 1, 200, 200, device=device)
        assert not torch.isnan(D(y)).any()


class TestCycleConsistency:
    def test_full_cycle_shape(self, models, device):
        """D(P(x)) must have same shape as x."""
        P, D = models
        x     = torch.randn(2, 1, 50, 50, device=device)
        y_hat = P(x)
        x_hat = D(y_hat)
        assert x_hat.shape == x.shape

    def test_param_count_reasonable(self, device):
        """Networks should not be absurdly large."""
        P, D = build_models(base_ch=32, device=device)
        n_P = sum(p.numel() for p in P.parameters())
        n_D = sum(p.numel() for p in D.parameters())
        assert n_P < 50_000_000, f"PrimaryNet too large: {n_P:,}"
        assert n_D < 10_000_000, f"DualNet too large: {n_D:,}"
