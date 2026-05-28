"""tests/test_loss.py — loss function and metrics tests."""
import pytest
import torch
from sdrl.model import build_models
from sdrl.loss  import CompositeLoss, SDRLLoss, compute_metrics


@pytest.fixture
def device():
    return "cuda" if torch.cuda.is_available() else "cpu"


@pytest.fixture
def models(device):
    return build_models(base_ch=16, device=device)


@pytest.fixture
def dummy_batch(device):
    x = torch.rand(2, 1, 50,  50, device=device)
    y = torch.rand(2, 1, 200, 200, device=device)
    return x, y


class TestCompositeLoss:
    def test_returns_scalar(self, device):
        L = CompositeLoss().to(device)
        a = torch.rand(2, 1, 200, 200, device=device)
        b = torch.rand(2, 1, 200, 200, device=device)
        loss = L(a, b)
        assert loss.shape == ()

    def test_identical_inputs_near_zero(self, device):
        L = CompositeLoss().to(device)
        a = torch.rand(2, 1, 50, 50, device=device)
        loss = L(a, a)
        assert loss.item() < 1e-4

    def test_alpha_in_range(self, device):
        L = CompositeLoss().to(device)
        assert 0.0 < L.alpha.item() < 1.0

    def test_alpha_is_learnable(self, device):
        L = CompositeLoss().to(device)
        a = torch.rand(2, 1, 50, 50, device=device)
        loss = L(a, torch.rand_like(a))
        loss.backward()
        assert L._alpha_logit.grad is not None


class TestSDRLLoss:
    def test_paired_keys(self, models, dummy_batch, device):
        P, D = models
        crit = SDRLLoss().to(device)
        x, y = dummy_batch
        out  = crit(P, D, x, y)
        for k in ("total", "cycle", "recon", "dual_reg", "dual_cons", "alpha"):
            assert k in out, f"Missing key: {k}"

    def test_unpaired_recon_zero(self, models, dummy_batch, device):
        P, D = models
        crit = SDRLLoss().to(device)
        x, _ = dummy_batch
        out  = crit(P, D, x, None)
        assert out["recon"].item() == 0.0
        assert out["dual_reg"].item() == 0.0

    def test_loss_backprop(self, models, dummy_batch, device):
        P, D = models
        crit = SDRLLoss().to(device)
        x, y = dummy_batch
        out  = crit(P, D, x, y)
        out["total"].backward()
        assert any(p.grad is not None for p in P.parameters())

    def test_supervised_only_mode(self, models, dummy_batch, device):
        """lam=0 means no cycle loss — should still train on recon only."""
        P, D = models
        crit = SDRLLoss(lam=0, mu=0, sigma=0).to(device)
        x, y = dummy_batch
        out  = crit(P, D, x, y)
        out["total"].backward()


class TestMetrics:
    def test_perfect_prediction(self, device):
        a = torch.rand(2, 1, 200, 200, device=device)
        m = compute_metrics(a, a)
        assert m["ssim"]  > 0.99
        assert m["mse"]   < 1e-8
        assert m["psnr"]  > 60.0

    def test_metric_keys(self, device):
        a = torch.rand(2, 1, 50, 50, device=device)
        b = torch.rand(2, 1, 50, 50, device=device)
        m = compute_metrics(a, b)
        for k in ("psnr", "snr", "ssim", "mse", "mae", "mre"):
            assert k in m
