"""Invariants for TopK (default) / L1 / JumpReLU SAEs. Fail if the math is wrong."""

from __future__ import annotations

import itertools

import numpy as np
import sae as sae_mod
import torch
import torch.nn.functional as F

from sae import (
    DeadLatentTracker,
    JumpReLUSAE,
    L1SAE,
    TopKSAE,
    _linear_sum_assignment,
    heaviside_ste,
    hungarian_decoder_cosine,
    jump_relu,
    sample_superposition,
    superposition_features,
    topk_latents,
    train_sae,
)


def test_no_identity_shims():
    """No back-compat aliases that only assert is / rename without a second algorithm."""
    banned = ("ReLUL1SAE", "SAE", "ByteLevelBPE", "HNSWOnly")
    for name in banned:
        assert not hasattr(sae_mod, name), f"shim {name} must not exist"
    assert sae_mod.TopKSAE is TopKSAE
    assert sae_mod.L1SAE is L1SAE
    assert sae_mod.JumpReLUSAE is JumpReLUSAE


def test_topk_exactly_k_nonzeros():
    pre = torch.tensor([[3.0, 1.0, 2.0, -4.0, 0.5], [0.1, 0.2, 0.0, -1.0, 5.0]])
    z = topk_latents(pre, k=2)
    assert (z[0] != 0).sum().item() == 2
    assert (z[1] != 0).sum().item() == 2
    assert torch.allclose(z[0], torch.tensor([3.0, 0.0, 2.0, 0.0, 0.0]))
    assert torch.allclose(z[1], torch.tensor([0.0, 0.2, 0.0, 0.0, 5.0]))


def test_topk_encoder_formula():
    torch.manual_seed(0)
    sae = TopKSAE(d=4, n_latents=6, k=2, tied=False)
    x = torch.randn(3, 4)
    pre = (x - sae.b_dec) @ sae.W_enc.t() + sae.b_enc
    z = topk_latents(pre, 2)
    out = sae(x)
    assert torch.allclose(out.pre, pre)
    assert torch.allclose(out.z, z)
    assert torch.allclose(out.x_hat, z @ sae.W_dec.t() + sae.b_dec)


def test_tied_encoder_uses_decoder_transpose():
    sae = TopKSAE(d=5, n_latents=7, k=2, tied=True)
    x = torch.randn(2, 5)
    pre = (x - sae.b_dec) @ sae.W_dec + sae.b_enc
    assert torch.allclose(sae.encode_pre(x), pre)
    assert not hasattr(sae, "W_enc")


def test_decoder_columns_unit_norm_after_step():
    torch.manual_seed(1)
    sae = TopKSAE(d=8, n_latents=12, k=2)
    sae.W_dec.data.mul_(3.4)
    x = torch.randn(16, 8)
    opt = torch.optim.SGD(sae.parameters(), lr=0.05)
    out = sae(x)
    sae.reconstruction_loss(x, out.x_hat).backward()
    opt.step()
    sae.project_decoder()
    norms = sae.W_dec.norm(dim=0)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)


def test_reconstruction_mse_drops():
    torch.manual_seed(2)
    W = superposition_features(8, 8, seed=2)
    sae = TopKSAE(d=8, n_latents=8, k=1)
    x = sample_superposition(W, 128, n_active=1)
    with torch.no_grad():
        mse0 = F.mse_loss(sae(x).x_hat, x).item()
    train_sae(sae, W, steps=200, batch=128, lr=3e-3, n_active=1, seed=2)
    with torch.no_grad():
        mse1 = F.mse_loss(sae(x).x_hat, x).item()
    assert mse1 < mse0 * 0.6
    norms = sae.W_dec.norm(dim=0)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-4)


def _brute_min_assignment_cost(cost: np.ndarray) -> float:
    n, m = cost.shape
    if n <= m:
        return float(min(sum(cost[i, cols[i]] for i in range(n)) for cols in itertools.permutations(range(m), n)))
    return float(min(sum(cost[rows[j], j] for j in range(m)) for rows in itertools.permutations(range(n), m)))


def test_linear_sum_assignment_matches_brute():
    rng = np.random.default_rng(0)
    for n, m in ((1, 1), (3, 3), (2, 4), (4, 2), (3, 5)):
        cost = rng.normal(size=(n, m))
        rows, cols = _linear_sum_assignment(cost)
        assert len(rows) == min(n, m)
        assert len(set(rows.tolist())) == len(rows)
        assert len(set(cols.tolist())) == len(cols)
        got = float(cost[rows, cols].sum())
        assert abs(got - _brute_min_assignment_cost(cost)) < 1e-9


def test_superposition_hungarian_cosine():
    torch.manual_seed(3)
    d, n_f = 10, 14
    W = superposition_features(d, n_f, seed=3)
    sae = TopKSAE(d=d, n_latents=n_f, k=1, tied=False, aux_k=4, aux_coef=0.1)
    train_sae(sae, W, steps=700, batch=256, lr=4e-3, n_active=1, n_dead=60, seed=3)
    mean_cos = hungarian_decoder_cosine(sae.W_dec, W)
    assert mean_cos > 0.7, f"mean matched cosine {mean_cos:.3f} <= 0.7"


def test_l1_high_sparsity():
    torch.manual_seed(4)
    W = superposition_features(8, 12, seed=4)
    sae = L1SAE(d=8, n_latents=48, l1=2.0)
    train_sae(sae, W, steps=280, batch=128, lr=2e-3, n_active=1, seed=4)
    x = sample_superposition(W, 256, n_active=1)
    with torch.no_grad():
        z = sae.encode(x)
    frac_zero = (z == 0).float().mean().item()
    assert frac_zero > 0.8, f"frac_zero={frac_zero:.3f}"


def test_jump_relu_forward_zeros_below_threshold():
    pre = torch.tensor([[0.05, 0.4, -0.2, 0.11]])
    theta = torch.tensor([0.1, 0.1, 0.1, 0.1])
    z = jump_relu(pre, theta, bandwidth=0.05)
    assert torch.allclose(z, torch.tensor([[0.0, 0.4, 0.0, 0.11]]))


def test_jump_relu_ste_threshold_grad():
    """Eq. 11: when pre is within ε/2 of θ, ∂L/∂θ is nonzero (hard gate alone gives 0)."""
    pre = torch.tensor([[0.12, 0.50]], requires_grad=False)
    theta = torch.nn.Parameter(torch.tensor([0.10, 0.10]))
    bw = 0.05
    # pre[0]=0.12 is within (θ ± bw/2) = (0.075, 0.125); pre[1]=0.50 is outside.
    z = jump_relu(pre, theta, bandwidth=bw)
    z.sum().backward()
    assert theta.grad is not None
    # First latent near threshold → nonzero θ grad; second far → ~0.
    assert abs(theta.grad[0].item()) > 1e-6
    assert abs(theta.grad[1].item()) < 1e-8


def test_heaviside_ste_l0_trains_threshold():
    """Eq. 12: L0 STE moves θ; a hard (z!=0) mean would give zero θ grad."""
    pre = torch.tensor([[0.11, 0.02]])
    theta = torch.nn.Parameter(torch.tensor([0.10, 0.10]))
    l0 = heaviside_ste(pre, theta, bandwidth=0.05).sum()
    l0.backward()
    assert theta.grad is not None
    assert abs(theta.grad[0].item()) > 1e-6  # near threshold
    # Far below: (0.02-0.10)/0.05 = -1.6, outside rect → 0
    assert abs(theta.grad[1].item()) < 1e-8


def test_jumprelu_loss_routes_through_ste():
    torch.manual_seed(5)
    sae = JumpReLUSAE(d=4, n_latents=6, threshold=0.1, l0_coef=0.5, bandwidth=0.2)
    # Force some pre-activations near θ so STE fires.
    with torch.no_grad():
        sae.W_enc.zero_()
        sae.b_enc.fill_(0.12)
        sae.threshold.fill_(0.10)
    x = torch.zeros(8, 4)
    out = sae(x)
    loss = sae.loss(x, out)
    loss.backward()
    assert sae.threshold.grad is not None
    assert sae.threshold.grad.abs().sum().item() > 0


def test_dead_auxk_matches_manual_residual():
    torch.manual_seed(6)
    sae = TopKSAE(d=4, n_latents=5, k=1, aux_k=2, aux_coef=1.0)
    x = torch.randn(3, 4)
    out = sae(x)
    dead = torch.tensor([True, True, False, False, True])
    residual = (x - out.x_hat).detach()
    got = sae.dead_auxk_loss(out.pre, residual, dead)
    masked = out.pre.masked_fill(~dead.unsqueeze(0), -1e9)
    z_aux = topk_latents(masked, 2)
    expect = F.mse_loss(z_aux @ sae.W_dec.t(), residual)
    assert torch.allclose(got, expect)


def test_dead_latent_resample_rewrites_column():
    sae = TopKSAE(d=4, n_latents=3, k=1)
    tracker = DeadLatentTracker(3, n_dead=2)
    tracker.steps_since_fire[:] = 5
    residual = torch.tensor([[0.0, 0.0, 3.0, 4.0]]).repeat(3, 1)
    n = tracker.resample(sae, residual)
    assert n == 3
    col = F.normalize(sae.W_dec[:, 0], dim=0)
    assert torch.allclose(col, torch.tensor([0.0, 0.0, 0.6, 0.8]), atol=1e-5)
    assert tracker.steps_since_fire[0].item() == 0
