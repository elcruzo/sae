"""Sparse autoencoders: OpenAI TopK (default), ReLU+L1, and JumpReLU.

Default encoder (Gao et al. / ICLR 2025):
    z = TopK(W_enc (x - b_dec) + b_enc)     # k largest kept, rest 0
Decoder:
    x_hat = W_dec z + b_dec                 # columns of W_dec are unit-norm

Named variants:
  - L1SAE: z = ReLU(pre), loss = MSE + λ‖z‖₁ (Bricken / early Anthropic)
  - JumpReLUSAE: z = JumpReLU_θ(pre) with rectangle-kernel STEs for θ and L0
    (Rajamanoharan et al. 2024)

Dead latents: Anthropic resampling onto high-residual inputs; TopK also supports
Gao AuxK (reconstruct residual with top-k_aux dead latents).

Papers: Bricken et al. 2023; Templeton et al. 2024; Gao et al. ICLR 2025;
Rajamanoharan et al. (JumpReLU); Elhage et al. 2022 (Toy Models of Superposition).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment


def rectangle(x: torch.Tensor) -> torch.Tensor:
    """Unit-height rect kernel on (-1/2, 1/2); used as KDE bandwidth kernel for STEs."""
    return ((x > -0.5) & (x < 0.5)).to(x.dtype)


class _JumpReLUFn(torch.autograd.Function):
    """Forward: z * H(z - θ). Backward w.r.t. θ uses Eq. (11) rectangle STE."""

    @staticmethod
    def forward(ctx, pre: torch.Tensor, threshold: torch.Tensor, bandwidth: float) -> torch.Tensor:
        ctx.save_for_backward(pre, threshold)
        ctx.bandwidth = bandwidth
        return pre * (pre > threshold).to(pre.dtype)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        pre, threshold = ctx.saved_tensors
        bw = ctx.bandwidth
        grad_pre = (pre > threshold).to(pre.dtype) * grad_output
        # ∂/∂θ JumpReLU = -(θ/ε) K((z-θ)/ε)  (Rajamanoharan et al. Eq. 11)
        kernel = rectangle((pre - threshold) / bw)
        grad_θ = (-(threshold / bw) * kernel * grad_output).sum(dim=0)
        return grad_pre, grad_θ, None


class _StepFn(torch.autograd.Function):
    """Forward: H(z - θ). Backward w.r.t. θ uses Eq. (12) rectangle STE for L0."""

    @staticmethod
    def forward(ctx, pre: torch.Tensor, threshold: torch.Tensor, bandwidth: float) -> torch.Tensor:
        ctx.save_for_backward(pre, threshold)
        ctx.bandwidth = bandwidth
        return (pre > threshold).to(pre.dtype)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        pre, threshold = ctx.saved_tensors
        bw = ctx.bandwidth
        # ∂/∂θ H(z-θ) = -(1/ε) K((z-θ)/ε)  (Eq. 12); no gradient into pre for the L0 term
        kernel = rectangle((pre - threshold) / bw)
        grad_θ = (-(1.0 / bw) * kernel * grad_output).sum(dim=0)
        return None, grad_θ, None


def jump_relu(pre: torch.Tensor, threshold: torch.Tensor, bandwidth: float = 0.05) -> torch.Tensor:
    """Trainable JumpReLU with rectangle-kernel STE on the threshold."""
    return _JumpReLUFn.apply(pre, threshold, bandwidth)


def heaviside_ste(pre: torch.Tensor, threshold: torch.Tensor, bandwidth: float = 0.05) -> torch.Tensor:
    """H(pre - θ) with STE so L0 can train θ."""
    return _StepFn.apply(pre, threshold, bandwidth)


def topk_latents(pre: torch.Tensor, k: int) -> torch.Tensor:
    """Keep the k largest pre-activations per row; zero the rest."""
    if k >= pre.shape[-1]:
        return pre
    values, index = torch.topk(pre, k, dim=-1)
    return torch.zeros_like(pre).scatter(-1, index, values)


@dataclass
class SAEOutput:
    x_hat: torch.Tensor
    z: torch.Tensor
    pre: torch.Tensor


class TopKSAE(nn.Module):
    """OpenAI-style TopK SAE (default). Optional AuxK dead-latent revival loss."""

    def __init__(
        self,
        d: int,
        n_latents: int,
        k: int,
        tied: bool = False,
        aux_k: int = 0,
        aux_coef: float = 1.0 / 32.0,
    ) -> None:
        super().__init__()
        if not 1 <= k <= n_latents:
            raise ValueError(f"k={k} must be in [1, {n_latents}]")
        self.d = d
        self.n_latents = n_latents
        self.k = k
        self.tied = tied
        self.aux_k = aux_k
        self.aux_coef = aux_coef
        # W_dec: (d, n) — each column is a dictionary atom; init unit-norm.
        w = F.normalize(torch.randn(d, n_latents), dim=0)
        self.W_dec = nn.Parameter(w)
        if not tied:
            # Gao: init encoder to decoder transpose (reduces dead latents).
            self.W_enc = nn.Parameter(w.t().contiguous())
        self.b_enc = nn.Parameter(torch.zeros(n_latents))
        self.b_dec = nn.Parameter(torch.zeros(d))

    def encode_pre(self, x: torch.Tensor) -> torch.Tensor:
        centered = x - self.b_dec
        if self.tied:
            return centered @ self.W_dec + self.b_enc
        return centered @ self.W_enc.t() + self.b_enc

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return topk_latents(self.encode_pre(x), self.k)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return z @ self.W_dec.t() + self.b_dec

    def forward(self, x: torch.Tensor) -> SAEOutput:
        pre = self.encode_pre(x)
        z = topk_latents(pre, self.k)
        return SAEOutput(x_hat=self.decode(z), z=z, pre=pre)

    @torch.no_grad()
    def project_decoder(self) -> None:
        """Project decoder columns onto the unit sphere (after each optimizer step)."""
        self.W_dec.data.copy_(F.normalize(self.W_dec.data, dim=0))

    def reconstruction_loss(self, x: torch.Tensor, x_hat: torch.Tensor) -> torch.Tensor:
        return F.mse_loss(x_hat, x)

    def dead_auxk_loss(self, pre: torch.Tensor, residual: torch.Tensor, dead: torch.Tensor) -> torch.Tensor:
        """AuxK over latents flagged dead: reconstruct residual with top-k_aux of them (Gao A.2)."""
        k_aux = min(self.aux_k, int(dead.sum().item()))
        if k_aux <= 0:
            return residual.new_zeros(())
        masked = pre.masked_fill(~dead.unsqueeze(0), -1e9)
        z_aux = topk_latents(masked, k_aux)
        return F.mse_loss(z_aux @ self.W_dec.t(), residual)

    def nonselected_auxk_loss(self, pre: torch.Tensor, residual: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """AuxK among latents that lost the main TopK (used when no latent is flagged dead yet)."""
        k_aux = min(self.aux_k, self.n_latents - self.k)
        if k_aux <= 0:
            return residual.new_zeros(())
        masked = pre.masked_fill(z != 0, -1e9)
        z_aux = topk_latents(masked, k_aux)
        return F.mse_loss(z_aux @ self.W_dec.t(), residual)

    def aux_loss(self, x: torch.Tensor, out: SAEOutput, dead: torch.Tensor | None) -> torch.Tensor:
        if self.aux_k <= 0:
            return x.new_zeros(())
        residual = (x - out.x_hat).detach()
        if dead is not None and dead.any():
            return self.dead_auxk_loss(out.pre, residual, dead)
        return self.nonselected_auxk_loss(out.pre, residual, out.z)

    def loss(self, x: torch.Tensor, out: SAEOutput, dead: torch.Tensor | None = None) -> torch.Tensor:
        recon = self.reconstruction_loss(x, out.x_hat)
        if self.aux_k <= 0:
            return recon
        return recon + self.aux_coef * self.aux_loss(x, out, dead)


class L1SAE(nn.Module):
    """Named variant: ReLU encoder + L1 sparsity (Bricken / early Anthropic)."""

    def __init__(self, d: int, n_latents: int, l1: float = 0.1, tied: bool = False) -> None:
        super().__init__()
        self.d = d
        self.n_latents = n_latents
        self.l1 = l1
        self.tied = tied
        w = F.normalize(torch.randn(d, n_latents), dim=0)
        self.W_dec = nn.Parameter(w)
        if not tied:
            self.W_enc = nn.Parameter(w.t().contiguous())
        self.b_enc = nn.Parameter(torch.zeros(n_latents))
        self.b_dec = nn.Parameter(torch.zeros(d))

    def encode_pre(self, x: torch.Tensor) -> torch.Tensor:
        centered = x - self.b_dec
        if self.tied:
            return centered @ self.W_dec + self.b_enc
        return centered @ self.W_enc.t() + self.b_enc

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu(self.encode_pre(x))

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return z @ self.W_dec.t() + self.b_dec

    def forward(self, x: torch.Tensor) -> SAEOutput:
        pre = self.encode_pre(x)
        z = F.relu(pre)
        return SAEOutput(x_hat=self.decode(z), z=z, pre=pre)

    @torch.no_grad()
    def project_decoder(self) -> None:
        self.W_dec.data.copy_(F.normalize(self.W_dec.data, dim=0))

    def loss(self, x: torch.Tensor, out: SAEOutput, dead: torch.Tensor | None = None) -> torch.Tensor:
        del dead
        return F.mse_loss(out.x_hat, x) + self.l1 * out.z.abs().mean()


class JumpReLUSAE(nn.Module):
    """Named variant: JumpReLU with learned θ and L0 via rectangle-kernel STEs."""

    def __init__(
        self,
        d: int,
        n_latents: int,
        threshold: float = 0.1,
        l0_coef: float = 0.01,
        bandwidth: float = 0.05,
    ) -> None:
        super().__init__()
        self.d = d
        self.n_latents = n_latents
        self.l0_coef = l0_coef
        self.bandwidth = bandwidth
        w = F.normalize(torch.randn(d, n_latents), dim=0)
        self.W_dec = nn.Parameter(w)
        self.W_enc = nn.Parameter(w.t().contiguous())
        self.b_enc = nn.Parameter(torch.zeros(n_latents))
        self.b_dec = nn.Parameter(torch.zeros(d))
        self.threshold = nn.Parameter(torch.full((n_latents,), threshold))

    def encode_pre(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.b_dec) @ self.W_enc.t() + self.b_enc

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return jump_relu(self.encode_pre(x), self.threshold, self.bandwidth)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return z @ self.W_dec.t() + self.b_dec

    def forward(self, x: torch.Tensor) -> SAEOutput:
        pre = self.encode_pre(x)
        z = jump_relu(pre, self.threshold, self.bandwidth)
        return SAEOutput(x_hat=self.decode(z), z=z, pre=pre)

    @torch.no_grad()
    def project_decoder(self) -> None:
        self.W_dec.data.copy_(F.normalize(self.W_dec.data, dim=0))
        self.threshold.data.clamp_(min=0.0)

    def loss(self, x: torch.Tensor, out: SAEOutput, dead: torch.Tensor | None = None) -> torch.Tensor:
        del dead
        recon = F.mse_loss(out.x_hat, x)
        if self.l0_coef <= 0:
            return recon
        # L0 via STE Heaviside so θ receives gradient (Eq. 10 + Eq. 12).
        l0 = heaviside_ste(out.pre, self.threshold, self.bandwidth).sum(dim=-1).mean()
        return recon + self.l0_coef * l0


class DeadLatentTracker:
    """Count steps since each latent last fired; resample if silent for `n_dead` steps."""

    def __init__(self, n_latents: int, n_dead: int = 200) -> None:
        self.n_dead = n_dead
        self.steps_since_fire = torch.zeros(n_latents, dtype=torch.long)

    def update(self, z: torch.Tensor) -> None:
        fired = (z != 0).any(dim=0).cpu()
        self.steps_since_fire += 1
        self.steps_since_fire[fired] = 0

    def dead_mask(self) -> torch.Tensor:
        return self.steps_since_fire >= self.n_dead

    @torch.no_grad()
    def resample(self, sae: nn.Module, residual: torch.Tensor) -> int:
        """Point dead decoder columns at high-residual samples (Anthropic resampling)."""
        dead = self.dead_mask()
        n_dead = int(dead.sum().item())
        if n_dead == 0 or residual.numel() == 0:
            return 0
        norms = residual.detach().norm(dim=-1)
        n_pick = min(n_dead, residual.shape[0])
        _, top = torch.topk(norms, n_pick)
        directions = F.normalize(residual.detach()[top], dim=-1)
        dead_idx = dead.nonzero(as_tuple=False).squeeze(-1)[:n_pick]
        scale = math.sqrt(sae.d)
        for i, j in enumerate(dead_idx.tolist()):
            sae.W_dec.data[:, j] = directions[i]
            if hasattr(sae, "W_enc"):
                sae.W_enc.data[j] = directions[i] * scale
            sae.b_enc.data[j] = 0.0
            if hasattr(sae, "threshold"):
                sae.threshold.data[j] = 0.0
            self.steps_since_fire[j] = 0
        return n_pick


def superposition_features(d: int, n_features: int, seed: int = 0) -> torch.Tensor:
    """Unit-norm feature dictionary, orthogonal-ish via QR when possible (Elhage toy)."""
    g = torch.Generator().manual_seed(seed)
    raw = torch.randn(max(d, n_features), max(d, n_features), generator=g)
    q, _ = torch.linalg.qr(raw)
    feats = q[:d, :n_features].contiguous()
    return F.normalize(feats, dim=0)


def sample_superposition(
    W: torch.Tensor,
    batch: int,
    n_active: int = 1,
    mag: tuple[float, float] = (0.8, 1.6),
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """x = W f with f n_active-sparse (uniform feature index, random magnitude)."""
    d, n_f = W.shape
    x = torch.zeros(batch, d)
    for _ in range(n_active):
        idx = torch.randint(0, n_f, (batch,), generator=generator)
        amp = mag[0] + (mag[1] - mag[0]) * torch.rand(batch, generator=generator)
        x = x + amp.unsqueeze(-1) * W[:, idx].t()
    return x


def hungarian_decoder_cosine(W_dec: torch.Tensor, W_true: torch.Tensor) -> float:
    """Mean cosine after optimal matching of decoder columns to true features."""
    dec = F.normalize(W_dec.detach(), dim=0)
    true = F.normalize(W_true.detach(), dim=0)
    cos = (true.t() @ dec).cpu().numpy()
    rows, cols = linear_sum_assignment(-cos)
    return float(cos[rows, cols].mean())


def sae_step(sae: nn.Module, opt: torch.optim.Optimizer, x: torch.Tensor, tracker: DeadLatentTracker | None) -> float:
    sae.train()
    opt.zero_grad(set_to_none=True)
    out = sae(x)
    dead = None if tracker is None else tracker.dead_mask().to(x.device)
    loss = sae.loss(x, out, dead)
    loss.backward()
    opt.step()
    sae.project_decoder()
    if tracker is not None:
        tracker.update(out.z.detach())
        tracker.resample(sae, (x - out.x_hat).detach())
    return float(loss.detach())


def train_sae(
    sae: nn.Module,
    W_true: torch.Tensor,
    steps: int = 800,
    batch: int = 256,
    lr: float = 3e-3,
    n_active: int = 1,
    n_dead: int = 80,
    seed: int = 0,
) -> list[float]:
    g = torch.Generator().manual_seed(seed)
    opt = torch.optim.Adam(sae.parameters(), lr=lr)
    tracker = DeadLatentTracker(sae.n_latents, n_dead=n_dead)
    losses: list[float] = []
    for _ in range(steps):
        x = sample_superposition(W_true, batch, n_active=n_active, generator=g)
        losses.append(sae_step(sae, opt, x, tracker))
    return losses
