"""Train a TopK SAE on the toy model of superposition and print recovery cosine."""

from __future__ import annotations

import torch

from sae import (
    JumpReLUSAE,
    L1SAE,
    TopKSAE,
    hungarian_decoder_cosine,
    sample_superposition,
    superposition_features,
    train_sae,
)


def main() -> None:
    torch.manual_seed(0)
    d, n_f = 10, 14
    W = superposition_features(d, n_f, seed=0)
    topk = TopKSAE(d=d, n_latents=n_f, k=1, aux_k=4)
    losses = train_sae(topk, W, steps=400, batch=256, lr=4e-3, seed=0)
    cos = hungarian_decoder_cosine(topk.W_dec, W)
    x = sample_superposition(W, 256)
    with torch.no_grad():
        mse = (topk(x).x_hat - x).pow(2).mean().item()
        nnz = (topk.encode(x) != 0).float().sum(-1).mean().item()
    print(f"TopK  steps={len(losses)}  last_loss={losses[-1]:.4f}  mse={mse:.4f}  nnz={nnz:.1f}  match_cos={cos:.3f}")

    l1 = L1SAE(d=d, n_latents=2 * n_f, l1=1.0)
    train_sae(l1, W, steps=200, batch=128, lr=2e-3, seed=1)
    with torch.no_grad():
        z = l1.encode(x)
    print(f"L1  frac_zero={(z == 0).float().mean():.3f}")

    jump = JumpReLUSAE(d=d, n_latents=n_f, threshold=0.15, l0_coef=0.05, bandwidth=0.1)
    train_sae(jump, W, steps=150, batch=128, lr=2e-3, seed=2)
    with torch.no_grad():
        zj = jump.encode(x)
    print(f"JumpReLU  frac_zero={(zj == 0).float().mean():.3f}  mean_θ={jump.threshold.mean():.3f}")


if __name__ == "__main__":
    main()
