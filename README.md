# 22 — Sparse Autoencoders (TopK)

From-scratch TopK SAE (OpenAI), ReLU+L1 (early Anthropic), and JumpReLU, trained on Elhage's toy model of superposition.

## Papers

- Bricken et al., *Towards Monosemanticity: Decomposing Language Models With Dictionary Learning* (Anthropic, 2023)
- Templeton et al., *Scaling Monosemanticity: Extracting Interpretable Features from Claude 3 Sonnet* (Anthropic, 2024)
- Gao, Gebauer, et al., *Scaling and Evaluating Sparse Autoencoders* (ICLR 2025) — TopK activation
- Rajamanoharan et al., *Jumping Ahead: Improving Reconstruction Fidelity with JumpReLU SAEs*
- Elhage et al., *Toy Models of Superposition* (2022)

## Formulas

Encoder (OpenAI TopK):

\[
z = \mathrm{TopK}\bigl(W_{\mathrm{enc}}(x - b_{\mathrm{dec}}) + b_{\mathrm{enc}}\bigr)
\]

Decoder, with **unit-norm columns** of \(W_{\mathrm{dec}}\) (projected after every step):

\[
\hat x = W_{\mathrm{dec}} z + b_{\mathrm{dec}}
\]

Loss is \(\mathrm{MSE}(x,\hat x)\). Optional AuxK reconstructs the residual with the top-\(k_{\mathrm{aux}}\) dead latents. Latents that have not fired in \(N\) steps are resampled onto high-residual examples.

ReLU+L1: \(z=\mathrm{ReLU}(\cdot)\), loss \(= \mathrm{MSE} + \lambda \|z\|_1\).

JumpReLU: \(z = z_{\mathrm{pre}} \cdot H(z_{\mathrm{pre}} - \theta)\).

Recovery: Hungarian-match decoder columns to the true feature dictionary; report mean cosine.

## Papers on disk

- [`papers/gao-scaling-monosemanticity-topk-sae-2024.pdf`](papers/gao-scaling-monosemanticity-topk-sae-2024.pdf) — Gao et al. Scaling and evaluating sparse autoencoders (2024) ([arXiv:2406.04093](https://arxiv.org/abs/2406.04093))

## Run

```bash
python -m pytest 22-sae -q
python 22-sae/demo.py
```
