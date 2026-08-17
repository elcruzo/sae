# Sparse Autoencoders — TopK (default)

From-scratch **TopK SAE** (OpenAI / Gao et al.), with named variants **L1** (ReLU + L1) and **JumpReLU** (learned threshold + L0 STEs). Trained on Elhage's toy model of superposition.

## Default: TopK

$$
z = \mathrm{TopK}_k\bigl(W_{\mathrm{enc}}(x - b_{\mathrm{dec}}) + b_{\mathrm{enc}}\bigr)
\qquad
\hat x = W_{\mathrm{dec}} z + b_{\mathrm{dec}}
$$

Decoder columns are **unit-norm** (projected after every step). Loss is MSE; optional **AuxK** reconstructs the residual with the top-$k_{\mathrm{aux}}$ dead latents (coefficient typically $1/32$). Encoder init = decoder transpose.

## Named variant: L1

`L1SAE`: $z=\mathrm{ReLU}(\cdot)$, loss $= \mathrm{MSE} + \lambda \|z\|_1$. Same unit-norm decoder projection (otherwise L1 is gamed by shrinking activations / growing decoder).

## Named variant: JumpReLU

`JumpReLUSAE`: $z = z_{\mathrm{pre}} \cdot H(z_{\mathrm{pre}} - \theta)$ with per-latent $\theta$. Threshold and L0 are trained with rectangle-kernel straight-through estimators (Rajamanoharan et al. Eqs. 11–12):

$$
\frac{\partial}{\partial\theta}\mathrm{JumpReLU}_\theta(z) = -\frac{\theta}{\varepsilon}K\!\left(\frac{z-\theta}{\varepsilon}\right),
\qquad
\frac{\partial}{\partial\theta}H(z-\theta) = -\frac{1}{\varepsilon}K\!\left(\frac{z-\theta}{\varepsilon}\right)
$$

with $K=\mathrm{rect}$ and bandwidth $\varepsilon$.

## Dead-latent resampling

Latents silent for $N$ steps are **resampled** onto high-residual examples (Anthropic): decoder column ← unit residual direction; encoder row ← $\sqrt{d}$ times that direction; $b_{\mathrm{enc}}\leftarrow 0$.

Recovery metric: Hungarian-match decoder columns to the true feature dictionary; report mean cosine.

## Papers on disk

- [`papers/gao-scaling-monosemanticity-topk-sae-2024.pdf`](papers/gao-scaling-monosemanticity-topk-sae-2024.pdf) — Gao et al. Scaling and evaluating sparse autoencoders (2024) ([arXiv:2406.04093](https://arxiv.org/abs/2406.04093))
- [`papers/rajamanoharan-jumprelu-sae-2024.pdf`](papers/rajamanoharan-jumprelu-sae-2024.pdf) — Rajamanoharan et al. JumpReLU SAEs (2024) ([arXiv:2407.14435](https://arxiv.org/abs/2407.14435))
- [`papers/elhage-toy-models-superposition-2022.pdf`](papers/elhage-toy-models-superposition-2022.pdf) — Elhage et al. Toy Models of Superposition (2022)

Also cited: Bricken et al. *Towards Monosemanticity* (Anthropic, 2023); Templeton et al. *Scaling Monosemanticity* (2024).

## Run

```bash
pip install -r requirements.txt
python main.py
python -m pytest test_sae.py -q
```
