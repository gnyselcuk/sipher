"""
FHE-Native Layers: Polynomial activations and normalization.

Every operation in these layers is expressible as a polynomial,
making them directly computable under CKKS without approximation
beyond the polynomial degree chosen.
"""

import torch
import torch.nn as nn
import math


class PolyActivation(nn.Module):
    """
    Learnable polynomial activation function.

    Replaces GELU/ReLU with p(x) = Σ a_i · x^i evaluated via Horner's method.
    FHE cost: `degree` ct-ct multiplications, depth = ceil(log2(degree)) + 1.

    Default degree 4 approximates GELU well on [-3, 3].
    """

    def __init__(self, degree: int = 4, init: str = "gelu_approx"):
        super().__init__()
        self.degree = degree
        self.coeffs = nn.Parameter(torch.zeros(degree + 1))
        if init == "gelu_approx":
            self._init_gelu_approx()
        elif init == "random":
            nn.init.normal_(self.coeffs, std=0.1)
            self.coeffs.data[0] = 0.0
            self.coeffs.data[1] = 1.0

    def _init_gelu_approx(self):
        """
        Least-squares polynomial approximation of GELU on [-3, 3].
        Precomputed coefficients for degree 4:
          GELU(x) ≈ 0.5x + 0.2023x² + 0.0187x³ - 0.0013x⁴
        """
        with torch.no_grad():
            if self.degree >= 1:
                self.coeffs[1] = 0.5
            if self.degree >= 2:
                self.coeffs[2] = 0.2023
            if self.degree >= 3:
                self.coeffs[3] = 0.0187
            if self.degree >= 4:
                self.coeffs[4] = -0.0013

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Vectorized polynomial evaluation (avoids Python loop overhead on GPU)
        # Build powers: [x^0, x^1, x^2, ..., x^degree]
        powers = torch.stack([x.pow(i) for i in range(self.degree + 1)], dim=0)
        # Dot product with coefficients
        shape = powers.shape
        return (powers.view(self.degree + 1, -1).T @ self.coeffs).view(shape[1:])

    def extra_repr(self) -> str:
        return f"degree={self.degree}"


class PolyLayerNorm(nn.Module):
    """
    FHE-friendly Layer Normalization.

    Training mode: uses standard LayerNorm (exact).
    FHE mode: uses polynomial approximation of 1/sqrt(var).

    The polynomial approximation is only needed at FHE inference time.
    During plaintext training, standard LayerNorm gives better gradients.
    """

    def __init__(self, d_model: int, invsqrt_degree: int = 8, eps: float = 1e-5):
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        self.invsqrt_degree = invsqrt_degree
        self.gamma = nn.Parameter(torch.ones(d_model))
        self.beta = nn.Parameter(torch.zeros(d_model))
        self.fhe_mode = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=-1, keepdim=True)
        x_centered = x - mean
        var = (x_centered * x_centered).mean(dim=-1, keepdim=True)

        if self.fhe_mode:
            inv_std = self._poly_invsqrt(var)
        else:
            inv_std = torch.rsqrt(var + self.eps)

        return self.gamma * (x_centered * inv_std) + self.beta

    def _poly_invsqrt(self, x: torch.Tensor) -> torch.Tensor:
        """Polynomial approximation of 1/sqrt(x) for FHE inference."""
        x_clamped = x.clamp(min=self.eps, max=2.0)
        # Use Taylor expansion around x=1: 1/sqrt(x) ≈ 1 - (x-1)/2 + 3(x-1)²/8 - ...
        d = x_clamped - 1.0
        result = 1.0
        coeff = 1.0
        for i in range(1, self.invsqrt_degree + 1):
            coeff *= -(2 * i - 1) / (2 * i)
            result = result + coeff * d.pow(i)
        return result
