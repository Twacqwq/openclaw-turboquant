"""Quantized Johnson-Lindenstrauss (QJL) 1-bit transform."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True, slots=True)
class QJLResult:
    """Output of the QJL quantization."""

    signs: NDArray[np.int8]  # {-1, +1}^d
    norm: float  # ||x|| for rescaling during dequantization


class QJLTransform:
    """1-bit Quantized Johnson-Lindenstrauss transform.

    For x ∈ S^{d-1}:
        Q_qjl(x)   = sign(S · x)
        Q_qjl⁻¹(z) = (π/2 / d) · S^T · z

    Provides unbiased inner product estimation with zero overhead.

    Parameters
    ----------
    d : int
        Vector dimension.
    seed : int or None
        Random seed for the projection matrix S.
    """

    def __init__(self, d: int, *, seed: int | None = None) -> None:
        self.d = d
        rng = np.random.default_rng(seed)
        self.projection: NDArray[np.float64] = rng.standard_normal((d, d))

    def quantize(self, x: NDArray[np.float64]) -> QJLResult:
        """Quantize a vector to 1-bit signs.

        Parameters
        ----------
        x : NDArray[np.float64]
            Input vector of shape (d,).

        Returns
        -------
        QJLResult
            Quantized signs and the L2 norm of x.
        """
        projected = self.projection @ x
        signs = np.sign(projected).astype(np.int8)
        signs[signs == 0] = 1  # tie-break
        return QJLResult(signs=signs, norm=float(np.linalg.norm(x)))

    def dequantize(self, result: QJLResult) -> NDArray[np.float64]:
        """Dequantize signs back to an approximate vector.

        Parameters
        ----------
        result : QJLResult
            The quantized representation.

        Returns
        -------
        NDArray[np.float64]
            Reconstructed vector of shape (d,).
        """
        scale = (np.pi / 2) / self.d * result.norm
        return scale * (self.projection.T @ result.signs.astype(np.float64))

    def estimate_inner_product(
        self,
        y: NDArray[np.float64],
        result: QJLResult,
    ) -> float:
        """Estimate ⟨y, x⟩ from the quantized representation of x.

        Uses the unbiased estimator:
            ⟨y, Q⁻¹(Q(x))⟩ = (π/2 / d) · norm · Σ_i (s_i^T y)(sign(s_i^T x))

        Parameters
        ----------
        y : NDArray[np.float64]
            Query vector of shape (d,).
        result : QJLResult
            Quantized representation of x.

        Returns
        -------
        float
            Estimated inner product.
        """
        return float(y @ self.dequantize(result))
