"""TurboQuant MSE and inner-product optimal vector quantizers."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from openclaw_turboquant.codebook import LloydMaxCodebook
from openclaw_turboquant.qjl import QJLResult, QJLTransform
from openclaw_turboquant.rotation import random_rotation_matrix

# ======================================================================
# Quantized representations
# ======================================================================


@dataclass(frozen=True, slots=True)
class MSEQuantized:
    """Quantized representation from TurboQuant_mse."""

    indices: NDArray[np.intp]  # shape (d,), codebook indices per coordinate
    norm: float  # original L2 norm (stored in full precision)


@dataclass(frozen=True, slots=True)
class ProdQuantized:
    """Quantized representation from TurboQuant_prod."""

    mse_indices: NDArray[np.intp]  # shape (d,), MSE quantizer indices (b-1 bits)
    qjl_signs: NDArray[np.int8]  # shape (d,), QJL signs for residual
    residual_norm: float  # ||r||_2
    norm: float  # original L2 norm


# ======================================================================
# TurboQuant MSE
# ======================================================================


class TurboQuantMSE:
    """MSE-optimal vector quantizer.

    Algorithm 1 from the paper:
    1. Randomly rotate input: y = Π · x
    2. Quantize each coordinate of y using the Lloyd-Max codebook
    3. Dequantize: x̃ = Π^T · ỹ

    Parameters
    ----------
    d : int
        Vector dimension.
    bit_width : int
        Bits per coordinate (1–8 typical).
    seed : int or None
        Random seed for rotation matrix.
    codebook_max_iter : int
        Max Lloyd-Max iterations for codebook construction.
    """

    def __init__(
        self,
        d: int,
        bit_width: int,
        *,
        seed: int | None = None,
        codebook_max_iter: int = 200,
    ) -> None:
        self.d = d
        self.bit_width = bit_width

        rng = np.random.default_rng(seed)
        self.rotation = random_rotation_matrix(d, rng=rng)
        self.codebook = LloydMaxCodebook(d, bit_width, max_iter=codebook_max_iter)

    def quantize(self, x: NDArray[np.float64]) -> MSEQuantized:
        """Quantize a vector with MSE-optimal distortion.

        Parameters
        ----------
        x : NDArray[np.float64]
            Input vector of shape (d,).

        Returns
        -------
        MSEQuantized
        """
        norm = float(np.linalg.norm(x))
        if norm < 1e-30:
            return MSEQuantized(
                indices=np.zeros(self.d, dtype=np.intp),
                norm=0.0,
            )

        # Normalize to unit sphere
        x_unit = x / norm

        # Step 1: random rotation
        y = self.rotation @ x_unit

        # Step 2: scalar quantize each coordinate
        indices = self.codebook.quantize_scalar(y)

        return MSEQuantized(indices=indices, norm=norm)

    def dequantize(self, q: MSEQuantized) -> NDArray[np.float64]:
        """Reconstruct a vector from its MSE-quantized representation.

        Parameters
        ----------
        q : MSEQuantized
            Quantized representation.

        Returns
        -------
        NDArray[np.float64]
            Reconstructed vector of shape (d,).
        """
        if q.norm < 1e-30:
            return np.zeros(self.d, dtype=np.float64)

        # Dequantize coordinates
        y_hat = self.codebook.dequantize_scalar(q.indices)

        # Rotate back
        x_hat = self.rotation.T @ y_hat

        # Rescale
        return x_hat * q.norm

    def quantize_batch(self, vectors: NDArray[np.float64]) -> list[MSEQuantized]:
        """Quantize a batch of vectors.

        Parameters
        ----------
        vectors : NDArray[np.float64]
            Input matrix of shape (n, d).

        Returns
        -------
        list[MSEQuantized]
        """
        return [self.quantize(v) for v in vectors]

    def dequantize_batch(self, quantized: list[MSEQuantized]) -> NDArray[np.float64]:
        """Dequantize a batch of vectors.

        Parameters
        ----------
        quantized : list[MSEQuantized]
            List of quantized representations.

        Returns
        -------
        NDArray[np.float64]
            Reconstructed matrix of shape (n, d).
        """
        return np.array([self.dequantize(q) for q in quantized])


# ======================================================================
# TurboQuant Prod (inner-product optimal)
# ======================================================================


class TurboQuantProd:
    """Inner-product optimal vector quantizer (unbiased).

    Algorithm 2 from the paper:
    1. Apply TurboQuant_mse with bit-width (b-1)
    2. Compute residual r = x - x̃_mse
    3. Apply QJL 1-bit quantization to r/||r||
    4. Store ||r|| for rescaling

    The dequantized estimate is: x̃ = x̃_mse + ||r|| · Q_qjl⁻¹(Q_qjl(r/||r||))

    Parameters
    ----------
    d : int
        Vector dimension.
    bit_width : int
        Total bits per coordinate (≥2). MSE stage uses b-1, QJL uses 1.
    seed : int or None
        Random seed.
    codebook_max_iter : int
        Max Lloyd-Max iterations.
    """

    def __init__(
        self,
        d: int,
        bit_width: int,
        *,
        seed: int | None = None,
        codebook_max_iter: int = 200,
    ) -> None:
        if bit_width < 2:
            raise ValueError("bit_width must be >= 2 for TurboQuantProd (1 bit reserved for QJL)")

        self.d = d
        self.bit_width = bit_width

        rng = np.random.default_rng(seed)
        mse_seed = int(rng.integers(0, 2**31))
        qjl_seed = int(rng.integers(0, 2**31))

        self.mse_quantizer = TurboQuantMSE(
            d, bit_width - 1, seed=mse_seed, codebook_max_iter=codebook_max_iter
        )
        self.qjl = QJLTransform(d, seed=qjl_seed)

    def quantize(self, x: NDArray[np.float64]) -> ProdQuantized:
        """Quantize a vector for unbiased inner-product estimation.

        Parameters
        ----------
        x : NDArray[np.float64]
            Input vector of shape (d,).

        Returns
        -------
        ProdQuantized
        """
        norm = float(np.linalg.norm(x))
        if norm < 1e-30:
            return ProdQuantized(
                mse_indices=np.zeros(self.d, dtype=np.intp),
                qjl_signs=np.ones(self.d, dtype=np.int8),
                residual_norm=0.0,
                norm=0.0,
            )

        # Stage 1: MSE quantize
        mse_q = self.mse_quantizer.quantize(x)
        x_mse = self.mse_quantizer.dequantize(mse_q)

        # Stage 2: QJL on residual
        residual = x - x_mse
        residual_norm = float(np.linalg.norm(residual))

        if residual_norm < 1e-30:
            qjl_signs = np.ones(self.d, dtype=np.int8)
        else:
            r_unit = residual / residual_norm
            qjl_result = self.qjl.quantize(r_unit)
            qjl_signs = qjl_result.signs

        return ProdQuantized(
            mse_indices=mse_q.indices,
            qjl_signs=qjl_signs,
            residual_norm=residual_norm,
            norm=norm,
        )

    def dequantize(self, q: ProdQuantized) -> NDArray[np.float64]:
        """Reconstruct a vector from inner-product quantized representation.

        Parameters
        ----------
        q : ProdQuantized
            Quantized representation.

        Returns
        -------
        NDArray[np.float64]
            Reconstructed vector of shape (d,).
        """
        if q.norm < 1e-30:
            return np.zeros(self.d, dtype=np.float64)

        # Reconstruct MSE part
        mse_q = MSEQuantized(indices=q.mse_indices, norm=q.norm)
        x_mse = self.mse_quantizer.dequantize(mse_q)

        # Reconstruct QJL residual part
        if q.residual_norm < 1e-30:
            return x_mse

        qjl_result = QJLResult(signs=q.qjl_signs, norm=1.0)
        r_hat = self.qjl.dequantize(qjl_result)
        return x_mse + q.residual_norm * r_hat

    def estimate_inner_product(
        self,
        y: NDArray[np.float64],
        q: ProdQuantized,
    ) -> float:
        """Estimate ⟨y, x⟩ from quantized x (unbiased).

        Parameters
        ----------
        y : NDArray[np.float64]
            Query vector of shape (d,).
        q : ProdQuantized
            Quantized representation of x.

        Returns
        -------
        float
            Estimated inner product.
        """
        x_hat = self.dequantize(q)
        return float(y @ x_hat)

    def quantize_batch(self, vectors: NDArray[np.float64]) -> list[ProdQuantized]:
        """Quantize a batch of vectors."""
        return [self.quantize(v) for v in vectors]

    def dequantize_batch(self, quantized: list[ProdQuantized]) -> NDArray[np.float64]:
        """Dequantize a batch of vectors."""
        return np.array([self.dequantize(q) for q in quantized])


# ======================================================================
# Utility: bits-per-vector accounting
# ======================================================================


def bits_per_vector(d: int, bit_width: int, mode: str = "mse") -> int:
    """Calculate total bits used by a quantized d-dimensional vector.

    Parameters
    ----------
    d : int
        Dimension.
    bit_width : int
        Bits per coordinate.
    mode : str
        "mse" or "prod".

    Returns
    -------
    int
        Total bits (excluding norm storage).
    """
    if mode == "mse":
        return d * bit_width
    elif mode == "prod":
        # (b-1) bits for MSE + 1 bit for QJL = b bits
        return d * bit_width
    else:
        raise ValueError(f"Unknown mode: {mode!r}")
