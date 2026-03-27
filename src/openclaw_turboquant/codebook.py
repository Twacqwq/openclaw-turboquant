"""Lloyd-Max optimal scalar quantizer for Beta-distributed coordinates."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.integrate import quad
from scipy.special import gamma as gamma_fn


def beta_pdf(x: float, d: int) -> float:
    """PDF of a coordinate of a uniformly random point on S^{d-1}.

    f_X(x) = Gamma(d/2) / (sqrt(pi) * Gamma((d-1)/2)) * (1 - x^2)^{(d-3)/2}
    for x in [-1, 1].
    """
    if abs(x) >= 1.0:
        return 0.0
    coeff = float(gamma_fn(d / 2)) / (np.sqrt(np.pi) * float(gamma_fn((d - 1) / 2)))
    return coeff * (1.0 - x * x) ** ((d - 3) / 2)


class LloydMaxCodebook:
    """Precomputed Lloyd-Max codebook for scalar quantization of Beta-distributed
    coordinates arising from random rotation on the unit hypersphere.

    Parameters
    ----------
    d : int
        Ambient dimension of the vectors.
    bit_width : int
        Number of bits per coordinate (1, 2, 3, 4, ...).
    max_iter : int
        Maximum Lloyd-Max iterations.
    tol : float
        Convergence tolerance on centroid movement.
    """

    def __init__(
        self,
        d: int,
        bit_width: int,
        *,
        max_iter: int = 200,
        tol: float = 1e-12,
    ) -> None:
        self.d = d
        self.bit_width = bit_width
        self.n_levels = 1 << bit_width  # 2^b

        centroids = self._init_centroids()
        self.centroids = self._lloyd_max(centroids, max_iter, tol)
        self.boundaries = self._compute_boundaries(self.centroids)

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------

    def _init_centroids(self) -> NDArray[np.float64]:
        """Initialize centroids via uniform quantiles of the distribution."""
        from scipy.integrate import quad as _quad

        # Initialize centroids as conditional means in equal-width bins

        # Bisect for CDF inverse
        centroids: list[float] = []
        for i in range(self.n_levels):
            lo = -1.0 + 2.0 * i / self.n_levels
            hi = -1.0 + 2.0 * (i + 1) / self.n_levels
            # centroid = conditional mean in [lo, hi]
            num, _ = _quad(lambda x: x * beta_pdf(x, self.d), lo, hi)
            den, _ = _quad(lambda x: beta_pdf(x, self.d), lo, hi)
            centroids.append(num / den if den > 0 else (lo + hi) / 2)

        return np.array(centroids, dtype=np.float64)

    # ------------------------------------------------------------------
    # Lloyd-Max iteration
    # ------------------------------------------------------------------

    def _lloyd_max(
        self,
        centroids: NDArray[np.float64],
        max_iter: int,
        tol: float,
    ) -> NDArray[np.float64]:
        """Run Lloyd-Max algorithm to find optimal scalar quantizer."""
        for _ in range(max_iter):
            boundaries = self._compute_boundaries(centroids)
            new_centroids = np.empty_like(centroids)

            for i in range(self.n_levels):
                lo = boundaries[i]
                hi = boundaries[i + 1]
                num, _ = quad(lambda x: x * beta_pdf(x, self.d), lo, hi)
                den, _ = quad(lambda x: beta_pdf(x, self.d), lo, hi)
                new_centroids[i] = num / den if den > 1e-30 else (lo + hi) / 2

            if np.max(np.abs(new_centroids - centroids)) < tol:
                centroids = new_centroids
                break
            centroids = new_centroids

        return centroids

    @staticmethod
    def _compute_boundaries(
        centroids: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Voronoi boundaries: midpoints between consecutive centroids."""
        midpoints = (centroids[:-1] + centroids[1:]) / 2
        return np.concatenate([[-1.0], midpoints, [1.0]])

    # ------------------------------------------------------------------
    # Quantize / dequantize helpers
    # ------------------------------------------------------------------

    def quantize_scalar(self, values: NDArray[np.float64]) -> NDArray[np.intp]:
        """Map each scalar to its nearest centroid index."""
        # searchsorted on boundaries gives bucket index
        indices = np.searchsorted(self.boundaries[1:-1], values).astype(np.intp)
        return np.clip(indices, 0, self.n_levels - 1)

    def dequantize_scalar(self, indices: NDArray[np.intp]) -> NDArray[np.float64]:
        """Map centroid indices back to centroid values."""
        return self.centroids[indices]
