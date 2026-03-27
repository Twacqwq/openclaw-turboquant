"""Random rotation matrix generation for TurboQuant."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def random_rotation_matrix(
    d: int,
    rng: np.random.Generator | None = None,
) -> NDArray[np.float64]:
    """Generate a random orthogonal rotation matrix via QR decomposition.

    Parameters
    ----------
    d : int
        Dimension of the rotation matrix (d × d).
    rng : numpy.random.Generator or None
        Random number generator for reproducibility.

    Returns
    -------
    NDArray[np.float64]
        A d × d orthogonal matrix (rotation).
    """
    if rng is None:
        rng = np.random.default_rng()

    # Draw d×d matrix with i.i.d. N(0,1) entries, then QR decompose
    z = rng.standard_normal((d, d))
    q, r = np.linalg.qr(z)

    # Ensure uniform Haar measure: fix signs so diag(R) > 0
    signs = np.sign(np.diag(r))
    signs[signs == 0] = 1
    q = q * signs[np.newaxis, :]

    return q
