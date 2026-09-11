"""Total Variation (edge-preserving) spatial regularization.

This is the primary/recommended regularizer (see
``goals/regularization_design_options.md``): unlike Tikhonov, an L1 penalty
on neighbor differences costs the same whether a jump is concentrated in one
pixel or spread over many, so it suppresses isolated pixel-to-pixel
oscillation without blurring a genuine multi-pixel edge (e.g. an ionization
front).

Implements Chambolle's (2004) dual-projection algorithm, in-house rather than
via ``scikit-image`` so that invalid/masked pixels can be handled exactly
(treated the same as a grid boundary — zero flux across them) instead of
requiring a dense rectangular image with no masking support.
"""

import numpy as np

from .base import Regularizer


def _forward_diff_x(u, mask):
    """Forward difference along columns, zeroed wherever either endpoint is invalid."""
    gx = np.zeros_like(u)
    valid_edge = mask[:, :-1] & mask[:, 1:]
    gx[:, :-1][valid_edge] = (u[:, 1:] - u[:, :-1])[valid_edge]
    return gx


def _forward_diff_y(u, mask):
    """Forward difference along rows, zeroed wherever either endpoint is invalid."""
    gy = np.zeros_like(u)
    valid_edge = mask[:-1, :] & mask[1:, :]
    gy[:-1, :][valid_edge] = (u[1:, :] - u[:-1, :])[valid_edge]
    return gy


def _divergence(p1, p2, mask):
    """Discrete adjoint of (_forward_diff_x, _forward_diff_y), zero-padded at borders."""
    div = np.array(p1, copy=True)
    div[:, 1:] -= p1[:, :-1]
    div += p2
    div[1:, :] -= p2[:-1, :]
    div[~mask] = 0.0
    return div


class TotalVariationRegularizer(Regularizer):
    """L1 penalty on differences between spatial neighbors ("fused lasso" / TV).

    Parameters
    ----------
    lam : float
        Regularization strength.
    mode : str
        ``"isotropic"`` (default, recommended) groups each pixel's x/y
        directional differences into a single vector norm before applying
        the L1 shrinkage, which is rotationally invariant and avoids
        staircase artifacts on diagonal edges. ``"anisotropic"`` shrinks the
        x/y components independently — mathematically identical to an L1
        penalty on neighbor differences ("LASSO on neighbor differences"),
        so it is not implemented as a separate class.
    n_iter : int
        Number of Chambolle dual-ascent iterations per call to `prox`.
    tau : float
        Dual step size; must satisfy ``tau <= 0.25`` for stability of the
        2-D isotropic/anisotropic scheme.
    """

    def __init__(self, lam, mode="isotropic", n_iter=50, tau=0.125):
        super().__init__(lam)
        if mode not in ("isotropic", "anisotropic"):
            raise ValueError("mode must be 'isotropic' or 'anisotropic'")
        self.mode = mode
        self.n_iter = n_iter
        self.tau = tau

    def prox(self, maps, valid_mask, step):
        valid_mask = np.asarray(valid_mask, dtype=bool)
        theta = step * self.lam
        return [self._denoise_one(np.asarray(m, dtype=float), valid_mask, theta) for m in maps]

    def _denoise_one(self, m, valid_mask, theta):
        if theta <= 0:
            return np.array(m, copy=True)

        y = np.where(valid_mask, m, 0.0)
        p1 = np.zeros_like(y)
        p2 = np.zeros_like(y)

        # Chambolle (2004) dual-ascent iteration for prox_{theta*TV}(y):
        # p* = argmin_{|p|<=1} ||div(p) + y/theta||^2, x = y + theta*div(p*).
        # (`_divergence` above already implements the sign convention this
        # derivation assumes, i.e. <grad u, p> = -<u, div p>.)
        for _ in range(self.n_iter):
            div_p = _divergence(p1, p2, valid_mask)
            w = div_p + y / theta
            gx = _forward_diff_x(w, valid_mask)
            gy = _forward_diff_y(w, valid_mask)
            p1c = p1 + self.tau * gx
            p2c = p2 + self.tau * gy
            if self.mode == "isotropic":
                norm = np.maximum(1.0, np.sqrt(p1c**2 + p2c**2))
                p1, p2 = p1c / norm, p2c / norm
            else:
                p1 = p1c / np.maximum(1.0, np.abs(p1c))
                p2 = p2c / np.maximum(1.0, np.abs(p2c))

        div_p = _divergence(p1, p2, valid_mask)
        x = y + theta * div_p
        out = np.array(m, copy=True)
        out[valid_mask] = x[valid_mask]
        return out
