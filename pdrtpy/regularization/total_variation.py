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
    """Forward difference of a 2-D array along columns (the x/horizontal axis).

    ``gx[i, j] = u[i, j+1] - u[i, j]`` wherever both ``mask[i, j]`` and
    ``mask[i, j+1]`` are True; zero everywhere else (the last column, and
    any location where either endpoint is invalid), which treats an invalid
    neighbor exactly like a grid boundary — no flux crosses it.

    Parameters
    ----------
    u : `~numpy.ndarray`
        2-D array to differentiate.
    mask : `~numpy.ndarray`
        2-D boolean array, same shape as ``u``; True where a pixel is valid.

    Returns
    -------
    `~numpy.ndarray`
        The forward difference along columns, same shape as ``u``.
    """
    gx = np.zeros_like(u)
    valid_edge = mask[:, :-1] & mask[:, 1:]
    gx[:, :-1][valid_edge] = (u[:, 1:] - u[:, :-1])[valid_edge]
    return gx


def _forward_diff_y(u, mask):
    """Forward difference of a 2-D array along rows (the y/vertical axis).

    ``gy[i, j] = u[i+1, j] - u[i, j]`` wherever both ``mask[i, j]`` and
    ``mask[i+1, j]`` are True; zero everywhere else (the last row, and any
    location where either endpoint is invalid) — see `_forward_diff_x` for
    the boundary-handling rationale, which is identical here.

    Parameters
    ----------
    u : `~numpy.ndarray`
        2-D array to differentiate.
    mask : `~numpy.ndarray`
        2-D boolean array, same shape as ``u``; True where a pixel is valid.

    Returns
    -------
    `~numpy.ndarray`
        The forward difference along rows, same shape as ``u``.
    """
    gy = np.zeros_like(u)
    valid_edge = mask[:-1, :] & mask[1:, :]
    gy[:-1, :][valid_edge] = (u[1:, :] - u[:-1, :])[valid_edge]
    return gy


def _divergence(p1, p2, mask):
    """Discrete divergence of a 2-D vector field, the adjoint of
    (`_forward_diff_x`, `_forward_diff_y`).

    Implements ``div = -(forward_diff)^T`` under the standard convention
    ``<grad u, p> = -<u, div p>``, so that the Chambolle dual-ascent
    iteration in `TotalVariationRegularizer._denoise_one` can use ``div``
    directly (see the derivation noted in that method). Invalid pixels are
    zeroed in the output, matching how `_forward_diff_x`/`_forward_diff_y`
    treat them as flux-free boundaries.

    Parameters
    ----------
    p1 : `~numpy.ndarray`
        x-component (column-direction) of the dual field.
    p2 : `~numpy.ndarray`
        y-component (row-direction) of the dual field, same shape as ``p1``.
    mask : `~numpy.ndarray`
        2-D boolean array, same shape as ``p1``/``p2``; True where a pixel
        is valid.

    Returns
    -------
    `~numpy.ndarray`
        The divergence field, same shape as ``p1``/``p2``, zero at invalid
        pixels.
    """
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
        """Construct a Total Variation regularizer.

        Parameters
        ----------
        lam : float
            Regularization strength (see `Regularizer.__init__`).
        mode : str, optional
            ``"isotropic"`` (default) or ``"anisotropic"`` — see the class
            docstring above.
        n_iter : int, optional
            Number of Chambolle dual-ascent iterations per call to `prox`.
            Default: 50.
        tau : float, optional
            Dual step size; must satisfy ``tau <= 0.25`` for stability of
            the 2-D scheme. Default: 0.125.

        Raises
        ------
        ValueError
            If ``lam`` is negative (raised by the parent `Regularizer`), or
            if ``mode`` is not ``"isotropic"`` or ``"anisotropic"``.
        """
        super().__init__(lam)
        if mode not in ("isotropic", "anisotropic"):
            raise ValueError("mode must be 'isotropic' or 'anisotropic'")
        self.mode = mode
        self.n_iter = n_iter
        self.tau = tau

    def prox(self, maps, valid_mask, step):
        """Proximal operator of ``step * lam * TV(.)``, applied independently to each map.

        Parameters
        ----------
        maps : list of `~numpy.ndarray`
            One or more independent 2-D parameter maps (same shape).
        valid_mask : `~numpy.ndarray`
            2-D boolean array; True where a pixel has a fitted value.
        step : float or `~numpy.ndarray`
            The proximal-gradient step size for this iteration — a single
            value shared by every pixel, or a 2-D array (same shape as
            ``valid_mask``) giving each pixel its own step (see
            `~pdrtpy.regularization.base.fista`'s ``step_mode="per_pixel"``).

        Returns
        -------
        list of `~numpy.ndarray`
            The TV-denoised maps, same shapes as the input. Invalid
            (masked) pixels are copied through unchanged.
        """
        valid_mask = np.asarray(valid_mask, dtype=bool)
        theta = np.asarray(step, dtype=float) * self.lam
        if theta.ndim == 0:
            theta = np.full(valid_mask.shape, float(theta))
        return [self._denoise_one(np.asarray(m, dtype=float), valid_mask, theta) for m in maps]

    def _denoise_one(self, m, valid_mask, theta):
        """Run Chambolle's dual-ascent TV-denoising iteration on a single map.

        Solves ``x* = argmin_x 0.5*||x - m||^2 + theta*TV(x)`` (restricted to
        ``valid_mask``) via the dual variable :math:`p = (p_1, p_2)`:
        :math:`p^* = \\mathrm{argmin}_{|p|\\le 1} \\|\\mathrm{div}(p) + m/\\theta\\|^2`,
        found by `self.n_iter` projected-gradient-ascent steps of size
        `self.tau`, then :math:`x^* = m + \\theta\\,\\mathrm{div}(p^*)`. See
        ``docs/chambolle.md`` for a plain-language walkthrough of the
        dual-projection idea.

        ``theta`` may vary per pixel (spatially-varying regularization
        strength): it enters every formula above elementwise (``m/theta``,
        ``theta*div(p*)``), which is the standard generalization used for
        locally-adaptive TV denoising in the literature. Physically, a
        pixel with a small local ``theta`` (e.g. one with a large FISTA
        step penalty because its data-fidelity gradient is locally steep)
        is pulled less strongly toward its neighbors' consensus per call;
        it still gets there over more outer `~pdrtpy.regularization.base.fista`
        iterations, rather than dragging every other pixel's ``theta`` down
        to match it the way a single shared scalar ``theta`` would.

        Parameters
        ----------
        m : `~numpy.ndarray`
            2-D map to denoise.
        valid_mask : `~numpy.ndarray`
            2-D boolean array, same shape as ``m``; True where a pixel is
            valid. Invalid pixels are excluded from the dual iteration and
            copied through unchanged in the output.
        theta : `~numpy.ndarray`
            Effective regularization strength for this call, i.e.
            ``step * self.lam`` from `prox`, already broadcast to a 2-D
            array the same shape as ``m``. Pixels where ``theta <= 0`` are
            treated as invalid (returned unchanged), same as a masked pixel.

        Returns
        -------
        `~numpy.ndarray`
            The denoised map, same shape as ``m``.
        """
        active = valid_mask & (theta > 0)
        if not np.any(active):
            return np.array(m, copy=True)
        # Avoid division by zero at inactive pixels; their result is discarded below.
        theta_safe = np.where(theta > 0, theta, 1.0)

        y = np.where(valid_mask, m, 0.0)
        p1 = np.zeros_like(y)
        p2 = np.zeros_like(y)

        # Chambolle (2004) dual-ascent iteration for prox_{theta*TV}(y):
        # p* = argmin_{|p|<=1} ||div(p) + y/theta||^2, x = y + theta*div(p*).
        # (`_divergence` above already implements the sign convention this
        # derivation assumes, i.e. <grad u, p> = -<u, div p>.)
        for _ in range(self.n_iter):
            div_p = _divergence(p1, p2, valid_mask)
            w = div_p + y / theta_safe
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
        x = y + theta_safe * div_p
        out = np.array(m, copy=True)
        out[active] = x[active]
        return out
