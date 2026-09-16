"""Tool-agnostic spatial-domain regularization primitives.

This module has no dependency on any particular fitting tool
(:class:`~pdrtpy.tool.lineratiofit.LineRatioFit`, and eventually
:class:`~pdrtpy.tool.excitation.ExcitationFit`). It operates only on plain
2-D parameter maps, a shared validity mask, and caller-supplied
objective/gradient/proximal callables, so it can be reused by any tool that
fits an independent value at every spatial pixel.
"""

from abc import ABC, abstractmethod

import numpy as np


def neighbor_graph(valid_mask, connectivity=4):
    """Build the spatial neighbor graph over the valid pixels of a 2-D mask.

    Parameters
    ----------
    valid_mask : `~numpy.ndarray`
        2-D boolean array; True where a pixel has a fitted value.
    connectivity : int
        4 (edge neighbors only) or 8 (edge + diagonal neighbors).

    Returns
    -------
    list of tuple
        Each edge is ``((r1, c1), (r2, c2))``. An edge is never created if
        either endpoint is invalid, so masked/NaN regions never propagate
        smoothing across a gap, and there is no wraparound at map borders.

    Raises
    ------
    ValueError
        If ``connectivity`` is not 4 or 8.
    """
    if connectivity not in (4, 8):
        raise ValueError("connectivity must be 4 or 8")
    valid_mask = np.asarray(valid_mask, dtype=bool)
    edges = []

    horiz = valid_mask[:, :-1] & valid_mask[:, 1:]
    rs, cs = np.nonzero(horiz)
    edges.extend(((int(r), int(c)), (int(r), int(c) + 1)) for r, c in zip(rs, cs, strict=True))

    vert = valid_mask[:-1, :] & valid_mask[1:, :]
    rs, cs = np.nonzero(vert)
    edges.extend(((int(r), int(c)), (int(r) + 1, int(c))) for r, c in zip(rs, cs, strict=True))

    if connectivity == 8:
        diag_down = valid_mask[:-1, :-1] & valid_mask[1:, 1:]
        rs, cs = np.nonzero(diag_down)
        edges.extend(((int(r), int(c)), (int(r) + 1, int(c) + 1)) for r, c in zip(rs, cs, strict=True))

        diag_up = valid_mask[:-1, 1:] & valid_mask[1:, :-1]
        rs, cs = np.nonzero(diag_up)
        edges.extend(((int(r), int(c) + 1), (int(r) + 1, int(c))) for r, c in zip(rs, cs, strict=True))

    return edges


class Regularizer(ABC):
    """Base class for a spatial-domain regularization penalty.

    A ``Regularizer`` knows only how to apply its proximal operator to a list
    of independent 2-D parameter maps; it has no notion of what those maps
    represent physically (density/radiation_field, temperature/column
    density, ...).
    """

    def __init__(self, lam):
        """Store the regularization strength shared by all subclasses.

        Parameters
        ----------
        lam : float
            Regularization strength (often written :math:`\\lambda`). Must be
            non-negative; ``lam=0`` disables the penalty entirely (`prox`
            becomes the identity).

        Raises
        ------
        ValueError
            If ``lam`` is negative.
        """
        if lam < 0:
            raise ValueError("lam (regularization strength) must be non-negative")
        self.lam = lam

    @abstractmethod
    def prox(self, maps, valid_mask, step):
        """Proximal operator of ``step * lam * R(.)``, applied independently to each map.

        Parameters
        ----------
        maps : list of `~numpy.ndarray`
            One or more independent 2-D parameter maps (same shape).
        valid_mask : `~numpy.ndarray`
            2-D boolean array; True where a pixel has a fitted value.
        step : float
            The proximal-gradient step size for this iteration.

        Returns
        -------
        list of `~numpy.ndarray`
            The proximally-updated maps, same shapes as the input. Invalid
            (masked) pixels are passed through unchanged.
        """


def fista(
    x0,
    objective_fn,
    grad_fn,
    prox_fn,
    valid_mask,
    step0=1.0,
    n_iter=100,
    tol=1e-6,
    beta=0.5,
    max_backtrack=40,
):
    """Proximal-gradient Fast Iterative Shrinkage-Thresholding Algorithm
    (FISTA) solver with backtracking line search.

    Minimizes ``objective_fn(maps) + R(maps)`` where ``R``'s proximal
    operator is supplied as ``prox_fn``. Only the smooth term
    (``objective_fn``/``grad_fn``) is checked in the backtracking line
    search, per Beck & Teboulle (2009) — the composite objective is not
    evaluated at every backtrack step, since the prox step already accounts
    for ``R`` exactly.

    Parameters
    ----------
    x0 : list of `~numpy.ndarray`
        Initial parameter maps.
    objective_fn : callable
        ``objective_fn(maps) -> float``, the smooth data-fidelity term.
    grad_fn : callable
        ``grad_fn(maps) -> list of ndarray``, gradient of ``objective_fn``
        with respect to each map.
    prox_fn : callable
        ``prox_fn(maps, valid_mask, step) -> list of ndarray``.
    valid_mask : `~numpy.ndarray`
        2-D boolean array shared by all maps.
    step0 : float
        Initial step size; backtracking only ever shrinks it, and the
        shrunk value carries forward into the next outer iteration (it is
        not reset to ``step0`` every iteration).
    n_iter : int
        Maximum number of outer (FISTA) iterations.
    tol : float
        Stop early once the relative change in the parameter maps (over
        valid pixels) between iterations drops below this value.
    beta : float
        Backtracking shrink factor, ``0 < beta < 1``.
    max_backtrack : int
        Maximum number of step-size halvings per outer iteration.

    Returns
    -------
    list of `~numpy.ndarray`
        The final parameter maps.

    Notes
    -----
    See ``docs/ista.md`` for a plain-language walkthrough of the shrinkage/
    thresholding idea this generalizes (ISTA/FISTA use a proximal step in
    place of ISTA's simple shrinkage-thresholding operator).
    """
    x = [np.array(m, dtype=float, copy=True) for m in x0]
    y = [m.copy() for m in x]
    t = 1.0
    step = step0

    for _ in range(n_iter):
        fy = objective_fn(y)
        grads = grad_fn(y)
        x_new = None
        for _ in range(max_backtrack):
            z = [yi - step * gi for yi, gi in zip(y, grads, strict=True)]
            x_new = prox_fn(z, valid_mask, step)
            diff = [xn - yi for xn, yi in zip(x_new, y, strict=True)]
            lin = sum(np.nansum(gi[valid_mask] * di[valid_mask]) for gi, di in zip(grads, diff, strict=True))
            quad = sum(np.nansum(di[valid_mask] ** 2) for di in diff) / (2 * step)
            f_new = objective_fn(x_new)
            if f_new <= fy + lin + quad + 1e-12:
                break
            step *= beta
        else:
            # No backtrack succeeded; accept the smallest step tried.
            pass

        t_new = (1 + np.sqrt(1 + 4 * t * t)) / 2
        y = [xn + ((t - 1) / t_new) * (xn - xo) for xn, xo in zip(x_new, x, strict=True)]

        num = sum(np.nansum((xn[valid_mask] - xo[valid_mask]) ** 2) for xn, xo in zip(x_new, x, strict=True))
        den = sum(np.nansum(xo[valid_mask] ** 2) for xo in x) + 1e-30
        rel = np.sqrt(num / den)

        x = x_new
        t = t_new
        if rel < tol:
            break

    return x
