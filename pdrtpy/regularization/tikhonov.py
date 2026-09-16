"""Tikhonov (L2 neighbor-difference) spatial regularization.

Retained as a documented, non-recommended alternative to
:class:`~pdrtpy.regularization.total_variation.TotalVariationRegularizer`: its
quadratic penalty smooths real multi-pixel edges (e.g. an ionization front)
along with pixel-to-pixel noise, whereas TV does not. See
``goals/regularization_design_options.md`` for the full reasoning.
"""

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve

from .base import Regularizer, neighbor_graph


class TikhonovRegularizer(Regularizer):
    """L2 penalty on differences between spatial neighbors.

    The proximal operator of ``step * lam * sum_edges (x_i - x_j)^2`` is the
    solution of the sparse linear system ``(I + step*lam*L) x = y``, where
    ``L`` is the graph Laplacian of the valid-pixel neighbor graph — a
    single non-iterative solve per map, not a scalar shrinkage.
    """

    def __init__(self, lam, connectivity=4):
        """Construct a Tikhonov regularizer.

        Parameters
        ----------
        lam : float
            Regularization strength (see `Regularizer.__init__`).
        connectivity : int, optional
            4 (default) or 8 connectivity for the neighbor graph the
            Laplacian is built from; passed to `~pdrtpy.regularization.base.neighbor_graph`.

        Raises
        ------
        ValueError
            If ``lam`` is negative (raised by the parent `Regularizer`).
        """
        super().__init__(lam)
        self.connectivity = connectivity

    def prox(self, maps, valid_mask, step):
        """Proximal operator of ``step * lam * sum_edges (x_i - x_j)^2``.

        Solves the sparse linear system ``(I + L_theta) x = y`` for each map
        independently, where ``L_theta`` is the graph Laplacian of the
        valid-pixel neighbor graph, weighted per edge by
        ``theta_edge = 0.5*(theta_i + theta_j)`` — the average of the two
        endpoint pixels' own ``theta = step*lam`` — and ``y`` is that map's
        input values. When ``step`` is a single scalar shared by every
        pixel, every edge weight reduces to that one ``theta`` and this is
        exactly the original, unweighted-Laplacian formula.

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
            The proximally-updated maps, same shapes as the input. Invalid
            (masked) pixels are copied through unchanged. If there are no
            valid pixels, no edges (e.g. all pixels isolated), or ``lam=0``,
            the input maps are returned unchanged (as copies).
        """
        valid_mask = np.asarray(valid_mask, dtype=bool)
        edges = neighbor_graph(valid_mask, connectivity=self.connectivity)

        idx = -np.ones(valid_mask.shape, dtype=int)
        n_valid = int(np.count_nonzero(valid_mask))
        idx[valid_mask] = np.arange(n_valid)

        if n_valid == 0 or not edges or self.lam == 0:
            return [np.array(m, dtype=float, copy=True) for m in maps]

        theta_field = np.asarray(step, dtype=float) * self.lam
        theta_node = np.full(n_valid, float(theta_field)) if theta_field.ndim == 0 else theta_field[valid_mask]

        rows, cols, data = [], [], []
        deg = np.zeros(n_valid)
        for (r1, c1), (r2, c2) in edges:
            i, j = idx[r1, c1], idx[r2, c2]
            theta_edge = 0.5 * (theta_node[i] + theta_node[j])
            rows += [i, j]
            cols += [j, i]
            data += [-theta_edge, -theta_edge]
            deg[i] += theta_edge
            deg[j] += theta_edge
        rows += list(range(n_valid))
        cols += list(range(n_valid))
        data += list(deg)
        weighted_laplacian = sparse.csr_matrix((data, (rows, cols)), shape=(n_valid, n_valid))
        system = (sparse.identity(n_valid, format="csc") + weighted_laplacian).tocsc()

        out = []
        for m in maps:
            m = np.asarray(m, dtype=float)
            y = m[valid_mask]
            x = spsolve(system, y)
            full = np.array(m, copy=True)
            full[valid_mask] = x
            out.append(full)
        return out
