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
        super().__init__(lam)
        self.connectivity = connectivity

    def prox(self, maps, valid_mask, step):
        valid_mask = np.asarray(valid_mask, dtype=bool)
        edges = neighbor_graph(valid_mask, connectivity=self.connectivity)

        idx = -np.ones(valid_mask.shape, dtype=int)
        n_valid = int(np.count_nonzero(valid_mask))
        idx[valid_mask] = np.arange(n_valid)

        out = []
        if n_valid == 0 or not edges or self.lam == 0:
            return [np.array(m, dtype=float, copy=True) for m in maps]

        rows, cols, data = [], [], []
        deg = np.zeros(n_valid)
        for (r1, c1), (r2, c2) in edges:
            i, j = idx[r1, c1], idx[r2, c2]
            rows += [i, j]
            cols += [j, i]
            data += [-1.0, -1.0]
            deg[i] += 1
            deg[j] += 1
        rows += list(range(n_valid))
        cols += list(range(n_valid))
        data += list(deg)
        laplacian = sparse.csr_matrix((data, (rows, cols)), shape=(n_valid, n_valid))
        theta = step * self.lam
        system = (sparse.identity(n_valid, format="csc") + theta * laplacian).tocsc()

        for m in maps:
            m = np.asarray(m, dtype=float)
            y = m[valid_mask]
            x = spsolve(system, y)
            full = np.array(m, copy=True)
            full[valid_mask] = x
            out.append(full)
        return out
