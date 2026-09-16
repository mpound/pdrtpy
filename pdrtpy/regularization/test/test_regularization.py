"""Unit tests for the tool-agnostic :mod:`pdrtpy.regularization` module.

These tests exercise `~pdrtpy.regularization.base.neighbor_graph`, the
`~pdrtpy.regularization.base.Regularizer` base class, the two concrete
regularizers (`~pdrtpy.regularization.tikhonov.TikhonovRegularizer` and
`~pdrtpy.regularization.total_variation.TotalVariationRegularizer`), and the
`~pdrtpy.regularization.base.fista` proximal-gradient solver, all in
isolation from any fitting tool (contrast
``pdrtpy/tool/test/test_lineratiofit_regularize.py``, which exercises
:meth:`~pdrtpy.tool.lineratiofit.LineRatioFit.regularize` end to end).
"""

import numpy as np
import pytest
from pdrtpy.regularization import (
    Regularizer,
    TikhonovRegularizer,
    TotalVariationRegularizer,
    fista,
    neighbor_graph,
)

# ──────────────────────────────────────────────────────────────
# neighbor_graph
# ──────────────────────────────────────────────────────────────


class TestNeighborGraph:
    """Tests for `~pdrtpy.regularization.base.neighbor_graph`."""

    def test_full_grid_4connectivity_edge_count(self):
        """A fully-valid 3x3 grid has the expected number of 4-connectivity edges.

        A 3x3 grid has 2 horizontal edges per row (3 rows) and 2 vertical
        edges per column (3 columns), for 12 edges total.
        """
        mask = np.ones((3, 3), dtype=bool)
        edges = neighbor_graph(mask, connectivity=4)
        assert len(edges) == 12

    def test_full_grid_8connectivity_edge_count(self):
        """8-connectivity adds the two diagonal directions, so it must
        produce strictly more edges than 4-connectivity on the same mask."""
        mask = np.ones((3, 3), dtype=bool)
        edges = neighbor_graph(mask, connectivity=4)
        edges8 = neighbor_graph(mask, connectivity=8)
        assert len(edges8) > len(edges)

    def test_invalid_connectivity_raises(self):
        """Any ``connectivity`` value other than 4 or 8 must raise ``ValueError``."""
        with pytest.raises(ValueError):
            neighbor_graph(np.ones((3, 3), dtype=bool), connectivity=5)

    def test_masked_pixel_produces_no_edges(self):
        """An invalid (masked) pixel must not appear as either endpoint of any edge.

        Otherwise a proximal step could read or write through it, letting
        smoothing propagate across a gap in the data.
        """
        mask = np.ones((3, 3), dtype=bool)
        mask[1, 1] = False
        edges = neighbor_graph(mask, connectivity=4)
        for a, b in edges:
            assert a != (1, 1)
            assert b != (1, 1)

    def test_no_wraparound_at_borders(self):
        """Edges must not wrap around the edge of the map.

        A 1x3 row of valid pixels has exactly 2 horizontal edges (0-1, 1-2);
        there must be no edge directly connecting column 0 to column 2.
        """
        mask = np.ones((1, 3), dtype=bool)
        edges = neighbor_graph(mask, connectivity=4)
        assert len(edges) == 2
        assert ((0, 0), (0, 2)) not in edges
        assert ((0, 2), (0, 0)) not in edges

    def test_disconnected_valid_regions_have_no_bridging_edge(self):
        """Two valid pixels separated by an invalid one must not be connected.

        ``[True, False, True]`` has no path between its two valid pixels
        that does not cross the masked one, so the edge list must be empty.
        """
        mask = np.array([[True, False, True]])
        edges = neighbor_graph(mask, connectivity=4)
        assert edges == []


# ──────────────────────────────────────────────────────────────
# Regularizer ABC
# ──────────────────────────────────────────────────────────────


def test_regularizer_rejects_negative_lambda():
    """Both concrete regularizers must reject a negative ``lam`` at construction.

    Negative regularization strength has no meaning (see
    `~pdrtpy.regularization.base.Regularizer.__init__`), so both subclasses
    are expected to raise ``ValueError`` via the shared base-class check.
    """
    with pytest.raises(ValueError):
        TikhonovRegularizer(lam=-1.0)
    with pytest.raises(ValueError):
        TotalVariationRegularizer(lam=-1.0)


def test_regularizer_is_abstract():
    """`~pdrtpy.regularization.base.Regularizer` cannot be instantiated directly.

    It declares `~pdrtpy.regularization.base.Regularizer.prox` as an
    abstract method, so attempting to construct it should raise
    ``TypeError`` (the standard `abc.ABC` behavior).
    """
    with pytest.raises(TypeError):
        Regularizer(lam=1.0)


# ──────────────────────────────────────────────────────────────
# TikhonovRegularizer
# ──────────────────────────────────────────────────────────────


class TestTikhonovRegularizer:
    """Tests for `~pdrtpy.regularization.tikhonov.TikhonovRegularizer`."""

    def test_constant_map_is_fixed_point(self):
        """A spatially-constant map has zero neighbor differences already.

        So it must be an (approximate) fixed point of the Tikhonov proximal
        operator regardless of ``lam`` or ``step``.
        """
        m = np.full((6, 6), 3.0)
        mask = np.ones((6, 6), dtype=bool)
        reg = TikhonovRegularizer(lam=2.0)
        out = reg.prox([m], mask, step=1.0)[0]
        assert np.allclose(out, 3.0, atol=1e-8)

    def test_zero_lambda_is_identity(self):
        """``lam=0`` must disable the penalty entirely, i.e. ``prox`` is the identity."""
        rng = np.random.default_rng(0)
        m = rng.normal(size=(5, 5))
        mask = np.ones((5, 5), dtype=bool)
        reg = TikhonovRegularizer(lam=0.0)
        out = reg.prox([m], mask, step=1.0)[0]
        assert np.allclose(out, m)

    def test_invalid_pixels_unchanged(self):
        """A masked/invalid pixel's value must pass through `prox` unchanged.

        Here pixel ``(0, 1)`` is masked out with a wildly different value
        (100.0) than its valid neighbors (1.0); it must not be touched by
        the Laplacian solve over the valid pixels.
        """
        m = np.array([[1.0, 100.0], [1.0, 1.0]])
        mask = np.array([[True, False], [True, True]])
        reg = TikhonovRegularizer(lam=5.0)
        out = reg.prox([m], mask, step=1.0)[0]
        assert out[0, 1] == 100.0

    def test_smooths_isolated_spike(self):
        """A single-pixel spike surrounded by zeros must be reduced in magnitude
        by the Tikhonov proximal operator (though not necessarily as
        aggressively as by Total Variation — see
        `TestTotalVariationRegularizer.test_isolated_spike_is_smoothed_toward_neighbors`)."""
        m = np.zeros((7, 7))
        m[3, 3] = 5.0
        mask = np.ones((7, 7), dtype=bool)
        reg = TikhonovRegularizer(lam=1.0)
        out = reg.prox([m], mask, step=1.0)[0]
        assert out[3, 3] < 5.0


# ──────────────────────────────────────────────────────────────
# TotalVariationRegularizer — the core scientific claim
# ──────────────────────────────────────────────────────────────


class TestTotalVariationRegularizer:
    """Tests for `~pdrtpy.regularization.total_variation.TotalVariationRegularizer`.

    The tests in this class verify the central scientific claim of
    ``goals/regularization_design_options.md``: Total Variation suppresses
    isolated pixel-to-pixel oscillation while leaving genuine multi-pixel
    edges largely intact, unlike Tikhonov.
    """

    def test_invalid_mode_raises(self):
        """Any ``mode`` other than ``"isotropic"``/``"anisotropic"`` must raise ``ValueError``."""
        with pytest.raises(ValueError):
            TotalVariationRegularizer(lam=1.0, mode="bogus")

    def test_zero_lambda_is_identity(self):
        """``lam=0`` must disable the penalty entirely, i.e. ``prox`` is the identity."""
        rng = np.random.default_rng(1)
        m = rng.normal(size=(5, 5))
        mask = np.ones((5, 5), dtype=bool)
        reg = TotalVariationRegularizer(lam=0.0)
        out = reg.prox([m], mask, step=1.0)[0]
        assert np.allclose(out, m)

    def test_invalid_pixels_unchanged(self):
        """A masked/invalid pixel's value must pass through `prox` unchanged.

        Here pixel ``(0, 1)`` is masked out with a wildly different value
        (100.0) than its valid neighbors (1.0); it must not be touched by
        the Chambolle dual-ascent iteration over the valid pixels.
        """
        m = np.array([[1.0, 100.0], [1.0, 1.0]])
        mask = np.array([[True, False], [True, True]])
        reg = TotalVariationRegularizer(lam=5.0, n_iter=50)
        out = reg.prox([m], mask, step=1.0)[0]
        assert out[0, 1] == 100.0

    def test_isolated_spike_is_smoothed_toward_neighbors(self):
        """An isolated single-pixel oscillation should be pulled toward its
        (zero-valued) neighbors' consensus.

        This models the failure mode the design doc is written around: a
        pixel that landed on a different, near-degenerate solution than its
        neighbors should be pulled back sharply.
        """
        m = np.zeros((7, 7))
        m[3, 3] = 5.0
        mask = np.ones((7, 7), dtype=bool)
        reg = TotalVariationRegularizer(lam=1.0, mode="isotropic", n_iter=100)
        out = reg.prox([m], mask, step=1.0)[0]
        assert out[3, 3] < 0.5 * m[3, 3]  # pulled well down from 5.0 toward the surrounding 0

    def test_genuine_multipixel_edge_survives(self):
        """A step edge spanning many contiguous pixels should stay close to
        sharp, unlike an isolated spike at the same lambda.

        Verifies that the jump across a wide (5-pixel) step edge remains
        most of its original magnitude, and that the flat interior on
        either side of the edge stays close to its original constant value.
        """
        m = np.zeros((10, 10))
        m[:, 5:] = 5.0
        mask = np.ones((10, 10), dtype=bool)
        reg = TotalVariationRegularizer(lam=1.0, mode="isotropic", n_iter=100)
        out = reg.prox([m], mask, step=1.0)[0]
        assert (out[5, 5] - out[5, 4]) > 3.0
        assert out[5, 0] < 0.5
        assert out[5, 9] > 4.5

    def test_edge_preserved_better_than_tikhonov_at_matched_lambda(self):
        """The central claim of the design doc: at the same lambda, TV preserves
        a genuine multi-pixel edge much better than Tikhonov does.

        Both regularizers are applied to the same step-edge map with the
        same ``lam``; the jump remaining across the edge after TV must be
        larger than the jump remaining after Tikhonov.
        """
        m = np.zeros((10, 10))
        m[:, 5:] = 5.0
        mask = np.ones((10, 10), dtype=bool)
        tv = TotalVariationRegularizer(lam=1.0, mode="isotropic", n_iter=100)
        tik = TikhonovRegularizer(lam=1.0)
        out_tv = tv.prox([m], mask, step=1.0)[0]
        out_tik = tik.prox([m], mask, step=1.0)[0]
        jump_tv = out_tv[5, 5] - out_tv[5, 4]
        jump_tik = out_tik[5, 5] - out_tik[5, 4]
        assert jump_tv > jump_tik

    def test_anisotropic_and_isotropic_both_smooth_a_spike(self):
        """Both TV modes must reduce an isolated spike's magnitude.

        Isotropic vs. anisotropic mode changes how diagonal edges are
        treated (see the class docstring of
        `~pdrtpy.regularization.total_variation.TotalVariationRegularizer`),
        but both must still smooth simple isolated noise.
        """
        m = np.zeros((7, 7))
        m[3, 3] = 5.0
        mask = np.ones((7, 7), dtype=bool)
        for mode in ("isotropic", "anisotropic"):
            reg = TotalVariationRegularizer(lam=1.0, mode=mode, n_iter=100)
            out = reg.prox([m], mask, step=1.0)[0]
            assert out[3, 3] < 5.0


# ──────────────────────────────────────────────────────────────
# fista driver
# ──────────────────────────────────────────────────────────────


class TestFista:
    """Tests for the `~pdrtpy.regularization.base.fista` proximal-gradient solver."""

    def test_pure_denoising_matches_direct_prox(self):
        """When ``grad_fn`` is exactly ``(x - y)``, FISTA on
        ``0.5*||x-y||^2 + R(x)`` should converge to ``prox_R(y)`` directly.

        This is the textbook special case where the smooth data-fidelity
        term's minimizer over ``x`` for any fixed proximal step is just
        ``y`` itself, so the fixed point of proximal-gradient iteration
        must coincide with a single evaluation of `prox` on ``y`` — a
        strong end-to-end check that the FISTA driver in
        `~pdrtpy.regularization.base.fista` is implemented correctly.
        """
        rng = np.random.default_rng(2)
        y = np.zeros((8, 8))
        y[:, 4:] = 5.0
        y += rng.normal(scale=0.3, size=y.shape)
        mask = np.ones((8, 8), dtype=bool)
        reg = TotalVariationRegularizer(lam=0.5, mode="isotropic", n_iter=100)

        def objective_fn(maps):
            """Data-fidelity term ``0.5*||maps[0]-y||^2`` for this test's toy problem.

            Parameters
            ----------
            maps : list of `~numpy.ndarray`
                Single-element list containing the current iterate.

            Returns
            -------
            float
                The objective value.
            """
            return 0.5 * np.sum((maps[0] - y) ** 2)

        def grad_fn(maps):
            """Gradient of `objective_fn` with respect to ``maps[0]``.

            Parameters
            ----------
            maps : list of `~numpy.ndarray`
                Single-element list containing the current iterate.

            Returns
            -------
            list of `~numpy.ndarray`
                Single-element list containing the gradient, ``maps[0] - y``.
            """
            return [maps[0] - y]

        x_final = fista([y.copy()], objective_fn, grad_fn, reg.prox, mask, step0=1.0, n_iter=200, tol=1e-10)[0]
        direct = reg.prox([y], mask, step=1.0)[0]
        assert np.allclose(x_final, direct, atol=1e-6)

    def test_multiple_maps_independent(self):
        """Two independent maps in the list should be regularized independently.

        A positive spike in the first map and a negative spike in the
        second map should each be pulled back toward zero, verifying that
        `~pdrtpy.regularization.base.fista` and
        `~pdrtpy.regularization.total_variation.TotalVariationRegularizer.prox`
        correctly handle a multi-map list rather than assuming exactly one map.
        """
        y1 = np.zeros((6, 6))
        y1[3, 3] = 5.0
        y2 = np.zeros((6, 6))
        y2[2, 2] = -5.0
        mask = np.ones((6, 6), dtype=bool)
        reg = TotalVariationRegularizer(lam=1.0, n_iter=100)

        def objective_fn(maps):
            """Sum of the two independent maps' data-fidelity terms.

            Parameters
            ----------
            maps : list of `~numpy.ndarray`
                ``[current estimate of y1, current estimate of y2]``.

            Returns
            -------
            float
                The combined objective value.
            """
            return 0.5 * np.sum((maps[0] - y1) ** 2) + 0.5 * np.sum((maps[1] - y2) ** 2)

        def grad_fn(maps):
            """Gradient of `objective_fn` with respect to each map.

            Parameters
            ----------
            maps : list of `~numpy.ndarray`
                ``[current estimate of y1, current estimate of y2]``.

            Returns
            -------
            list of `~numpy.ndarray`
                ``[maps[0] - y1, maps[1] - y2]``.
            """
            return [maps[0] - y1, maps[1] - y2]

        out = fista([y1.copy(), y2.copy()], objective_fn, grad_fn, reg.prox, mask, n_iter=100)
        assert out[0][3, 3] < 5.0
        assert out[1][2, 2] > -5.0  # pulled up toward 0, i.e. |value| shrinks
