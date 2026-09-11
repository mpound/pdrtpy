"""Tests for the tool-agnostic pdrtpy.regularization module."""

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
    def test_full_grid_4connectivity_edge_count(self):
        mask = np.ones((3, 3), dtype=bool)
        edges = neighbor_graph(mask, connectivity=4)
        # 2 horizontal edges per row * 3 rows + 2 vertical edges per column * 3 columns
        assert len(edges) == 12

    def test_full_grid_8connectivity_edge_count(self):
        mask = np.ones((3, 3), dtype=bool)
        edges = neighbor_graph(mask, connectivity=4)
        edges8 = neighbor_graph(mask, connectivity=8)
        assert len(edges8) > len(edges)

    def test_invalid_connectivity_raises(self):
        with pytest.raises(ValueError):
            neighbor_graph(np.ones((3, 3), dtype=bool), connectivity=5)

    def test_masked_pixel_produces_no_edges(self):
        mask = np.ones((3, 3), dtype=bool)
        mask[1, 1] = False
        edges = neighbor_graph(mask, connectivity=4)
        for a, b in edges:
            assert a != (1, 1)
            assert b != (1, 1)

    def test_no_wraparound_at_borders(self):
        mask = np.ones((1, 3), dtype=bool)
        edges = neighbor_graph(mask, connectivity=4)
        # only 2 horizontal edges in a 1x3 row; no edge should connect column 0 to column 2
        assert len(edges) == 2
        assert ((0, 0), (0, 2)) not in edges
        assert ((0, 2), (0, 0)) not in edges

    def test_disconnected_valid_regions_have_no_bridging_edge(self):
        mask = np.array([[True, False, True]])
        edges = neighbor_graph(mask, connectivity=4)
        assert edges == []


# ──────────────────────────────────────────────────────────────
# Regularizer ABC
# ──────────────────────────────────────────────────────────────


def test_regularizer_rejects_negative_lambda():
    with pytest.raises(ValueError):
        TikhonovRegularizer(lam=-1.0)
    with pytest.raises(ValueError):
        TotalVariationRegularizer(lam=-1.0)


def test_regularizer_is_abstract():
    with pytest.raises(TypeError):
        Regularizer(lam=1.0)


# ──────────────────────────────────────────────────────────────
# TikhonovRegularizer
# ──────────────────────────────────────────────────────────────


class TestTikhonovRegularizer:
    def test_constant_map_is_fixed_point(self):
        m = np.full((6, 6), 3.0)
        mask = np.ones((6, 6), dtype=bool)
        reg = TikhonovRegularizer(lam=2.0)
        out = reg.prox([m], mask, step=1.0)[0]
        assert np.allclose(out, 3.0, atol=1e-8)

    def test_zero_lambda_is_identity(self):
        rng = np.random.default_rng(0)
        m = rng.normal(size=(5, 5))
        mask = np.ones((5, 5), dtype=bool)
        reg = TikhonovRegularizer(lam=0.0)
        out = reg.prox([m], mask, step=1.0)[0]
        assert np.allclose(out, m)

    def test_invalid_pixels_unchanged(self):
        m = np.array([[1.0, 100.0], [1.0, 1.0]])
        mask = np.array([[True, False], [True, True]])
        reg = TikhonovRegularizer(lam=5.0)
        out = reg.prox([m], mask, step=1.0)[0]
        assert out[0, 1] == 100.0

    def test_smooths_isolated_spike(self):
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
    def test_invalid_mode_raises(self):
        with pytest.raises(ValueError):
            TotalVariationRegularizer(lam=1.0, mode="bogus")

    def test_zero_lambda_is_identity(self):
        rng = np.random.default_rng(1)
        m = rng.normal(size=(5, 5))
        mask = np.ones((5, 5), dtype=bool)
        reg = TotalVariationRegularizer(lam=0.0)
        out = reg.prox([m], mask, step=1.0)[0]
        assert np.allclose(out, m)

    def test_invalid_pixels_unchanged(self):
        m = np.array([[1.0, 100.0], [1.0, 1.0]])
        mask = np.array([[True, False], [True, True]])
        reg = TotalVariationRegularizer(lam=5.0, n_iter=50)
        out = reg.prox([m], mask, step=1.0)[0]
        assert out[0, 1] == 100.0

    def test_isolated_spike_is_smoothed_toward_neighbors(self):
        """An isolated single-pixel oscillation should be pulled toward its
        (zero-valued) neighbors' consensus."""
        m = np.zeros((7, 7))
        m[3, 3] = 5.0
        mask = np.ones((7, 7), dtype=bool)
        reg = TotalVariationRegularizer(lam=1.0, mode="isotropic", n_iter=100)
        out = reg.prox([m], mask, step=1.0)[0]
        assert out[3, 3] < 0.5 * m[3, 3]  # pulled well down from 5.0 toward the surrounding 0

    def test_genuine_multipixel_edge_survives(self):
        """A step edge spanning many contiguous pixels should stay close to
        sharp, unlike an isolated spike at the same lambda."""
        m = np.zeros((10, 10))
        m[:, 5:] = 5.0
        mask = np.ones((10, 10), dtype=bool)
        reg = TotalVariationRegularizer(lam=1.0, mode="isotropic", n_iter=100)
        out = reg.prox([m], mask, step=1.0)[0]
        # the jump across the edge should remain most of its original size...
        assert (out[5, 5] - out[5, 4]) > 3.0
        # ...while the interior on each side stays essentially flat at 0 / 5.
        assert out[5, 0] < 0.5
        assert out[5, 9] > 4.5

    def test_edge_preserved_better_than_tikhonov_at_matched_lambda(self):
        """The central claim of the design doc: at the same lambda, TV preserves
        a genuine multi-pixel edge much better than Tikhonov does."""
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
    def test_pure_denoising_matches_direct_prox(self):
        """When grad_fn is exactly (x - y), FISTA on 0.5||x-y||^2 + R(x)
        should converge to prox_R(y) directly."""
        rng = np.random.default_rng(2)
        y = np.zeros((8, 8))
        y[:, 4:] = 5.0
        y += rng.normal(scale=0.3, size=y.shape)
        mask = np.ones((8, 8), dtype=bool)
        reg = TotalVariationRegularizer(lam=0.5, mode="isotropic", n_iter=100)

        def objective_fn(maps):
            return 0.5 * np.sum((maps[0] - y) ** 2)

        def grad_fn(maps):
            return [maps[0] - y]

        x_final = fista([y.copy()], objective_fn, grad_fn, reg.prox, mask, step0=1.0, n_iter=200, tol=1e-10)[0]
        direct = reg.prox([y], mask, step=1.0)[0]
        assert np.allclose(x_final, direct, atol=1e-6)

    def test_multiple_maps_independent(self):
        """Two independent maps in the list should be regularized independently."""
        y1 = np.zeros((6, 6))
        y1[3, 3] = 5.0
        y2 = np.zeros((6, 6))
        y2[2, 2] = -5.0
        mask = np.ones((6, 6), dtype=bool)
        reg = TotalVariationRegularizer(lam=1.0, n_iter=100)

        def objective_fn(maps):
            return 0.5 * np.sum((maps[0] - y1) ** 2) + 0.5 * np.sum((maps[1] - y2) ** 2)

        def grad_fn(maps):
            return [maps[0] - y1, maps[1] - y2]

        out = fista([y1.copy(), y2.copy()], objective_fn, grad_fn, reg.prox, mask, n_iter=100)
        assert out[0][3, 3] < 5.0
        assert out[1][2, 2] > -5.0  # pulled up toward 0, i.e. |value| shrinks
