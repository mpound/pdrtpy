"""Synthetic ground-truth tests for LineRatioFit.regularize().

These tests build a map with a *known* true (density, radiation_field) at
every pixel, forward-modeled from a real :class:`~pdrtpy.modelset.ModelSet`,
rather than using the real N22 data exercised in
``test_lineratiofit_regularize.py``. This matters because N22 needs
``density_range``/``radiation_field_range`` restriction before regularizing
(see ``goals/regularization_design_options.md``) — its oscillation comes
from a genuine, map-wide, two-branch degeneracy that spatial smoothness
cannot and should not resolve on its own. A synthetic map with a *single*
true solution plus deliberately isolated, noise-driven outliers has no such
degeneracy, so it can demonstrate :meth:`~pdrtpy.tool.lineratiofit.LineRatioFit.regularize`
correcting genuine salt-and-pepper misfits with **no** phase-space
restriction — directly against known ground truth, which N22 cannot offer
since its true answer isn't known independently of the fit.

See ``goals/synthetic_regularization_test_findings.md`` for the full record
of what was tried before arriving at this design (two abandoned model/ratio
combinations, and the discovery that injecting a genuinely different truth
is the wrong test: a well-conditioned fit correctly and strongly resists
being overridden by a smoothness prior — only a pixel whose *data*, not its
*true parameters*, is an outlier is something regularization can legitimately
correct).
"""

import numpy as np
import pytest
from astropy.nddata import StdDevUncertainty
from pdrtpy.measurement import Measurement
from pdrtpy.modelset import ModelSet
from pdrtpy.tool.lineratiofit import LineRatioFit

# ---------------------------------------------------------------------------
# Synthetic map construction
# ---------------------------------------------------------------------------

# A single, well-conditioned solution branch (verified in
# goals/synthetic_regularization_test_findings.md to self-recover cleanly at
# 10% noise) used as the map's uniform background truth.
_BRANCH_A = (1.0e4, 0.032)  # (density [cm^-3], radiation_field [native wk2020 units], ~20 Habing)
# A second, well-separated, independently self-recovering point used only
# for the contiguous "real feature" block (not injected as a competing
# population — see the module docstring on why that distinction matters).
_BRANCH_B = (1.0e6, 1.0)
_LINE_IDS = ["OI_63", "CI_609", "CO_43", "CII_158"]
_NOISE = 0.10  # 10% relative, matching real-observation S/N (see findings doc)
_OUTLIER_LINE = "CO_43"
_OUTLIER_N_SIGMA = 2.5  # calibrated in the findings doc to a modestly (not catastrophically) correctable excursion
_SHAPE = (15, 15)
_BLOCK_ORIGIN = (10, 2)  # (row, col) of the block's top-left corner
_BLOCK_SIZE = 5
_N_SCATTERED = 10
_RNG_SEED = 99


def _block_slice():
    """The 2-D slice covering the contiguous Branch-B feature block.

    Returns
    -------
    tuple of slice
        ``(row_slice, col_slice)``, usable as a NumPy 2-D index.
    """
    r0, c0 = _BLOCK_ORIGIN
    return slice(r0, r0 + _BLOCK_SIZE), slice(c0, c0 + _BLOCK_SIZE)


def _scattered_pixel_coords(rng):
    """Choose non-adjacent scattered pixel coordinates, excluding the block.

    Each coordinate is at least 2 pixels (Chebyshev distance) from every
    other chosen coordinate, so no two scattered outliers are spatial
    neighbors of each other, and all are kept clear of the contiguous
    Branch-B block (with a 1-pixel margin) so a scattered outlier can never
    silently fall inside, or directly adjacent to, the block's footprint.

    Parameters
    ----------
    rng : `numpy.random.Generator`
        Random number generator used to draw candidate coordinates.

    Returns
    -------
    list of tuple
        ``(row, col)`` pairs, length `_N_SCATTERED`.
    """
    r0, c0 = _BLOCK_ORIGIN

    def in_or_near_block(r, c, pad=1):
        return (r0 - pad <= r < r0 + _BLOCK_SIZE + pad) and (c0 - pad <= c < c0 + _BLOCK_SIZE + pad)

    coords = []
    tries = 0
    while len(coords) < _N_SCATTERED and tries < 10000:
        tries += 1
        r, c = rng.integers(1, _SHAPE[0] - 1), rng.integers(1, _SHAPE[1] - 1)
        if in_or_near_block(r, c):
            continue
        if all(abs(r - rr) > 1 or abs(c - cc) > 1 for rr, cc in coords):
            coords.append((int(r), int(c)))
    if len(coords) < _N_SCATTERED:
        raise RuntimeError("could not place all scattered outlier pixels; shape/_N_SCATTERED mismatch")
    return coords


def _forward_model_intensities(modelset):
    """Build direct-lookup forward models for every line in `_LINE_IDS`.

    Parameters
    ----------
    modelset : `~pdrtpy.modelset.ModelSet`
        The model set to forward-model from (must have an intensity model
        for every identifier in `_LINE_IDS`).

    Returns
    -------
    dict
        Maps each identifier to its intensity `~pdrtpy.measurement.Measurement`
        (a 2-D (density, radiation_field) grid); call ``.get(n, g0)`` on the
        value to interpolate the model intensity at an arbitrary point.
    """
    return modelset.get_models(_LINE_IDS, model_type="intensity")


def _true_intensities(intensity_models, n, g0):
    """Evaluate every line's true (noiseless) intensity at one (n, G0) point.

    Parameters
    ----------
    intensity_models : dict
        Return value of `_forward_model_intensities`.
    n : float
        Density, in the model's native units.
    g0 : float
        Radiation field, in the model's native units.

    Returns
    -------
    dict
        Maps each identifier in `_LINE_IDS` to its true intensity at
        ``(n, g0)``.
    """
    return {line_id: model.get(n, g0) for line_id, model in intensity_models.items()}


def _build_synthetic_measurements(modelset):
    """Build the synthetic map's input `~pdrtpy.measurement.Measurement` objects.

    Constructs a `_SHAPE` map whose true (density, radiation_field) is
    `_BRANCH_A` everywhere except a contiguous `_BLOCK_SIZE`-by-`_BLOCK_SIZE`
    block (truly a different, independently self-recovering solution,
    `_BRANCH_B`), then adds realistic (`_NOISE`) multiplicative Gaussian
    noise to every pixel's forward-modeled flux. At `_N_SCATTERED` isolated,
    non-adjacent, non-block pixels, `_OUTLIER_LINE`'s flux additionally gets
    a deliberate `_OUTLIER_N_SIGMA`-sized offset — a single-line noise
    excursion calibrated (see the findings doc) to a modestly, not
    catastrophically, correctable displacement, modeling the "salt and
    pepper" misfit regularization is meant to fix. No pixel's *true*
    (density, radiation_field) differs from its neighbors because of this
    — only its *data* does, which is the scientifically correct scenario
    for `~pdrtpy.tool.lineratiofit.LineRatioFit.regularize` to act on.

    Parameters
    ----------
    modelset : `~pdrtpy.modelset.ModelSet`
        The model set to forward-model from.

    Returns
    -------
    tuple
        ``(measurements, scattered_coords)`` — ``measurements`` is a list of
        4 map `~pdrtpy.measurement.Measurement` objects (one per
        `_LINE_IDS` entry) ready to pass to `~pdrtpy.tool.lineratiofit.LineRatioFit`;
        ``scattered_coords`` is the list of ``(row, col)`` pairs returned by
        `_scattered_pixel_coords`.
    """
    intensity_models = _forward_model_intensities(modelset)
    truth_a = _true_intensities(intensity_models, *_BRANCH_A)
    truth_b = _true_intensities(intensity_models, *_BRANCH_B)

    rng = np.random.default_rng(_RNG_SEED)
    scattered_coords = _scattered_pixel_coords(rng)
    block_rows, block_cols = _block_slice()

    true_flux = {line_id: np.full(_SHAPE, truth_a[line_id]) for line_id in _LINE_IDS}
    for line_id in _LINE_IDS:
        true_flux[line_id][block_rows, block_cols] = truth_b[line_id]

    noisy_flux = {line_id: true_flux[line_id] * (1 + rng.normal(0, _NOISE, _SHAPE)) for line_id in _LINE_IDS}
    for r, c in scattered_coords:
        noisy_flux[_OUTLIER_LINE][r, c] = truth_a[_OUTLIER_LINE] * (1 + _OUTLIER_N_SIGMA * _NOISE)

    measurements = [
        Measurement(
            data=noisy_flux[line_id],
            uncertainty=StdDevUncertainty(_NOISE * np.abs(true_flux[line_id])),
            identifier=line_id,
            unit=str(intensity_models[line_id].unit),
        )
        for line_id in _LINE_IDS
    ]
    return measurements, scattered_coords


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def wk2020():
    """The wk2020 (z=1) model set used to forward-model the synthetic map."""
    return ModelSet("wk2020", z=1)


@pytest.fixture(scope="module")
def synthetic_fit(wk2020):
    """Run the synthetic map's fit once for the whole module (expensive: ~1 fit).

    Returns
    -------
    tuple
        ``(p, scattered_coords)`` — ``p`` is the `~pdrtpy.tool.lineratiofit.LineRatioFit`
        after `~pdrtpy.tool.lineratiofit.LineRatioFit.run` (no
        ``density_range``/``radiation_field_range`` anywhere), and
        ``scattered_coords`` is the list of deliberately-perturbed pixel
        coordinates.
    """
    measurements, scattered_coords = _build_synthetic_measurements(wk2020)
    p = LineRatioFit(wk2020, measurements=measurements)
    p.run()
    return p, scattered_coords


@pytest.fixture(scope="module")
def synthetic_regularized(synthetic_fit):
    """Run `regularize()` once on the synthetic fit for the whole module.

    Uses ``method="tv"`` at a lambda/max_iter combination verified (see
    ``goals/synthetic_regularization_test_findings.md``) to correct the
    scattered outliers substantially while leaving the block essentially
    untouched, under the default ``step_mode="global"`` — no
    ``density_range``/``radiation_field_range`` restriction is used or
    needed anywhere in this module.

    Returns
    -------
    tuple
        ``(density_regularized, radiation_field_regularized)``, the return
        value of :meth:`~pdrtpy.tool.lineratiofit.LineRatioFit.regularize`.
    """
    p, _ = synthetic_fit
    return p.regularize(method="tv", lam=25, max_iter=300, tol=0)


# ---------------------------------------------------------------------------
# Sanity checks on the synthetic map itself
# ---------------------------------------------------------------------------


class TestSyntheticMapSanity:
    """Checks that the synthetic map was constructed as intended, independent
    of `regularize()` — if any of these fail, the map itself (not
    regularization) needs attention."""

    def test_background_self_recovers(self, synthetic_fit):
        """Background pixels (neither scattered outliers nor the block) must
        recover close to `_BRANCH_A`'s true density — confirming this
        model/noise combination is well-posed, per the findings doc."""
        p, scattered_coords = synthetic_fit
        logn_fit = np.log10(p.density.data)
        is_special = np.zeros(_SHAPE, dtype=bool)
        for r, c in scattered_coords:
            is_special[r, c] = True
        block_rows, block_cols = _block_slice()
        is_special[block_rows, block_cols] = True
        is_background = ~is_special

        err = np.abs(logn_fit - np.log10(_BRANCH_A[0]))[is_background]
        assert np.max(err) < 0.3

    def test_scattered_pixels_are_modest_outliers(self, synthetic_fit):
        """The deliberately-perturbed pixels should show a real but modest
        displacement from truth before regularizing — not astronomically
        far (which would mean the data, not just noise, genuinely prefers a
        different answer; see the findings doc on why that's the wrong
        test)."""
        p, scattered_coords = synthetic_fit
        logn_fit = np.log10(p.density.data)
        truth_logn = np.log10(_BRANCH_A[0])
        err = np.array([abs(logn_fit[r, c] - truth_logn) for r, c in scattered_coords])
        assert np.all(err > 0.03)
        assert np.all(err < 0.3)

    def test_block_recovers_its_own_truth(self, synthetic_fit):
        """The contiguous block should recover close to `_BRANCH_B`'s true
        density — confirming it is a genuine, well-fit feature, not an
        artifact."""
        p, _ = synthetic_fit
        block_rows, block_cols = _block_slice()
        logn_block = np.log10(p.density.data[block_rows, block_cols])
        assert np.abs(logn_block.mean() - np.log10(_BRANCH_B[0])) < 0.1


# ---------------------------------------------------------------------------
# The actual regularization claims, against known ground truth
# ---------------------------------------------------------------------------


class TestRegularizeSyntheticMap:
    """The core claims this module exists to test: on a map with no
    map-wide solution degeneracy, `regularize()` corrects isolated,
    noise-driven misfits and preserves a genuine multi-pixel feature —
    with no phase-space restriction anywhere."""

    def test_scattered_outliers_move_closer_to_truth(self, synthetic_fit, synthetic_regularized):
        """Every deliberately-perturbed pixel's regularized density should be
        closer to the true background value than its unregularized fit was."""
        p, scattered_coords = synthetic_fit
        density_regularized, _ = synthetic_regularized
        logn_fit = np.log10(p.density.data)
        logn_reg = np.log10(density_regularized.data)
        truth_logn = np.log10(_BRANCH_A[0])

        err_before = np.array([abs(logn_fit[r, c] - truth_logn) for r, c in scattered_coords])
        err_after = np.array([abs(logn_reg[r, c] - truth_logn) for r, c in scattered_coords])

        assert np.all(err_after < err_before)
        assert err_after.mean() < 0.7 * err_before.mean()

    def test_block_is_preserved_not_eroded(self, synthetic_fit, synthetic_regularized):
        """The contiguous Branch-B block should change only slightly —
        regularization must not erode a genuine, spatially-supported
        feature into the surrounding background."""
        p, _ = synthetic_fit
        density_regularized, _ = synthetic_regularized
        block_rows, block_cols = _block_slice()

        logn_block_before = np.log10(p.density.data[block_rows, block_cols])
        logn_block_after = np.log10(density_regularized.data[block_rows, block_cols])

        assert np.abs(logn_block_after - logn_block_before).mean() < 0.1
        # still clearly separated from the background, not pulled toward it
        assert logn_block_after.mean() - np.log10(_BRANCH_A[0]) > 1.0

    def test_no_phase_space_restriction_used(self, synthetic_fit):
        """Confirms this module's claim by construction: the fit never sets
        a `density_range`/`radiation_field_range`, unlike the real N22 data
        (see ``goals/regularization_design_options.md``)."""
        p, _ = synthetic_fit
        assert p._density_range is None
        assert p._radiation_field_range is None
