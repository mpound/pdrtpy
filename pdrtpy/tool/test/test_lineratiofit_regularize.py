"""Integration tests for LineRatioFit.regularize()."""

import numpy as np
import pdrtpy.utils as utils
import pytest
from astropy.nddata import StdDevUncertainty
from pdrtpy.measurement import Measurement
from pdrtpy.modelset import ModelSet
from pdrtpy.tool.lineratiofit import LineRatioFit

_MYUNIT = "erg s-1 cm-2 sr-1"

# ---------------------------------------------------------------------------
# Module-scoped fixtures (mirrors pdrtpy/tool/test/test_lineratiofit.py)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def wk2020():
    return ModelSet("wk2020", z=1)


@pytest.fixture(scope="module")
def smc_ms():
    return ModelSet("smc", z=0.2)


@pytest.fixture(scope="module")
def single_pixel_measurements():
    m1 = Measurement(data=3.6e-4, uncertainty=StdDevUncertainty(1.2e-4), identifier="OI_63", unit=_MYUNIT)
    m2 = Measurement(data=1e-6, uncertainty=StdDevUncertainty([3e-7]), identifier="CI_609", unit=_MYUNIT)
    m3 = Measurement(
        data=26, uncertainty=StdDevUncertainty([5]), identifier="CO_43", restfreq="461.04077 GHz", unit="K km/s"
    )
    m4 = Measurement(data=8e-5, uncertainty=StdDevUncertainty([8e-6]), identifier="CII_158", unit=_MYUNIT)
    return [m1, m2, m3, m4]


@pytest.fixture(scope="module")
def map_measurements(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("reg_maps")
    cii_flux = utils.get_testdata("n22_cii_flux.fits")
    cii_err = utils.get_testdata("n22_cii_error.fits")
    oi_flux = utils.get_testdata("n22_oi_flux.fits")
    oi_err = utils.get_testdata("n22_oi_error.fits")
    fir_flux = utils.get_testdata("n22_FIR.fits")

    cii_out = str(tmp / "cii.fits")
    oi_out = str(tmp / "oi.fits")
    fir_out = str(tmp / "fir.fits")

    Measurement.make_measurement(cii_flux, cii_err, outfile=cii_out, overwrite=True)
    Measurement.make_measurement(oi_flux, oi_err, outfile=oi_out, overwrite=True)
    Measurement.make_measurement(fir_flux, error="10%", outfile=fir_out, overwrite=True)

    cii_meas = Measurement.read(cii_out, identifier="CII_158")
    oi_meas = Measurement.read(oi_out, identifier="OI_63")
    fir_meas = Measurement.read(fir_out, identifier="FIR")
    return [cii_meas, oi_meas, fir_meas]


@pytest.fixture
def map_fit(smc_ms, map_measurements):
    """Function-scoped (not module-scoped): tests below mutate .density.data
    in place to simulate an oscillating pixel, so each test needs its own fit."""
    p = LineRatioFit(smc_ms, measurements=map_measurements)
    p.run()
    return p


@pytest.fixture(scope="module")
def single_pixel_fit(wk2020, single_pixel_measurements):
    p = LineRatioFit(wk2020, measurements=single_pixel_measurements)
    p.run()
    return p


def _first_interior_valid_pixel(density_map):
    """Pick a valid pixel away from the mask edge, so it has full 4-connectivity."""
    valid = ~np.isnan(density_map)
    ny, nx = valid.shape
    candidates = np.argwhere(valid[1 : ny - 1, 1 : nx - 1]) + 1
    return tuple(candidates[len(candidates) // 2])


# ---------------------------------------------------------------------------
# Preconditions
# ---------------------------------------------------------------------------


class TestRegularizePreconditions:
    def test_raises_before_run(self, smc_ms, map_measurements):
        p = LineRatioFit(smc_ms, measurements=map_measurements)
        with pytest.raises(Exception):
            p.regularize()

    def test_raises_for_single_pixel_fit(self, single_pixel_fit):
        with pytest.raises(Exception):
            single_pixel_fit.regularize()

    def test_raises_for_unknown_method(self, map_fit):
        with pytest.raises(ValueError):
            map_fit.regularize(method="bogus")


# ---------------------------------------------------------------------------
# Behavior
# ---------------------------------------------------------------------------


class TestRegularizeBehavior:
    def test_does_not_mutate_original_fit(self, map_fit):
        original = map_fit.density.data.copy()
        map_fit.regularize(method="tv", lam=0.2, max_iter=30)
        assert np.array_equal(map_fit.density.data, original, equal_nan=True)

    def test_returns_and_stores_regularized_measurements(self, map_fit):
        dreg, rreg = map_fit.regularize(method="tv", lam=0.2, max_iter=30)
        assert dreg is map_fit.density_regularized
        assert rreg is map_fit.radiation_field_regularized
        assert dreg.data.shape == map_fit.density.data.shape
        assert dreg.unit == map_fit.density.unit

    def test_header_records_provenance(self, map_fit):
        dreg, _ = map_fit.regularize(method="tv", lam=0.25, mode="anisotropic", max_iter=30)
        assert dreg.header["REGMETH"] == "tv"
        assert dreg.header["REGLAM"] == 0.25
        assert dreg.header["REGMODE"] == "anisotropic"

    def test_uncertainty_preserved_from_prefit(self, map_fit):
        dreg, rreg = map_fit.regularize(method="tv", lam=0.2, max_iter=30)
        assert np.array_equal(dreg.uncertainty.array, map_fit.density.uncertainty.array, equal_nan=True)
        assert np.array_equal(rreg.uncertainty.array, map_fit.radiation_field.uncertainty.array, equal_nan=True)

    def test_oscillating_pixel_pulled_toward_neighbor_consensus(self, map_fit):
        """The scenario the design doc is written around: a pixel that landed on
        a different, near-degenerate solution than its neighbors should move
        back toward them after regularization."""
        r0, c0 = _first_interior_valid_pixel(map_fit.density.data)
        original_density = float(map_fit.density.data[r0, c0])

        # Perturb the pixel to a different (but still in-bounds) density value,
        # simulating an oscillation into an alternate near-degenerate solution.
        perturbed = min(original_density * 3.0, np.nanmax(map_fit.density.data) * 0.9)
        map_fit.density.data[r0, c0] = perturbed

        dreg, _ = map_fit.regularize(method="tv", lam=0.3, max_iter=150)

        assert abs(dreg.data[r0, c0] - original_density) < abs(perturbed - original_density)
        # unregularized fit result is untouched by the call
        assert map_fit.density.data[r0, c0] == perturbed

    def test_tikhonov_method_runs(self, map_fit):
        dreg, rreg = map_fit.regularize(method="tikhonov", lam=0.2, max_iter=30)
        assert np.all(np.isfinite(dreg.data[~np.isnan(dreg.data)]))
        assert np.all(np.isfinite(rreg.data[~np.isnan(rreg.data)]))
