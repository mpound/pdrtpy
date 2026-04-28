#!/usr/bin/env python3
"""
Created on Wed Feb 26 16:01:56 2025

@author: mpound
"""

# h2excitation test code

import warnings

import numpy as np
import pdrtpy.pdrutils as utils
import pytest
from astropy.nddata import StdDevUncertainty
from dust_extinction.parameter_averages import G23
from pdrtpy.measurement import Measurement
from pdrtpy.tool.excitation import H2ExcitationFit
from pdrtpy.tool.fitmap import FitMap


def _make_h2_map_measurements(slc=None):
    """Build a list of six H2 H200S Measurements from FITS test data.

    If `slc` is given, slice the data to a smaller region (faster tests).
    """
    out = []
    for j in range(6):
        ident = f"H200S{j}"
        m = Measurement.read(utils.get_testdata(f"{ident}_test_data.fits"), identifier=ident)
        if slc is not None:
            data = m.data[slc]
            err = m.uncertainty.array[slc]
            mask = None if m.mask is None else m.mask[slc]
            m = Measurement(data=data, uncertainty=StdDevUncertainty(err), unit=m.unit, identifier=ident, mask=mask)
        out.append(m)
    return out


@pytest.fixture(scope="module")
def h2_map_measurements_small():
    """Six H2 H200S Measurements sliced to a 3x3 region for fast pixel-loop tests."""
    return _make_h2_map_measurements(slc=np.s_[45:48, 45:48])


@pytest.fixture(scope="module")
def h2_map_measurements_full():
    """Six full 100x100 H2 H200S Measurements for cutout tests (no fit run)."""
    return _make_h2_map_measurements()


@pytest.fixture(scope="module")
def h2_map_fit_two_component(h2_map_measurements_small):
    """Run H2ExcitationFit.run(components=2) once for the 3x3 fixture."""
    h = H2ExcitationFit(list(h2_map_measurements_small))
    h.run(components=2, verbose=False)
    return h


@pytest.fixture(scope="module")
def h2_map_fit_one_component(h2_map_measurements_small):
    """Run H2ExcitationFit.run(components=1) once for the 3x3 fixture."""
    h = H2ExcitationFit(list(h2_map_measurements_small))
    h.run(components=1, verbose=False)
    return h


class TestH2Excitation:
    """test the H2Excitation tool"""

    def setup_method(self):
        self._intensity = {}

    def test_fit_opr(self):
        self._intensity = {
            "H200S0": 3.00e-05,
            "H200S1": 5.16e-04,
            "H200S2": 3.71e-04,
            "H200S3": 1.76e-03,
            "H200S4": 5.28e-04,
            "H200S5": 9.73e-04,
        }
        a = []
        for i in self._intensity:
            # For this example, set a largish uncertainty on the intensity.
            m = Measurement(
                data=self._intensity[i],
                uncertainty=StdDevUncertainty(0.25 * self._intensity[i]),
                identifier=i,
                unit="erg cm-2 s-1 sr-1",
            )
            # print(m)
            a.append(m)
        h = H2ExcitationFit(a)
        # cd = h.column_densities(norm=False)

        # without opr
        h.run(fit_opr=False)
        assert h.thot.data == pytest.approx(692.82, rel=1e-3)
        assert h.tcold.data == pytest.approx(210.233, rel=1e-3)
        assert h.cold_colden.data == pytest.approx(1.802324e21, rel=1e-3)
        assert h.hot_colden.data == pytest.approx(2.0493e20, rel=1e-3)
        assert h.opr.data == pytest.approx(3.0, rel=1e-3)
        # with opr
        self._intensity = {
            "H200S0": 3.00e-05,
            "H200S1": 3.143e-4,
            "H200S2": 3.706e-04,
            "H200S3": 1.060e-03,
            "H200S4": 5.282e-04,
            "H200S5": 5.795e-04,
        }
        a = []
        for i in self._intensity:
            m = Measurement(
                data=self._intensity[i],
                uncertainty=StdDevUncertainty(0.25 * self._intensity[i]),
                identifier=i,
                unit="erg cm-2 s-1 sr-1",
            )
            a.append(m)
        h = H2ExcitationFit(a)

        h.run(fit_opr=True)
        assert h.thot.data == pytest.approx(687.467, rel=1e-3)
        assert h.tcold.data == pytest.approx(207.032, rel=1e-3)
        assert h.cold_colden.data == pytest.approx(1.835378e21, rel=1e-3)
        assert h.hot_colden.data == pytest.approx(2.07123e20, rel=1e-3)
        assert h.opr.data == pytest.approx(1.8615, rel=1e-3)

    def test_fit_av(self):
        # Previous data attenuated by Av=30 using Gordon 2023 extinction curve
        self._intensity = {
            "H200S0": 1.9604610972792677e-05,
            "H200S1": 0.00029530010810198534,
            "H200S2": 0.00021510356715925098,
            "H200S3": 0.0005620620764730837,
            "H200S4": 0.00030386531387721416,
            "H200S5": 0.000576178279762827,
        }
        a = []
        for i in self._intensity:
            # For this example, set a largish uncertainty on the intensity.
            m = Measurement(
                data=self._intensity[i],
                uncertainty=StdDevUncertainty(0.25 * self._intensity[i]),
                identifier=i,
                unit="erg cm-2 s-1 sr-1",
            )
            # print(m)
            a.append(m)
        h = H2ExcitationFit(a)
        g23 = G23(Rv=5.5)
        h.set_extinction_model(g23)
        h.run(fit_av=True)
        assert h.av.data == pytest.approx(27.750895, abs=1e-3)


class TestH2MapFit:
    """Tests covering the pixel-loop / map-fitting code path."""

    def test_two_component_shapes(self, h2_map_fit_two_component):
        h = h2_map_fit_two_component
        assert h.numcomponents == 2
        assert isinstance(h.fit_result, FitMap)
        assert h.fit_result.data.shape == (3, 3)
        assert h.tcold.data.shape == (3, 3)
        assert h.thot.data.shape == (3, 3)
        assert h.cold_colden.data.shape == (3, 3)
        assert h.hot_colden.data.shape == (3, 3)

    def test_two_component_thot_gt_tcold_where_finite(self, h2_map_fit_two_component):
        h = h2_map_fit_two_component
        finite = np.isfinite(h.tcold.data) & np.isfinite(h.thot.data)
        assert finite.any(), "expected at least one finite pixel"
        assert (h.thot.data[finite] >= h.tcold.data[finite]).all()

    def test_two_component_pinned_pixel(self, h2_map_fit_two_component):
        h = h2_map_fit_two_component
        # Reference pixel: middle of the 3x3 cutout (1, 1)
        i, j = 1, 1
        assert h.tcold.data[i, j] == pytest.approx(98.48048, rel=1e-3)
        assert h.thot.data[i, j] == pytest.approx(618.28602, rel=1e-3)
        assert h.cold_colden.data[i, j] == pytest.approx(1.03574e22, rel=1e-3)
        assert h.hot_colden.data[i, j] == pytest.approx(2.26282e20, rel=1e-3)

    def test_one_component_shapes(self, h2_map_fit_one_component):
        h = h2_map_fit_one_component
        assert h.numcomponents == 1
        assert h.tcold.data.shape == (3, 3)
        finite = np.isfinite(h.tcold.data)
        assert finite.any()
        # In one-component mode, tcold == thot
        assert np.array_equal(h.tcold.data, h.thot.data, equal_nan=True)

    def test_one_component_pinned_pixel(self, h2_map_fit_one_component):
        h = h2_map_fit_one_component
        i, j = 1, 1
        assert h.tcold.data[i, j] == pytest.approx(543.69904, rel=1e-3)
        assert h.cold_colden.data[i, j] == pytest.approx(4.17742e20, rel=1e-3)

    def test_nan_in_data_marks_pixel_bad(self, h2_map_measurements_small):
        # Inject NaN in pixel (0, 0) of one Measurement, refit, assert that pixel masked.
        # Build fresh Measurements rather than .copy() (which has a known astropy-units quirk).
        meas = []
        for k, m in enumerate(h2_map_measurements_small):
            d = m.data.copy()
            if k == 0:
                d[0, 0] = np.nan
            meas.append(
                Measurement(
                    data=d,
                    uncertainty=StdDevUncertainty(m.uncertainty.array.copy()),
                    unit=m.unit,
                    identifier=m.id,
                )
            )
        h = H2ExcitationFit(meas)
        h.run(components=2, verbose=False)
        assert h.fit_result.mask[0, 0]


class TestAverageColumnDensity:
    """Tests for average_column_density() input validation and cutout path."""

    def test_cutout_returns_all_keys(self, h2_map_measurements_full):
        h = H2ExcitationFit(list(h2_map_measurements_full))
        result = h.average_column_density(position=(50, 50), size=3, norm=True, line=True)
        assert set(result.keys()) == {f"H200S{j}" for j in range(6)}
        for ident, m in result.items():
            # Each result is a scalar Measurement (np.average over 3x3 cutout)
            assert np.isfinite(m.data), f"{ident} avg data is not finite"
            assert m.error > 0, f"{ident} error is not positive"

    def test_size_without_position_raises(self, h2_map_measurements_full):
        h = H2ExcitationFit(list(h2_map_measurements_full))
        with pytest.raises(Exception, match="position in addition to size"):
            h.average_column_density(size=3)


class TestH2RunErrors:
    """Tests for input-validation error paths in `.run()`."""

    def _few_point_measurements(self, n):
        intensity = {
            "H200S0": 3.00e-05,
            "H200S1": 5.16e-04,
            "H200S2": 3.71e-04,
            "H200S3": 1.76e-03,
            "H200S4": 5.28e-04,
            "H200S5": 9.73e-04,
        }
        out = []
        for k in list(intensity.keys())[:n]:
            out.append(
                Measurement(
                    data=intensity[k],
                    uncertainty=StdDevUncertainty(0.25 * intensity[k]),
                    identifier=k,
                    unit="erg cm-2 s-1 sr-1",
                )
            )
        return out

    def test_too_few_points_raises(self):
        # components=2 fit_opr=True needs 5 points; supply only 4.
        h = H2ExcitationFit(self._few_point_measurements(4))
        with pytest.raises(Exception, match="at least 5"):
            h.run(components=2, fit_opr=True)

    def test_over_constrained_warning(self):
        # components=2 fit_opr=False needs 4 points; supplying exactly 4 fires the warning.
        h = H2ExcitationFit(self._few_point_measurements(4))
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            h.run(components=2, fit_opr=False)
        assert any("over-constrained" in str(w.message) for w in caught)


class TestH2BadStderrCurrentBehavior:
    """Pin the *current* raise-on-bad-stderr behavior.

    This will be replaced by `TestH2BadStderrWarnAndMask` in the
    `_compute_quantities` refactor (Phase 3), which converts the raise
    to a `UserWarning` and per-pixel mask so that one bad pixel cannot
    abort an entire map fit.
    """

    def test_raises_when_any_param_stderr_is_none(self, h2_map_fit_two_component):
        h = h2_map_fit_two_component
        # Find a successful pixel to vandalize
        ff = h.fit_result.data.flatten()
        ffmask = h.fit_result.mask.flatten()
        good_idx = next(i for i, ok in enumerate(ffmask) if not ok and ff[i] is not None)
        # Save and restore to leave the module-scoped fixture untouched for other tests
        saved = ff[good_idx].params["m1"].stderr
        try:
            ff[good_idx].params["m1"].stderr = None
            with pytest.raises(Exception, match="Something went wrong with the fit"):
                h._compute_quantities(h._fitresult)
        finally:
            ff[good_idx].params["m1"].stderr = saved
            # Restore _compute_quantities-derived state so other tests using the fixture see clean values
            h._compute_quantities(h._fitresult)
