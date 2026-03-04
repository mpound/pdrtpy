"""Tests for pdrutils utility functions"""
import numpy as np
import numpy.ma as ma
import pytest
import astropy.units as u
from astropy.nddata import StdDevUncertainty
from astropy.wcs import WCS
from copy import deepcopy

import pdrtpy.pdrutils as utils
from pdrtpy.measurement import Measurement


# ──────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────

def make_measurement(data, unit, err=None, identifier="test"):
    if err is not None:
        unc = StdDevUncertainty(np.atleast_1d(np.array(err, dtype=float)))
    else:
        unc = None
    return Measurement(
        data=np.atleast_1d(np.array(data, dtype=float)),
        uncertainty=unc,
        unit=unit,
        identifier=identifier,
    )


def make_wcs_2d(crpix=(5, 5), cdelt=(0.01, 0.01), crval=(10.0, -70.0), naxis=(10, 10)):
    w = WCS(naxis=2)
    w.wcs.crpix = list(crpix)
    w.wcs.cdelt = list(cdelt)
    w.wcs.crval = list(crval)
    w.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    w._naxis = list(naxis)
    return w


# ──────────────────────────────────────────────────────────────
# Radiation-field unit conversions
# ──────────────────────────────────────────────────────────────

class TestRadiationFieldConversions:
    def test_toHabing_from_draine(self):
        m = make_measurement(1.0, "Draine")
        result = utils.toHabing(m)
        assert result.unit == u.Unit("Habing")
        expected = u.Unit("Draine").to("Habing")
        assert np.isclose(result.data[0], expected)

    def test_toDraine_from_habing(self):
        m = make_measurement(1.0, "Habing")
        result = utils.toDraine(m)
        assert result.unit == u.Unit("Draine")
        expected = u.Unit("Habing").to("Draine")
        assert np.isclose(result.data[0], expected)

    def test_toMathis_from_habing(self):
        m = make_measurement(1.0, "Habing")
        result = utils.toMathis(m)
        assert result.unit == u.Unit("Mathis")
        expected = u.Unit("Habing").to("Mathis")
        assert np.isclose(result.data[0], expected)

    def test_tocgs_from_habing(self):
        m = make_measurement(1.0, "Habing")
        result = utils.tocgs(m)
        # 1 Habing = 1.6e-3 erg/s/cm^2
        assert result.unit.is_equivalent(utils._RFS_UNIT_)
        assert np.isclose(result.data[0], 1.6e-3)

    def test_roundtrip_habing_draine(self):
        m = make_measurement(2.5, "Habing")
        result = utils.toHabing(utils.toDraine(m))
        assert np.isclose(result.data[0], 2.5, rtol=1e-10)

    def test_to_preserves_uncertainty(self):
        m = make_measurement(1.0, "Habing", err=0.1)
        result = utils.toDraine(m)
        assert result.uncertainty is not None
        # uncertainty should also be scaled
        expected_err = 0.1 * u.Unit("Habing").to("Draine")
        assert np.isclose(result.uncertainty.array[0], expected_err)

    def test_to_does_not_modify_original(self):
        m = make_measurement(1.0, "Habing")
        original_data = m.data[0]
        utils.toDraine(m)
        assert m.data[0] == original_data  # deepcopy ensures no mutation

    def test_get_rad_habing(self):
        assert utils.get_rad("Habing") == "$G_0$"

    def test_get_rad_draine(self):
        assert utils.get_rad("Draine") == "$\\chi$"

    def test_get_rad_mathis(self):
        assert utils.get_rad("Mathis") == "FUV"

    def test_get_rad_unknown(self):
        assert utils.get_rad("unknown_unit") == "FUV"

    def test_get_rad_unit_object(self):
        assert utils.get_rad(u.Unit("Habing")) == "$G_0$"


# ──────────────────────────────────────────────────────────────
# convert_integrated_intensity
# ──────────────────────────────────────────────────────────────

class TestConvertIntegratedIntensity:
    def _kkms_measurement(self, restfreq_hz=None):
        m = make_measurement(1.0, "K km s-1")
        if restfreq_hz is not None:
            m.header["RESTFREQ"] = restfreq_hz
        return m

    def test_conversion_with_wavelength(self):
        m = self._kkms_measurement()
        wavelength = u.Quantity(2.6e-1, "cm")  # approx CO 1-0
        result = utils.convert_integrated_intensity(m, wavelength=wavelength)
        assert result.unit == utils._OBS_UNIT_
        assert result.data[0] > 0

    def test_conversion_with_restfreq_in_header(self):
        m = self._kkms_measurement(restfreq_hz=115.271e9)
        result = utils.convert_integrated_intensity(m)
        assert result.unit == utils._OBS_UNIT_
        assert result.data[0] > 0

    def test_conversion_uncertainty_propagated(self):
        m = make_measurement(1.0, "K km s-1", err=0.1)
        m.header["RESTFREQ"] = 115.271e9
        result = utils.convert_integrated_intensity(m)
        assert result.uncertainty is not None
        assert result.uncertainty.array[0] > 0

    def test_conversion_no_restfreq_no_wavelength_raises(self):
        m = self._kkms_measurement()  # no RESTFREQ
        with pytest.raises(Exception, match="RESTFREQ"):
            utils.convert_integrated_intensity(m)

    def test_conversion_wrong_unit_raises(self):
        m = make_measurement(1.0, "erg s-1 cm-2 sr-1")
        with pytest.raises(Exception, match="BUNIT"):
            utils.convert_integrated_intensity(m, wavelength=u.Quantity(0.26, "cm"))

    def test_conversion_with_restfreq_attribute(self):
        """Test using _restfreq attribute instead of header RESTFREQ"""
        m = make_measurement(1.0, "K km s-1", identifier="CO_10")
        m._restfreq = u.Quantity(115.271, "GHz")
        result = utils.convert_integrated_intensity(m)
        assert result.unit == utils._OBS_UNIT_


# ──────────────────────────────────────────────────────────────
# mask_union
# ──────────────────────────────────────────────────────────────

class TestMaskUnion:
    def test_basic_union(self):
        a = ma.MaskedArray([1, 2, 3], mask=[True, False, False])
        b = ma.MaskedArray([1, 2, 3], mask=[False, True, False])
        result = utils.mask_union([a, b])
        assert np.all(result == np.array([True, True, False]))

    def test_no_masked_pixels(self):
        a = ma.MaskedArray([1, 2, 3], mask=[False, False, False])
        b = ma.MaskedArray([4, 5, 6], mask=[False, False, False])
        result = utils.mask_union([a, b])
        assert not np.any(result)

    def test_all_masked(self):
        a = ma.MaskedArray([1, 2], mask=[True, True])
        b = ma.MaskedArray([1, 2], mask=[True, True])
        result = utils.mask_union([a, b])
        assert np.all(result)


# ──────────────────────────────────────────────────────────────
# WCS helpers: dropaxis, has_single_axis, squeeze
# ──────────────────────────────────────────────────────────────

class TestWCSHelpers:
    def test_has_single_axis_true(self):
        w = WCS(naxis=3)
        w._naxis = [10, 1, 5]
        assert utils.has_single_axis(w) is True

    def test_has_single_axis_false(self):
        w = WCS(naxis=2)
        w._naxis = [10, 5]
        assert utils.has_single_axis(w) is False

    def test_dropaxis_removes_single_dim(self):
        # Put the single-dimension axis on FREQ (non-celestial), not a celestial axis
        w = WCS(naxis=3)
        w.wcs.crpix = [1, 1, 1]
        w.wcs.cdelt = [1, 1, 1]
        w.wcs.crval = [0, 0, 0]
        w.wcs.ctype = ["RA---TAN", "DEC--TAN", "FREQ"]
        w._naxis = [10, 5, 1]  # FREQ axis has size 1
        result = utils.dropaxis(w)
        assert result.naxis == 2

    def test_dropaxis_no_single_dim_unchanged(self):
        w = WCS(naxis=2)
        w._naxis = [10, 5]
        result = utils.dropaxis(w)
        assert result.naxis == 2  # unchanged

    def test_squeeze_removes_single_axis(self):
        """squeeze should drop single-dim WCS axes and squeeze data"""
        cii_file = utils.get_testdata("n22_cii_flux.fits")
        m = Measurement.read(cii_file, identifier="CII_158")
        original_naxis = m.wcs.naxis
        # ConvP_S1.fits has a 3D WCS; n22 files are 2D, test squeeze is no-op
        squeezed = utils.squeeze(m)
        assert squeezed.wcs.naxis <= original_naxis


# ──────────────────────────────────────────────────────────────
# fliplabel
# ──────────────────────────────────────────────────────────────

class TestFliplabel:
    def test_simple_flip(self):
        assert utils.fliplabel("OI_145/CII_158") == "CII_158/OI_145"

    def test_complex_numerator(self):
        assert utils.fliplabel("OI_63+CII_158/FIR") == "FIR/OI_63+CII_158"

    def test_no_slash_raises(self):
        with pytest.raises(ValueError):
            utils.fliplabel("OI_145")


# ──────────────────────────────────────────────────────────────
# Simple utility functions
# ──────────────────────────────────────────────────────────────

class TestSimpleUtils:
    def test_is_ratio_true(self):
        assert utils.is_ratio("OI_145/CII_158") is True

    def test_is_ratio_false(self):
        assert utils.is_ratio("OI_145") is False

    def test_is_ratio_slash_at_zero(self):
        # slash at position 0 should return False
        assert utils.is_ratio("/OI_145") is False

    def test_is_even(self):
        assert utils.is_even(2) is True
        assert utils.is_even(3) is False
        assert utils.is_even(-4) is True
        assert utils.is_even(0) is True

    def test_is_odd(self):
        assert utils.is_odd(3) is True
        assert utils.is_odd(2) is False
        assert utils.is_odd(-3) is True


# ──────────────────────────────────────────────────────────────
# get_xy_from_wcs
# ──────────────────────────────────────────────────────────────

class TestGetXYFromWCS:
    @pytest.fixture(autouse=True)
    def setup(self):
        from pdrtpy.modelset import ModelSet

        self.ms = ModelSet("wk2020", z=1)
        self.model = self.ms.get_model("OI_63/CII_158")

    def test_returns_two_arrays(self):
        x, y = utils.get_xy_from_wcs(self.model)
        assert x is not None
        assert y is not None

    def test_default_log_space(self):
        x, y = utils.get_xy_from_wcs(self.model)
        # values are in log space (small numbers like 1..7)
        assert np.all(np.isfinite(x))
        assert np.all(np.isfinite(y))

    def test_linear_space(self):
        x_lin, y_lin = utils.get_xy_from_wcs(self.model, linear=True)
        x_log, y_log = utils.get_xy_from_wcs(self.model, linear=False)
        # linear values should be larger than log values for typical grid
        assert np.nanmax(x_lin) >= np.nanmax(x_log)

    def test_quantity_mode(self):
        x, y = utils.get_xy_from_wcs(self.model, quantity=True)
        # Should be Quantity objects
        assert hasattr(x, "unit")
        assert hasattr(y, "unit")

    def test_quantity_linear(self):
        x, y = utils.get_xy_from_wcs(self.model, quantity=True, linear=True)
        assert hasattr(x, "unit")
        assert hasattr(y, "unit")
        assert np.all(np.isfinite(x.value))
