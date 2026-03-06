"""Tests for LineRatioFit tool."""

import os

import pytest
from astropy.nddata import CCDData, NDData, StdDevUncertainty
from astropy.table import Table

import pdrtpy.pdrutils as utils
from pdrtpy.measurement import Measurement
from pdrtpy.modelset import ModelSet
from pdrtpy.tool.fitmap import FitMap
from pdrtpy.tool.lineratiofit import LineRatioFit

# ---------------------------------------------------------------------------
# Module-scoped fixtures
# ---------------------------------------------------------------------------

_MYUNIT = "erg s-1 cm-2 sr-1"


@pytest.fixture(scope="module")
def wk2020():
    return ModelSet("wk2020", z=1)


@pytest.fixture(scope="module")
def smc_ms():
    return ModelSet("smc", z=0.1)


@pytest.fixture(scope="module")
def single_pixel_measurements():
    """Four measurements matching Listing A.2 in the paper."""
    m1 = Measurement(data=3.6e-4, uncertainty=StdDevUncertainty(1.2e-4), identifier="OI_63", unit=_MYUNIT)
    m2 = Measurement(data=1e-6, uncertainty=StdDevUncertainty([3e-7]), identifier="CI_609", unit=_MYUNIT)
    m3 = Measurement(
        data=26, uncertainty=StdDevUncertainty([5]), identifier="CO_43", restfreq="461.04077 GHz", unit="K km/s"
    )
    m4 = Measurement(data=8e-5, uncertainty=StdDevUncertainty([8e-6]), identifier="CII_158", unit=_MYUNIT)
    return [m1, m2, m3, m4]


@pytest.fixture(scope="module")
def single_pixel_fit(wk2020, single_pixel_measurements):
    """Run a single-pixel LineRatioFit once for the whole module."""
    p = LineRatioFit(wk2020, measurements=single_pixel_measurements)
    p.run()
    return p


@pytest.fixture(scope="module")
def map_measurements(tmp_path_factory):
    """Create n22 map Measurements from FITS test data."""
    tmp = tmp_path_factory.mktemp("maps")
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


@pytest.fixture(scope="module")
def map_fit(smc_ms, map_measurements):
    """Run a map LineRatioFit once for the whole module."""
    p = LineRatioFit(smc_ms, measurements=map_measurements)
    p.run()
    return p


# ---------------------------------------------------------------------------
# TestLineRatioFitInit
# ---------------------------------------------------------------------------


class TestLineRatioFitInit:
    def _make_measurements(self):
        m1 = Measurement(data=3.6e-4, uncertainty=StdDevUncertainty(1.2e-4), identifier="OI_63", unit=_MYUNIT)
        m2 = Measurement(data=8e-5, uncertainty=StdDevUncertainty([8e-6]), identifier="CII_158", unit=_MYUNIT)
        return m1, m2

    def test_init_with_list(self, wk2020):
        m1, m2 = self._make_measurements()
        p = LineRatioFit(wk2020, measurements=[m1, m2])
        assert "OI_63" in p.measurementIDs
        assert "CII_158" in p.measurementIDs

    def test_init_with_dict(self, wk2020):
        m1, m2 = self._make_measurements()
        p = LineRatioFit(wk2020, measurements={"OI_63": m1, "CII_158": m2})
        assert "OI_63" in p.measurementIDs
        assert "CII_158" in p.measurementIDs

    def test_init_with_tuple(self, wk2020):
        m1, m2 = self._make_measurements()
        p = LineRatioFit(wk2020, measurements=(m1, m2))
        assert "OI_63" in p.measurementIDs

    def test_init_with_none(self, wk2020):
        p = LineRatioFit(wk2020, measurements=None)
        assert p.measurementIDs is None

    def test_init_invalid_type(self, wk2020):
        with pytest.raises((ValueError, TypeError)):
            LineRatioFit(wk2020, measurements=42)


# ---------------------------------------------------------------------------
# TestLineRatioFitProperties
# ---------------------------------------------------------------------------


class TestLineRatioFitProperties:
    def test_modelset_property(self, single_pixel_fit):
        assert isinstance(single_pixel_fit.modelset, ModelSet)

    def test_measurements_property(self, single_pixel_fit):
        m = single_pixel_fit.measurements
        assert isinstance(m, dict)
        assert "OI_63" in m

    def test_measurementIDs_property(self, single_pixel_fit):
        ids = list(single_pixel_fit.measurementIDs)
        assert "OI_63" in ids
        assert "CII_158" in ids

    def test_fit_result_property(self, single_pixel_fit):
        fr = single_pixel_fit.fit_result
        assert fr is not None
        assert isinstance(fr, FitMap)

    def test_density_property(self, single_pixel_fit):
        d = single_pixel_fit.density
        assert d is not None
        assert d.data.flatten()[0] > 0

    def test_radiation_field_property(self, single_pixel_fit):
        rf = single_pixel_fit.radiation_field
        assert rf is not None
        assert rf.data.flatten()[0] > 0

    def test_density_has_unit(self, single_pixel_fit):
        assert single_pixel_fit.density.unit is not None

    def test_radiation_field_has_unit(self, single_pixel_fit):
        assert single_pixel_fit.radiation_field.unit is not None


# ---------------------------------------------------------------------------
# TestChisq
# ---------------------------------------------------------------------------


class TestChisq:
    def test_chisq_single_pixel(self, single_pixel_fit):
        cs = single_pixel_fit.chisq()
        assert cs is not None

    def test_chisq_single_pixel_min(self, single_pixel_fit):
        cs_min = single_pixel_fit.chisq(min=True)
        assert cs_min is not None
        val = cs_min.data.flatten()[0]
        import numpy as np

        assert np.isfinite(val)

    def test_reduced_chisq_single_pixel(self, single_pixel_fit):
        rcs = single_pixel_fit.reduced_chisq()
        assert rcs is not None

    def test_reduced_chisq_single_pixel_min(self, single_pixel_fit):
        import numpy as np

        rcs_min = single_pixel_fit.reduced_chisq(min=True)
        assert rcs_min is not None
        assert np.isfinite(rcs_min.data.flatten()[0])

    def test_chisq_map(self, map_fit):
        cs = map_fit.chisq()
        assert cs is not None
        # map chisq is a hypercube: dimensions > 2
        assert cs.data.ndim > 2

    def test_chisq_map_min(self, map_fit):
        cs_min = map_fit.chisq(min=True)
        assert cs_min is not None
        # min chisq for a map is a 2D spatial map
        assert cs_min.data.ndim == 2

    def test_reduced_chisq_map_min(self, map_fit):
        rcs_min = map_fit.reduced_chisq(min=True)
        assert rcs_min is not None
        assert rcs_min.data.ndim == 2


# ---------------------------------------------------------------------------
# TestObservedRatios
# ---------------------------------------------------------------------------


class TestObservedRatios:
    def test_observed_ratios_type(self, single_pixel_fit):
        or_ = single_pixel_fit.observed_ratios
        assert isinstance(or_, list)

    def test_observed_ratios_nonempty(self, single_pixel_fit):
        assert len(single_pixel_fit.observed_ratios) >= 1

    def test_ratiocount(self, single_pixel_fit):
        assert single_pixel_fit.ratiocount >= 2

    def test_observed_ratios_are_strings(self, single_pixel_fit):
        for r in single_pixel_fit.observed_ratios:
            assert isinstance(r, str)
            assert "/" in r


# ---------------------------------------------------------------------------
# TestAddRemoveMeasurement
# ---------------------------------------------------------------------------


class TestAddRemoveMeasurement:
    def test_add_measurement(self, wk2020, single_pixel_measurements):
        p = LineRatioFit(wk2020, measurements=list(single_pixel_measurements))
        m_new = Measurement(data=5e-5, uncertainty=StdDevUncertainty(1e-5), identifier="OI_145", unit=_MYUNIT)
        p.add_measurement(m_new)
        assert "OI_145" in p.measurementIDs

    def test_remove_measurement(self, wk2020, single_pixel_measurements):
        p = LineRatioFit(wk2020, measurements=list(single_pixel_measurements))
        p.remove_measurement("CI_609")
        assert "CI_609" not in p.measurementIDs

    def test_remove_invalid_id(self, wk2020, single_pixel_measurements):
        p = LineRatioFit(wk2020, measurements=list(single_pixel_measurements))
        with pytest.raises(KeyError):
            p.remove_measurement("NONEXISTENT_LINE")


# ---------------------------------------------------------------------------
# TestTable
# ---------------------------------------------------------------------------


class TestTable:
    def test_table_type(self, single_pixel_fit):
        t = single_pixel_fit.table
        assert isinstance(t, Table)

    def test_table_has_density_column(self, single_pixel_fit):
        t = single_pixel_fit.table
        assert "H2 Volume Density" in t.colnames

    def test_table_has_radiation_field_column(self, single_pixel_fit):
        t = single_pixel_fit.table
        assert "Radiation Field" in t.colnames

    def test_table_has_chisq_column(self, single_pixel_fit):
        t = single_pixel_fit.table
        assert "Chi-square" in t.colnames

    def test_table_has_measurement_columns(self, single_pixel_fit):
        t = single_pixel_fit.table
        assert "OI_63" in t.colnames
        assert "CII_158" in t.colnames


# ---------------------------------------------------------------------------
# TestWriteChisq
# ---------------------------------------------------------------------------


class TestWriteChisq:
    def test_write_chisq_creates_files(self, single_pixel_fit, tmp_path):
        chi_file = str(tmp_path / "chisq.fits")
        rchi_file = str(tmp_path / "rchisq.fits")
        single_pixel_fit.write_chisq(chi=chi_file, rchi=rchi_file, overwrite=True)
        assert os.path.isfile(chi_file)
        assert os.path.isfile(rchi_file)

    def test_write_chisq_files_are_valid_fits(self, single_pixel_fit, tmp_path):
        from astropy.io import fits

        chi_file = str(tmp_path / "chisq2.fits")
        rchi_file = str(tmp_path / "rchisq2.fits")
        single_pixel_fit.write_chisq(chi=chi_file, rchi=rchi_file, overwrite=True)
        with fits.open(chi_file) as hdul:
            assert len(hdul) >= 1
        with fits.open(rchi_file) as hdul:
            assert len(hdul) >= 1


# ---------------------------------------------------------------------------
# TestRunVariants
# ---------------------------------------------------------------------------


class TestRunVariants:
    def test_run_no_refine(self, wk2020, single_pixel_measurements):
        p = LineRatioFit(wk2020, measurements=list(single_pixel_measurements))
        p.run(refine=False)
        assert p.density is not None
        assert p.radiation_field is not None

    def test_run_insufficient_ratios(self, wk2020):
        """Only 2 measurements → 1 ratio → should raise Exception."""
        m1 = Measurement(data=3.6e-4, uncertainty=StdDevUncertainty(1.2e-4), identifier="OI_63", unit=_MYUNIT)
        m2 = Measurement(data=8e-5, uncertainty=StdDevUncertainty([8e-6]), identifier="CII_158", unit=_MYUNIT)
        p = LineRatioFit(wk2020, measurements=[m1, m2])
        with pytest.raises(Exception, match="[Nn]ot enough ratios"):
            p.run()

    def test_run_emcee(self, wk2020, single_pixel_measurements):
        p = LineRatioFit(wk2020, measurements=list(single_pixel_measurements))
        p.run(method="emcee", steps=500)
        assert p.fit_result is not None
        assert p.density is not None


# ---------------------------------------------------------------------------
# TestRunMasking
# ---------------------------------------------------------------------------


class TestRunMasking:
    def test_run_mask_mad(self, smc_ms, map_measurements):
        p = LineRatioFit(smc_ms, measurements=list(map_measurements))
        p.run(mask=["mad", 1.0])
        assert p.density is not None

    def test_run_mask_data(self, smc_ms, map_measurements):
        p = LineRatioFit(smc_ms, measurements=list(map_measurements))
        p.run(mask=["data", (0, 1e-10)])
        assert p.density is not None

    def test_run_mask_clip(self, smc_ms, map_measurements):
        p = LineRatioFit(smc_ms, measurements=list(map_measurements))
        p.run(mask=["clip", (1e-10, 1e3)])
        assert p.density is not None

    def test_run_mask_error(self, smc_ms, map_measurements):
        p = LineRatioFit(smc_ms, measurements=list(map_measurements))
        p.run(mask=["error", (0, 1e-10)])
        assert p.density is not None

    def test_run_mask_invalid(self, smc_ms, map_measurements):
        p = LineRatioFit(smc_ms, measurements=list(map_measurements))
        with pytest.raises(ValueError, match="[Uu]nrecognized mask"):
            p.run(mask=["bogus", 1.0])

    def test_run_mask_ignored_for_scalar(self, wk2020, single_pixel_measurements):
        """Mask kwarg should be silently ignored for single-pixel data."""
        p = LineRatioFit(wk2020, measurements=list(single_pixel_measurements))
        # Should complete without error (mask is ignored with a warning)
        p.run(mask=["mad", 1.0])
        assert p.density is not None


# ---------------------------------------------------------------------------
# TestOiCiiFir  (special combined ratio handling)
# ---------------------------------------------------------------------------


class TestOiCiiFir:
    def test_oi_63_cii_fir_ratio_present(self, map_measurements, smc_ms):
        """OI_63+CII_158/FIR should appear in observed_ratios when all three are present."""
        p = LineRatioFit(smc_ms, measurements=list(map_measurements))
        p.run()
        assert "OI_63+CII_158/FIR" in p.observed_ratios

    def test_oi_145_cii_fir_ratio_present(self, wk2020):
        """OI_145+CII_158/FIR should appear in observed_ratios when OI_145, CII_158, FIR are present."""
        m_oi145 = Measurement(data=5e-5, uncertainty=StdDevUncertainty(1e-5), identifier="OI_145", unit=_MYUNIT)
        m_cii = Measurement(data=8e-5, uncertainty=StdDevUncertainty(8e-6), identifier="CII_158", unit=_MYUNIT)
        m_fir = Measurement(data=5e-3, uncertainty=StdDevUncertainty(5e-4), identifier="FIR", unit=_MYUNIT)
        p = LineRatioFit(wk2020, measurements=[m_oi145, m_cii, m_fir])
        p.run()
        assert "OI_145+CII_158/FIR" in p.observed_ratios


# ---------------------------------------------------------------------------
# TestMapFit
# ---------------------------------------------------------------------------


class TestMapFit:
    def test_map_density_shape(self, map_fit):
        assert map_fit.density.data.ndim == 2

    def test_map_radiation_field_shape(self, map_fit):
        assert map_fit.radiation_field.data.ndim == 2

    def test_map_fit_result_type(self, map_fit):
        assert isinstance(map_fit.fit_result, FitMap)

    def test_map_density_positive_values_exist(self, map_fit):
        import numpy as np

        finite = map_fit.density.data[np.isfinite(map_fit.density.data)]
        assert len(finite) > 0
        assert (finite > 0).any()
