"""Comprehensive tests for ModelPlot, LineRatioPlot, and PlotBase."""
import matplotlib

matplotlib.use("Agg")

import numpy as np
import pytest
import astropy.units as u
from astropy.nddata import StdDevUncertainty

import pdrtpy.pdrutils as utils
from pdrtpy.measurement import Measurement
from pdrtpy.modelset import ModelSet
from pdrtpy.plot.modelplot import ModelPlot
from pdrtpy.plot.lineratioplot import LineRatioPlot
from pdrtpy.tool.lineratiofit import LineRatioFit


# ──────────────────────────────────────────────────────────────
# Module-scoped fixtures (expensive – created once per test run)
# ──────────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def wk2020():
    return ModelSet("wk2020", z=1)


@pytest.fixture(scope="module")
def single_pixel_fit(wk2020):
    """Single-pixel LineRatioFit result, shared across tests."""
    myunit = "erg s-1 cm-2 sr-1"
    m1 = Measurement(data=3.6e-4, uncertainty=StdDevUncertainty(1.2e-4), identifier="OI_63", unit=myunit)
    m2 = Measurement(data=1e-6, uncertainty=StdDevUncertainty([3e-7]), identifier="CI_609", unit=myunit)
    m3 = Measurement(
        data=26, uncertainty=StdDevUncertainty([5]), identifier="CO_43", restfreq="461.04077 GHz", unit="K km/s"
    )
    m4 = Measurement(data=8e-5, uncertainty=StdDevUncertainty([8e-6]), identifier="CII_158", unit=myunit)
    p = LineRatioFit(wk2020, measurements=[m1, m2, m3, m4])
    p.run()
    return p


@pytest.fixture(scope="module")
def map_fit():
    """Map-based LineRatioFit result, shared across tests."""
    import tempfile, os

    cii_flux = utils.get_testdata("n22_cii_flux.fits")
    cii_err = utils.get_testdata("n22_cii_error.fits")
    oi_flux = utils.get_testdata("n22_oi_flux.fits")
    oi_err = utils.get_testdata("n22_oi_error.fits")
    fir_flux = utils.get_testdata("n22_FIR.fits")

    tmpdir = tempfile.mkdtemp()
    cii_comb = os.path.join(tmpdir, "cii.fits")
    oi_comb = os.path.join(tmpdir, "oi.fits")
    fir_comb = os.path.join(tmpdir, "fir.fits")

    Measurement.make_measurement(cii_flux, cii_err, cii_comb, overwrite=True)
    Measurement.make_measurement(oi_flux, oi_err, oi_comb, overwrite=True)
    Measurement.make_measurement(fir_flux, error="10%", outfile=fir_comb, overwrite=True)

    cii_meas = Measurement.read(cii_comb, identifier="CII_158")
    fir_meas = Measurement.read(fir_comb, identifier="FIR")
    oi_meas = Measurement.read(oi_comb, identifier="OI_63")

    smcmod = ModelSet("smc", z=0.1)
    p = LineRatioFit(modelset=smcmod, measurements=[cii_meas, fir_meas, oi_meas])
    p.run()
    return p


# ──────────────────────────────────────────────────────────────
# Per-test fixtures (closed after each test)
# ──────────────────────────────────────────────────────────────


@pytest.fixture
def mp(wk2020):
    import matplotlib.pyplot as plt

    plot = ModelPlot(wk2020)
    yield plot
    plt.close("all")


@pytest.fixture
def lrp(single_pixel_fit):
    import matplotlib.pyplot as plt

    plot = LineRatioPlot(single_pixel_fit)
    yield plot
    plt.close("all")


@pytest.fixture
def lrp_map(map_fit):
    import matplotlib.pyplot as plt

    plot = LineRatioPlot(map_fit)
    yield plot
    plt.close("all")


# ──────────────────────────────────────────────────────────────
# PlotBase internal helpers  (accessed via ModelPlot / LineRatioPlot)
# ──────────────────────────────────────────────────────────────


class TestPlotBaseHelpers:
    def test_autolevels_log(self, mp):
        data = np.logspace(0, 3, 100)
        levels = mp._autolevels(data, steps="log")
        assert len(levels) >= 5
        assert len(levels) <= 15
        assert np.all(np.diff(levels) > 0)  # monotonically increasing

    def test_autolevels_lin(self, mp):
        data = np.linspace(1, 100, 50)
        levels = mp._autolevels(data, steps="lin")
        assert len(levels) >= 5
        assert np.all(np.diff(levels) > 0)

    def test_autolevels_numlevels(self, mp):
        data = np.logspace(0, 3, 100)
        levels = mp._autolevels(data, steps="log", numlevels=7)
        assert len(levels) == 7

    def test_autolevels_bad_steps_raises(self, mp):
        data = np.array([1.0, 10.0, 100.0])
        with pytest.raises(ValueError):
            mp._autolevels(data, steps="bad_steps")

    def test_autolevels_nonpositive_min(self, mp):
        """Min <= 0 should be handled (clamped to 1e-10)"""
        data = np.array([-1.0, 0.0, 1.0, 10.0])
        levels = mp._autolevels(data, steps="log")
        assert len(levels) >= 5

    def test_zscale_linear(self, mp, wk2020):
        model = wk2020.get_model("OI_63/CII_158")
        norm = mp._zscale(model.data, vmin=None, vmax=None, stretch="linear")
        assert norm is not None

    def test_zscale_sqrt(self, mp, wk2020):
        model = wk2020.get_model("OI_63/CII_158")
        norm = mp._zscale(model.data, vmin=None, vmax=None, stretch="sqrt")
        assert norm is not None

    def test_zscale_log(self, mp, wk2020):
        model = wk2020.get_model("OI_63/CII_158")
        norm = mp._zscale(model.data, vmin=None, vmax=None, stretch="log")
        assert norm is not None

    def test_zscale_asinh(self, mp, wk2020):
        model = wk2020.get_model("OI_63/CII_158")
        norm = mp._zscale(model.data, vmin=None, vmax=None, stretch="asinh")
        assert norm is not None

    def test_zscale_power(self, mp, wk2020):
        model = wk2020.get_model("OI_63/CII_158")
        norm = mp._zscale(model.data, vmin=None, vmax=None, stretch="power")
        assert norm is not None

    def test_zscale_bad_stretch_raises(self, mp, wk2020):
        model = wk2020.get_model("OI_63/CII_158")
        with pytest.raises(ValueError):
            mp._zscale(model.data, vmin=None, vmax=None, stretch="invalid")

    def test_get_norm_simple(self, mp, wk2020):
        model = wk2020.get_model("OI_63/CII_158")
        norm = mp._get_norm("simple", model.data, vmin=0.1, vmax=10.0, stretch="linear")
        assert norm is not None

    def test_get_norm_zscale(self, mp, wk2020):
        model = wk2020.get_model("OI_63/CII_158")
        norm = mp._get_norm("zscale", model.data, vmin=0.1, vmax=10.0, stretch="linear")
        assert norm is not None

    def test_get_norm_log(self, mp, wk2020):
        model = wk2020.get_model("OI_63/CII_158")
        norm = mp._get_norm("log", model.data, vmin=0.1, vmax=10.0, stretch="linear")
        assert norm is not None

    def test_get_norm_bad_norm_raises(self, mp, wk2020):
        model = wk2020.get_model("OI_63/CII_158")
        with pytest.raises(ValueError, match="Unrecognized normalization"):
            mp._get_norm("bad_norm", model.data, vmin=0.1, vmax=10.0, stretch="linear")

    def test_get_norm_bad_stretch_raises(self, mp, wk2020):
        model = wk2020.get_model("OI_63/CII_158")
        with pytest.raises(ValueError, match="Unrecognized stretch"):
            mp._get_norm("simple", model.data, vmin=0.1, vmax=10.0, stretch="bad_stretch")

    def test_text(self, mp):
        """text() should add text to the current axis without raising"""
        mp.ratio("OI_63/CII_158", legend=False)
        # This just calls through to matplotlib - should not raise
        mp.text(2.0, 3.0, "hello")


# ──────────────────────────────────────────────────────────────
# ModelPlot
# ──────────────────────────────────────────────────────────────


class TestModelPlot:
    def test_plot_routes_to_ratio(self, mp):
        mp.plot("OI_63/CII_158")
        assert mp.figure is not None

    def test_plot_routes_to_intensity(self, mp):
        mp.plot("CII_158")
        assert mp.figure is not None

    def test_ratio_basic(self, mp):
        mp.ratio("OI_63/CII_158")
        assert mp.figure is not None
        assert mp.axis is not None

    def test_ratio_no_image(self, mp):
        mp.ratio("OI_63/CII_158", image=False, colors=["black"])
        assert mp.figure is not None

    def test_ratio_no_legend(self, mp):
        mp.ratio("OI_63/CII_158", legend=False)
        assert mp.figure is not None

    def test_ratio_with_measurement_overlay(self, mp):
        m = Measurement(
            data=0.5,
            uncertainty=StdDevUncertainty(0.1),
            unit=u.dimensionless_unscaled,
            identifier="OI_63/CII_158",
        )
        mp.ratio("OI_63/CII_158", measurements=[m, m], meas_color=["red", "blue"])
        assert mp.figure is not None

    def test_ratio_yaxis_unit(self, mp):
        mp.ratio("OI_63/CII_158", yaxis_unit="Habing")
        assert mp.figure is not None

    def test_intensity_basic(self, mp):
        mp.intensity("CII_158")
        assert mp.figure is not None

    def test_intensity_no_legend(self, mp):
        mp.intensity("CII_158", legend=False)
        assert mp.figure is not None

    def test_intensity_wrong_unit_raises(self, mp, wk2020):
        model = wk2020.get_model("CII_158")
        m = Measurement(
            data=1.0,
            uncertainty=StdDevUncertainty(0.1),
            unit="K km s-1",  # wrong unit
            identifier="CII_158",
        )
        with pytest.raises(TypeError):
            mp.intensity("CII_158", measurements=[m])

    def test_intensity_wrong_type_raises(self, mp):
        with pytest.raises(TypeError):
            mp.intensity("CII_158", measurements=["not_a_measurement"])

    def test_intensity_mismatched_id_warns(self, mp, wk2020):
        model = wk2020.get_model("CII_158")
        m = Measurement(
            data=1.0,
            uncertainty=StdDevUncertainty(0.1),
            unit=model._unit,
            identifier="OI_63",  # wrong identifier
        )
        import warnings

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            mp.intensity("CII_158", measurements=[m])
        assert any("do not match" in str(warning.message) for warning in w)

    def test_phasespace_basic(self, mp):
        mp.phasespace(
            ["CII_158/FIR", "CII_158/CO_32"],
            nax1_clip=[1e2, 1e5] * u.Unit("cm-3"),
            nax2_clip=[1e1, 1e6] * utils.habing_unit,
        )
        assert mp.figure is not None

    def test_phasespace_wrong_num_identifiers_raises(self, mp):
        with pytest.raises(ValueError, match="exactly 2"):
            mp.phasespace(
                ["CII_158/FIR"],  # only 1, needs 2
                nax1_clip=[1e2, 1e5] * u.Unit("cm-3"),
                nax2_clip=[1e1, 1e6] * utils.habing_unit,
            )

    def test_phasespace_with_measurements(self, mp):
        m1 = Measurement.from_table(utils.get_testdata("cii-fir-nc.tab"))
        m2 = Measurement.from_table(utils.get_testdata("cii-co-nc.tab"))
        mp.phasespace(
            ["CII_158/FIR", "CII_158/CO_32"],
            nax1_clip=[1e2, 1e5] * u.Unit("cm-3"),
            nax2_clip=[1e1, 1e6] * utils.habing_unit,
            measurements=[m1, m2],
            fmt=["ks"],
            label=["nc"],
        )
        assert mp.figure is not None


# ──────────────────────────────────────────────────────────────
# LineRatioPlot – single-pixel
# ──────────────────────────────────────────────────────────────


class TestLineRatioPlotSinglePixel:
    def test_density_returns_value(self, lrp):
        """Single-pixel density() returns a unit-converted value, not a plot"""
        result = lrp.density()
        assert result is not None

    def test_density_custom_units(self, lrp):
        result = lrp.density(units="cm^-3")
        assert result is not None

    def test_radiation_field_returns_value(self, lrp):
        result = lrp.radiation_field()
        assert result is not None

    def test_radiation_field_habing(self, lrp):
        result = lrp.radiation_field(units="Habing")
        assert result is not None

    def test_radiation_field_draine(self, lrp):
        result = lrp.radiation_field(units="Draine")
        assert result is not None

    def test_show_both_single_pixel(self, lrp):
        """show_both on a single-pixel fit returns (rf, density) tuple"""
        result = lrp.show_both()
        assert result is not None
        assert len(result) == 2

    def test_modelintensity(self, lrp):
        lrp.modelintensity("CII_158")
        assert lrp.figure is not None

    def test_modelintensity_bad_id_raises(self, lrp):
        with pytest.raises(KeyError):
            lrp.modelintensity("not_a_real_line")

    def test_modelratio(self, lrp, single_pixel_fit):
        # Use the first ratio in the fit's model ratios
        first_ratio = list(single_pixel_fit._modelratios.keys())[0]
        lrp.modelratio(first_ratio)
        assert lrp.figure is not None

    def test_chisq_single_pixel(self, lrp):
        lrp.chisq()
        assert lrp.figure is not None

    def test_chisq_image_false(self, lrp):
        lrp.chisq(image=False, colors=["black"])
        assert lrp.figure is not None

    def test_chisq_with_legend(self, lrp):
        lrp.chisq(legend=True)
        assert lrp.figure is not None

    def test_chisq_vectors_raises(self, single_pixel_fit):
        """chisq should raise NotImplementedError for vector data"""
        import matplotlib.pyplot as plt
        from unittest.mock import patch

        plot = LineRatioPlot(single_pixel_fit)
        with patch.object(type(single_pixel_fit), "has_vectors", new_callable=lambda: property(lambda self: True)):
            with pytest.raises(NotImplementedError):
                plot.chisq()
        plt.close("all")

    def test_reduced_chisq_single_pixel(self, lrp):
        lrp.reduced_chisq()
        assert lrp.figure is not None

    def test_reduced_chisq_image_false(self, lrp):
        lrp.reduced_chisq(image=False, colors=["black"])
        assert lrp.figure is not None

    def test_confidence_intervals(self, lrp):
        lrp.confidence_intervals()
        assert lrp.figure is not None

    def test_confidence_intervals_custom_levels(self, lrp):
        lrp.confidence_intervals(levels=[50.0, 68.0, 95.0])
        assert lrp.figure is not None

    def test_overlay_all_ratios(self, lrp):
        lrp.overlay_all_ratios()
        assert lrp.figure is not None

    def test_overlay_all_ratios_no_legend(self, lrp):
        lrp.overlay_all_ratios(legend=False)
        assert lrp.figure is not None

    def test_ratios_on_models(self, lrp):
        lrp.ratios_on_models()
        assert lrp.figure is not None

    def test_ratios_on_models_ncols(self, lrp):
        lrp.ratios_on_models(ncols=2)
        assert lrp.figure is not None


# ──────────────────────────────────────────────────────────────
# LineRatioPlot – map data
# ──────────────────────────────────────────────────────────────


class TestLineRatioPlotMap:
    def test_density_map_plot(self, lrp_map):
        lrp_map.density()
        assert lrp_map.figure is not None

    def test_density_map_with_contours(self, lrp_map):
        lrp_map.density(contours=True, norm="log")
        assert lrp_map.figure is not None

    def test_radiation_field_map_plot(self, lrp_map):
        lrp_map.radiation_field()
        assert lrp_map.figure is not None

    def test_radiation_field_map_draine(self, lrp_map):
        lrp_map.radiation_field(units="Draine")
        assert lrp_map.figure is not None

    def test_show_both_map(self, lrp_map):
        import matplotlib.pyplot as plt

        lrp_map.show_both()
        assert lrp_map.figure is not None
        plt.close("all")

    def test_chisq_map(self, lrp_map):
        lrp_map.chisq()
        assert lrp_map.figure is not None

    def test_reduced_chisq_map(self, lrp_map):
        lrp_map.reduced_chisq()
        assert lrp_map.figure is not None

    def test_confidence_intervals_map_raises(self, lrp_map):
        with pytest.raises(NotImplementedError):
            lrp_map.confidence_intervals()

    def test_overlay_all_ratios_map_raises(self, lrp_map):
        with pytest.raises(NotImplementedError):
            lrp_map.overlay_all_ratios()

    def test_ratios_on_models_map_raises(self, lrp_map):
        with pytest.raises(NotImplementedError):
            lrp_map.ratios_on_models()
