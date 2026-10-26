"""Tests for LineRatioPlot.regularization_histogram."""

import matplotlib

matplotlib.use("Agg")

import pdrtpy.utils as utils
import pytest
from pdrtpy.measurement import Measurement
from pdrtpy.modelset import ModelSet
from pdrtpy.plot.lineratioplot import LineRatioPlot
from pdrtpy.tool.lineratiofit import LineRatioFit


@pytest.fixture(scope="module")
def map_fit(tmp_path_factory):
    """A small SMC map fit (not yet regularized); module-scoped since the tests only add regularization."""
    tmp = tmp_path_factory.mktemp("reg_hist_maps")
    cii_out = str(tmp / "cii.fits")
    oi_out = str(tmp / "oi.fits")
    fir_out = str(tmp / "fir.fits")
    Measurement.make_measurement(
        utils.get_testdata("n22_cii_flux.fits"),
        utils.get_testdata("n22_cii_error.fits"),
        outfile=cii_out,
        overwrite=True,
    )
    Measurement.make_measurement(
        utils.get_testdata("n22_oi_flux.fits"), utils.get_testdata("n22_oi_error.fits"), outfile=oi_out, overwrite=True
    )
    Measurement.make_measurement(utils.get_testdata("n22_FIR.fits"), error="10%", outfile=fir_out, overwrite=True)
    measurements = [
        Measurement.read(cii_out, identifier="CII_158"),
        Measurement.read(oi_out, identifier="OI_63"),
        Measurement.read(fir_out, identifier="FIR"),
    ]
    p = LineRatioFit(ModelSet("smc", z=0.2), measurements=measurements)
    p.run()
    return p


def test_raises_if_not_regularized(map_fit):
    with pytest.raises(ValueError):
        LineRatioPlot(map_fit).regularization_histogram()


def test_bad_quantity_raises(map_fit):
    with pytest.raises(ValueError):
        LineRatioPlot(map_fit).regularization_histogram(quantity="bogus")


@pytest.mark.parametrize(("quantity", "npanels"), [("density", 1), ("radiation_field", 1), ("both", 2)])
def test_histogram_runs(map_fit, quantity, npanels):
    map_fit.regularize(method="tv", lam=0.3, max_iter=20)
    plot = LineRatioPlot(map_fit)
    plot.regularization_histogram(quantity=quantity)
    assert len(plot.axis) == npanels
    plot._plt.close(plot.figure)
