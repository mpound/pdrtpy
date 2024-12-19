import unittest
from pdrtpy.measurement import Measurement
from pdrtpy.modelset import ModelSet
import pdrtpy.pdrutils as utils
from pdrtpy.tool.lineratiofit import LineRatioFit
from pdrtpy.plot.lineratioplot import LineRatioPlot


class TestPlotBase(unittest.TestCase):
    def test_plt_axis_plot(self):
        """This is a regression test for issue #89."""
        # basic map fitting example from notebooks. Jameson et al.
        cii_flux = utils.get_testdata("n22_cii_flux.fits")  # [C II] flux
        cii_err = utils.get_testdata("n22_cii_error.fits")  # [C II] error
        oi_flux = utils.get_testdata("n22_oi_flux.fits")  # [O I] flux
        oi_err = utils.get_testdata("n22_oi_error.fits")  # [O I] error
        FIR_flux = utils.get_testdata("n22_FIR.fits")  # FIR flux

        # Output file names
        cii_combined = "n22_cii_flux_error.fits"
        oi_combined = "n22_oi_flux_error.fits"
        FIR_combined = "n22_FIR_flux_error.fits"

        # create the Measurements and write them out as FITS files with two HDUs.
        # Set overwrite=True to allow multiple runs of this notebook.
        Measurement.make_measurement(cii_flux, cii_err, cii_combined, overwrite=True)
        Measurement.make_measurement(oi_flux, oi_err, oi_combined, overwrite=True)
        # Assign a 10% error in FIR flux
        Measurement.make_measurement(FIR_flux, error="10%", outfile=FIR_combined, overwrite=True)

        # Read in the FITS files to Measurements
        cii_meas = Measurement.read(cii_combined, identifier="CII_158")
        FIR_meas = Measurement.read(FIR_combined, identifier="FIR")
        oi_meas = Measurement.read(oi_combined, identifier="OI_63")
        smcmod = ModelSet("smc", z=0.1)
        p = LineRatioFit(modelset=smcmod, measurements=[cii_meas, FIR_meas, oi_meas])
        p.run()
        plot = LineRatioPlot(p)
        plot.density(contours=True, norm="log", colorbar=True)

        xmin, xmax, ymin, ymax = plot._plt.axis()
        # print(f"{xmin=} {xmax=} {ymin=} {ymax=}")
        # plot._plt.scatter(
        #     [(xmax - xmin) / 2.0],
        #     [(ymax - ymin) / 2.0],
        #     color="green",
        #     s=10,
        # )
        assert xmin == -0.5
        assert xmax == 80.5
        assert ymin == -0.5
        assert ymax == 138.5
