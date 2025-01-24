import math
import os
import unittest

import astropy.units as u
import numpy as np
from astropy.nddata import StdDevUncertainty

import pdrtpy.pdrutils as utils
from pdrtpy.measurement import Measurement


class TestMeasurement(unittest.TestCase):
    def setUp(self):
        self.q = []
        self.intensity = dict()
        self.intensity["H200S0"] = 3.003e-05
        self.intensity["H200S1"] = 3.143e-04
        self.intensity["H200S2"] = 3.706e-04
        self.intensity["H200S3"] = 1.060e-03
        # Add a point for J=6 so that the fit is not overconstrained.
        self.intensity["H200S4"] = 5.282e-04
        self.intensity["H200S5"] = 5.795e-04
        for j in self.intensity:
            infile = utils.get_testdata(f"{j:s}_test_data.fits")
            m = Measurement.read(infile, identifier=j)
            self.q.append(m)

    def _check_title_card(self, a, b, op, c):
        return c.header["TITLE"] == f'{a.header["TITLE"]}{op}{b.header["TITLE"]}'

    def test_arithmetic(self):
        print("Measurement Unit Test")
        _data = np.array([np.array([30, 20]), 10, 10, 100], dtype=object)
        _error = np.array([np.array([5, 5]), 2, 1.5, 100], dtype=object)
        _id = ["OI_145", "CI_609", "CO_21", "CII_158"]
        m = list()
        for i in range(len(_data)):
            hdr = {"TITLE": _id[i]}
            x = Measurement(
                data=_data[i], uncertainty=StdDevUncertainty(_error[i]), identifier=_id[i], unit="adu", meta=hdr
            )
            m.append(x)

        for q in range(len(m)):
            a = m[0] / m[q]
            d = _data[0] / _data[q]
            e = d * np.sqrt((_error[0] / _data[0]) ** 2 + (_error[q] / _data[q]) ** 2)
            # print(q,a,d,e)
            self.assertTrue(np.all(a.data == d))
            self.assertTrue(np.all(np.round(a.error, 3) == np.round(e, 3)))
            self.assertTrue(self._check_title_card(m[0], m[q], "/", a))

            a = m[0] * m[q]
            d = _data[0] * _data[q]
            e = d * np.sqrt((_error[0] / _data[0]) ** 2 + (_error[q] / _data[q]) ** 2)
            # print(q,a,d,e)
            self.assertTrue(np.all(a.data == d))
            self.assertTrue(np.all(np.round(a.error, 3) == np.round(e, 3)))
            self.assertTrue(self._check_title_card(m[0], m[q], "*", a))

            a = m[0] + m[q]
            d = _data[0] + _data[q]
            e = np.sqrt(_error[0] ** 2 + _error[q] ** 2)
            # print(q,a,d,e)
            self.assertTrue(np.all(a.data == d))
            self.assertTrue(np.all(np.round(a.error, 3) == np.round(e, 3)))
            self.assertTrue(self._check_title_card(m[0], m[q], "+", a))

            a = m[0] - m[q]
            d = _data[0] - _data[q]
            e = np.sqrt(_error[0] ** 2 + _error[q] ** 2)
            # print(q,a,d,e)
            self.assertTrue(np.all(a.data == d))
            self.assertTrue(np.all(np.round(a.error, 3) == np.round(e, 3)))
            self.assertTrue(self._check_title_card(m[0], m[q], "-", a))

            self.assertTrue(m[q].unit == u.adu)

    # @todo add operations with numerics (e.g. m*3.14)

    def test_interpolation(self):
        shape = self.q[1].data.shape
        x1 = self.q[1]._world_axis_lin[0][shape[0] // 2]
        y1 = self.q[1]._world_axis_lin[1][shape[1] // 2]
        # make sure inperlation on an existing point returns that point
        z1 = np.float64(0.00028160451797954307)
        self.assertTrue(self.q[1]._interp_lin((x1, y1)) == z1)
        x1 = x1 + self.q[1].wcs.wcs.cdelt[0] * np.cos(math.radians(y1))
        y1 = y1 - 3 * self.q[1].wcs.wcs.cdelt[1]
        z2 = np.float64(0.0003121150896071409)
        self.assertTrue(self.q[1]._interp_lin((x1, y1)) == z2)

    def test_read_write(self):
        # Get the input filenames of the FITS files in the testdata directory
        # These are maps from Jameson et al 2018.
        print("Test FITS files are in: %s" % utils.testdata_dir())
        cii_flux = utils.get_testdata("n22_cii_flux.fits")  # [C II] flux
        cii_err = utils.get_testdata("n22_cii_error.fits")  # [C II] error
        oi_flux = utils.get_testdata("n22_oi_flux.fits")  # [O I] flux
        oi_err = utils.get_testdata("n22_oi_error.fits")  # [O I] error
        FIR_flux = utils.get_testdata("n22_FIR.fits")  # FIR flux

        # Output file names
        cii_combined = utils.testdata_dir() + "n22_cii_flux_error.fits"
        oi_combined = utils.testdata_dir() + "n22_oi_flux_error.fits"
        FIR_combined = utils.testdata_dir() + "n22_FIR_flux_error.fits"

        # create the Measurements and write out the FITS files.
        # Set overwrite=True to allow multiple runs of this notebook.
        Measurement.make_measurement(cii_flux, cii_err, cii_combined, overwrite=True)
        Measurement.make_measurement(oi_flux, oi_err, oi_combined, overwrite=True)
        # Assign a 10% error in FIR flux
        Measurement.make_measurement(FIR_flux, error="10%", outfile=FIR_combined, overwrite=True)

        # Read in the FITS files to Measurements
        cii_meas = Measurement.read(cii_combined, identifier="CII_158")
        FIR_meas = Measurement.read(FIR_combined, identifier="FIR")
        oi_meas = Measurement.read(oi_combined, identifier="OI_63")

        self.assertTrue(oi_meas.unit == u.Unit("W / (m2 sr)"))
        self.assertTrue(oi_meas.wcs.naxis == 2)
        self.assertTrue(oi_meas.wcs._naxis == [81, 139])
        self.assertTrue(np.all(oi_meas.wcs.wcs.crval == np.array([12.10878606, -73.33488267])))
        self.assertTrue((np.round(1e7 * np.nanmax(oi_meas.data), 3)) == 2.481)

    def test_2DWCS(self):
        """regression test for issue #90.
        Read a file that has 2 Naxis and 3 WCS coordinate axis
        """
        file = utils.get_testdata("ConvP_S1.fits")
        m = Measurement.read(file, error="10%", unit="MJy/sr")

    def tearDown(self):
        print("cleaning up " + utils.testdata_dir())
        files = ["n22_cii_flux_error.fits", "n22_oi_flux_error.fits", "n22_FIR_flux_error.fits"]
        for f in files:
            try:
                os.remove(utils.testdata_dir() + f)
            except OSError:
                pass


if __name__ == "__main__":
    unittest.main()
