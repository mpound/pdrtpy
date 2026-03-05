import math
import os

import astropy.units as u
import numpy as np
import pytest
from astropy.nddata import StdDevUncertainty

import pdrtpy.pdrutils as utils
from pdrtpy.measurement import Measurement


class TestMeasurement:
    @pytest.fixture(autouse=True)
    def setup(self):
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
            assert np.all(a.data == d)
            assert np.all(np.round(a.error, 3) == np.round(e, 3))
            assert self._check_title_card(m[0], m[q], "/", a)

            a = m[0] * m[q]
            d = _data[0] * _data[q]
            e = d * np.sqrt((_error[0] / _data[0]) ** 2 + (_error[q] / _data[q]) ** 2)
            # print(q,a,d,e)
            assert np.all(a.data == d)
            assert np.all(np.round(a.error, 3) == np.round(e, 3))
            assert self._check_title_card(m[0], m[q], "*", a)

            a = m[0] + m[q]
            d = _data[0] + _data[q]
            e = np.sqrt(_error[0] ** 2 + _error[q] ** 2)
            # print(q,a,d,e)
            assert np.all(a.data == d)
            assert np.all(np.round(a.error, 3) == np.round(e, 3))
            assert self._check_title_card(m[0], m[q], "+", a)

            a = m[0] - m[q]
            d = _data[0] - _data[q]
            e = np.sqrt(_error[0] ** 2 + _error[q] ** 2)
            # print(q,a,d,e)
            assert np.all(a.data == d)
            assert np.all(np.round(a.error, 3) == np.round(e, 3))
            assert self._check_title_card(m[0], m[q], "-", a)

            assert m[q].unit == u.adu

    # @todo add operations with numerics (e.g. m*3.14)

    def test_interpolation(self):
        shape = self.q[1].data.shape
        x1 = self.q[1]._world_axis_lin[0][shape[0] // 2]
        y1 = self.q[1]._world_axis_lin[1][shape[1] // 2]
        # make sure inperlation on an existing point returns that point
        z1 = np.float64(0.00028160451797954307)
        assert self.q[1]._interp_lin((x1, y1)) == z1
        x1 = x1 + self.q[1].wcs.wcs.cdelt[0] * np.cos(math.radians(y1))
        y1 = y1 - 3 * self.q[1].wcs.wcs.cdelt[1]
        z2 = np.float64(0.0003121150896071409)
        assert self.q[1]._interp_lin((x1, y1)) == z2

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
        assert cii_meas.id == "CII_158"
        FIR_meas = Measurement.read(FIR_combined, identifier="FIR")
        assert FIR_meas.id == "FIR"
        oi_meas = Measurement.read(oi_combined, identifier="OI_63")

        assert oi_meas.unit == u.Unit("W / (m2 sr)")
        assert oi_meas.wcs.naxis == 2
        assert oi_meas.wcs._naxis == [81, 139]
        assert np.all(oi_meas.wcs.wcs.crval == np.array([12.10878606, -73.33488267]))
        assert (np.round(1e7 * np.nanmax(oi_meas.data), 3)) == 2.481

    def test_2DWCS(self):
        """regression test for issue #90.
        Read a file that has 2 Naxis and 3 WCS coordinate axis
        """
        file = utils.get_testdata("ConvP_S1.fits")
        # @todo uncertainty does nothing.
        m = Measurement.read(file, uncertainty=StdDevUncertainty([0.1]), unit="MJy/sr")
        assert m.data.mean() == pytest.approx(4124.39469065)


class TestMeasurementFromTable:
    """Tests for Measurement.from_table"""

    @pytest.fixture(autouse=True)
    def setup(self):
        self.tab_no_beam = utils.get_testdata("cii-co-nc.tab")
        self.tab_abs_err = utils.get_testdata("rcw49_nc_cii158.tab")

    def test_from_table_array_false_percent_error(self):
        """from_table with array=False, % uncertainty"""
        m = Measurement.from_table(self.tab_no_beam, array=False)
        assert isinstance(m, Measurement)
        assert m.data is not None
        assert m.uncertainty is not None
        # identifier from first row
        assert m.id == "CII_158/CO_32"

    def test_from_table_array_true_percent_error(self):
        """from_table with array=True returns list of Measurements"""
        result = Measurement.from_table(self.tab_no_beam, array=True)
        assert isinstance(result, list)
        assert len(result) > 0
        for m in result:
            assert isinstance(m, Measurement)
            assert m.id == "CII_158/CO_32"

    def test_from_table_absolute_error(self):
        """from_table with absolute (non-%) uncertainty"""
        m = Measurement.from_table(self.tab_abs_err, array=False)
        assert isinstance(m, Measurement)
        assert m.uncertainty is not None
        assert m.id == "CII_158"

    def test_from_table_array_true_absolute_error(self):
        result = Measurement.from_table(self.tab_abs_err, array=True)
        assert isinstance(result, list)
        assert len(result) > 0

    def test_from_table_uncertainty_values(self):
        """% errors should be converted to absolute"""
        result = Measurement.from_table(self.tab_no_beam, array=True)
        for m in result:
            # 8.3% of ~300-500 -> ~25-41
            assert m.uncertainty.array[0] > 0

    def test_from_table_missing_required_column_raises(self):
        """Missing required columns should raise an exception"""
        import tempfile

        bad_tab = """| data|    id|
| double|  char|
|      |      |
|  null|  null|
  1.0  CII_158
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".tab", delete=False) as f:
            f.write(bad_tab)
            fname = f.name
        try:
            with pytest.raises(Exception):
                Measurement.from_table(fname, format="ipac")
        finally:
            os.unlink(fname)


class TestMeasurementGetPixel:
    """Tests for Measurement.get_pixel and get"""

    @pytest.fixture(autouse=True)
    def setup(self):
        cii_combined = utils.testdata_dir() + "n22_cii_flux_error.fits"
        self.m = Measurement.read(cii_combined, identifier="CII_158")

    def test_get_pixel_returns_tuple(self):
        crval = self.m.wcs.wcs.crval
        px = self.m.get_pixel(crval[0], crval[1])
        assert isinstance(px, tuple)
        assert len(px) == 2

    def test_get_pixel_roundtrip(self):
        """Convert pixel to world and back"""
        center_x = self.m.wcs._naxis[0] // 2
        center_y = self.m.wcs._naxis[1] // 2
        world = self.m.wcs.pixel_to_world_values(center_x, center_y)
        px = self.m.get_pixel(world[0], world[1])
        assert px[0] == center_x
        assert px[1] == center_y


class TestMeasurementBeamConvert:
    """Tests for _beam_convert"""

    def test_beam_convert_none(self):
        m = Measurement(data=np.array([1.0]), unit="adu", identifier="test")
        # None beam parameters stored as None in header
        assert m.header["BMAJ"] is None

    def test_beam_convert_quantity(self):
        m = Measurement(
            data=np.array([1.0]),
            unit="adu",
            identifier="test",
            bmaj=10 * u.arcsec,
            bmin=8 * u.arcsec,
            bpa=45 * u.deg,
        )
        # stored as degrees
        assert m.header["BMAJ"] == pytest.approx(10 / 3600.0, abs=1e-10)
        assert m.header["BMIN"] == pytest.approx(8 / 3600.0, abs=1e-10)
        assert m.header["BPA"] == pytest.approx(45.0, abs=1e-10)

    def test_beam_convert_non_quantity_raises(self):
        with pytest.raises(TypeError):
            Measurement(
                data=np.array([1.0]),
                unit="adu",
                identifier="test",
                bmaj=10.0,  # not a Quantity
            )
