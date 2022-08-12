import unittest
from pdrtpy.measurement import Measurement
import pdrtpy.pdrutils as utils
from astropy.nddata import StdDevUncertainty
import astropy.units as u
import numpy as np
import os

class TestMeasurement(unittest.TestCase):
    def test_arithmetic(self):
        print("Measurement Unit Test")
        _data = np.array([np.array([30,20]),10,10,100])
        _error = np.array([np.array([5,5]),2,1.5,100])
        _id = ["OI_145","CI_609","CO_21","CII_158"]
        m = list()
        for i in range(len(_data)):
            x = Measurement(data=_data[i],
                            uncertainty = StdDevUncertainty(_error[i]),
                            identifier = _id[i], unit = "adu")
            m.append(x)

        for q in range(len(m)):
            a = m[0]/m[q]
            d = _data[0]/_data[q]
            e = d*np.sqrt( (_error[0]/_data[0])**2+ (_error[q]/_data[q])**2)
            #print(q,a,d,e)
            self.assertTrue(np.all(a.data == d))
            self.assertTrue(np.all(np.round(a.error,3) == np.round(e,3)))

            a = m[0]*m[q]
            d = _data[0]*_data[q]
            e = d*np.sqrt( (_error[0]/_data[0])**2+ (_error[q]/_data[q])**2)
            #print(q,a,d,e)
            self.assertTrue(np.all(a.data == d))
            self.assertTrue(np.all(np.round(a.error,3) == np.round(e,3)))

            a = m[0]+m[q]
            d = _data[0]+_data[q]
            e = np.sqrt(_error[0]**2+_error[q]**2)
            #print(q,a,d,e)
            self.assertTrue(np.all(a.data == d))
            self.assertTrue(np.all(np.round(a.error,3) == np.round(e,3)))

            a = m[0]-m[q]
            d = _data[0]-_data[q]
            e = np.sqrt(_error[0]**2+_error[q]**2)
            #print(q,a,d,e)
            self.assertTrue(np.all(a.data == d))
            self.assertTrue(np.all(np.round(a.error,3) == np.round(e,3)))

            self.assertTrue(m[q].unit == u.adu)
    #@todo add operations with numerics (e.g. m*3.14)

    def test_read_write(self):
        # Get the input filenames of the FITS files in the testdata directory
        # These are maps from Jameson et al 2018.
        print("Test FITS files are in: %s"%utils.testdata_dir())
        cii_flux = utils.get_testdata("n22_cii_flux.fits")  # [C II] flux
        cii_err = utils.get_testdata("n22_cii_error.fits")  # [C II] error
        oi_flux = utils.get_testdata("n22_oi_flux.fits")    # [O I] flux
        oi_err = utils.get_testdata("n22_oi_error.fits")    # [O I] error
        FIR_flux = utils.get_testdata("n22_FIR.fits")       # FIR flux

        # Output file names
        cii_combined = utils.testdata_dir()+"n22_cii_flux_error.fits"
        oi_combined = utils.testdata_dir()+"n22_oi_flux_error.fits"
        FIR_combined = utils.testdata_dir()+"n22_FIR_flux_error.fits"

        # create the Measurements and write out the FITS files.
        # Set overwrite=True to allow multiple runs of this notebook.
        Measurement.make_measurement(cii_flux, cii_err, cii_combined,overwrite=True)
        Measurement.make_measurement(oi_flux, oi_err, oi_combined,overwrite=True)
        # Assign a 10% error in FIR flux
        Measurement.make_measurement(FIR_flux, error='10%', outfile=FIR_combined,overwrite=True)

        # Read in the FITS files to Measurements
        cii_meas = Measurement.read(cii_combined, identifier="CII_158")
        FIR_meas = Measurement.read(FIR_combined, identifier="FIR")
        oi_meas = Measurement.read(oi_combined, identifier="OI_63")

        self.assertTrue(oi_meas.unit == u.Unit("W / (m2 sr)"))
        self.assertTrue(oi_meas.wcs.naxis == 2)
        self.assertTrue(oi_meas.wcs._naxis == [81, 139])
        self.assertTrue(np.all(oi_meas.wcs.wcs.crval== np.array([ 12.10878606, -73.33488267])))
        self.assertTrue((np.round(1E7*np.nanmax(oi_meas.data),3)) == 2.481)

    def tearDown(self):
        print('cleaning up '+utils.testdata_dir())
        files = ["n22_cii_flux_error.fits",
                 "n22_oi_flux_error.fits",
                 "n22_FIR_flux_error.fits"
                ]
        for f in files:
            try:
                os.remove(utils.testdata_dir()+f)
            except OSError:
                pass

if __name__ == '__main__':
    unittest.main()
