import unittest
from pdrtpy.measurement import Measurement
from astropy.nddata import StdDevUncertainty

class TestMeasurement(unittest.TestCase):
    def test_a(self):
        m1 = Measurement(data=[30.,20.],uncertainty = StdDevUncertainty([5.,5.]),identifier="OI_145",unit="adu")
        m2 = Measurement(data=10.,uncertainty = StdDevUncertainty(2.),identifier="CI_609",unit="adu")
        m3 = Measurement(data=10.,uncertainty = StdDevUncertainty(1.5),identifier="CO_21",unit="adu")
        m4 = Measurement(data=100.,uncertainty = StdDevUncertainty(10.),identifier="CII_158",unit="adu")
        print(m1/m2)
        print(m2/m3)
        print(m1*m2)
        print(m2/m4)
        print(m4*m3)
        print(m4+m3)
        print(m3-m1)
#
#        print(m3.levels)

        self.assertTrue(True)

if __name__ == '__main__':
    unittest.main()

