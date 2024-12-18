# test modelset.ModelSet
import unittest

from pdrtpy.plot.modelplot import ModelPlot
from pdrtpy.modelset import ModelSet
from pdrtpy.measurement import Measurement
from astropy.nddata import StdDevUncertainty


class TestModelSet(unittest.TestCase):
    def test_multimeasurement_overlay(self):
        # this is a regression test for issue 94.
        # it should complete with no exception
        try:
            ms = ModelSet("wk2020", z=1)
            mp = ModelPlot(ms)
            m1 = Measurement(1.0, uncertainty=StdDevUncertainty(0.002), unit="erg s-1 cm-2", identifier="OI_145")
            m2 = Measurement(1 / 7.2e-2, uncertainty=StdDevUncertainty(1), unit="erg s-1 cm-2", identifier="OI_63")
            m3 = Measurement(1 / 3.0e-2, uncertainty=StdDevUncertainty(2), unit="erg s-1 cm-2", identifier="OI_63")
            mp.plot(
                "OI_145/OI_63",
                yaxis_unit="Habing",
                meas_color=["red", "orange"],
                measurements=[m1 / m2, m1 / m3],
                shading=False,
                cmap="viridis",
            )
            assert True
        except Exception:
            assert False


if __name__ == "__main__":
    unittest.main()
