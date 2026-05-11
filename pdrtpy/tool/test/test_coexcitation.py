#!/usr/bin/env python3
# coexcitation test code

import warnings

import astropy.units as u
from astropy.nddata import StdDevUncertainty
from pdrtpy.measurement import Measurement
from pdrtpy.tool.excitation import C13OExcitationFit


class TestCOExcitation:
    """test the CO Excitation fitting tools"""

    def test_13co_over_constrained_warning(self):
        """Trim 13CO to 4 points to make components=2 fit_opr=False over-constrained."""
        unit = u.Unit("W m-2 sr-1")
        intensity = [
            Measurement(data=[2.6e-8], uncertainty=StdDevUncertainty(0.85e-8), identifier="13COv0-0J5-4", unit=unit),
            Measurement(data=[2.5e-8], uncertainty=StdDevUncertainty(0.8e-8), identifier="13COv0-0J6-5", unit=unit),
            Measurement(data=[3.7e-8], uncertainty=StdDevUncertainty(1e-8), identifier="13COv0-0J7-6", unit=unit),
            Measurement(data=[3.8e-8], uncertainty=StdDevUncertainty(1.2e-8), identifier="13COv0-0J8-7", unit=unit),
        ]
        h = C13OExcitationFit(intensity)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            h.run(components=2, fit_opr=False)
        assert any("over-constrained" in str(w.message) for w in caught)
