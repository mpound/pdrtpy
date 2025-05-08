#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# coexcitation test code

import pytest
import astropy.units as u
from astropy.nddata import StdDevUncertainty


from pdrtpy.measurement import Measurement
from pdrtpy.tool.excitation import COExcitationFit, C13OExcitationFit


class TestCOExcitation:
    """test the CO Excitation fitting tools"""

    def test_co_fit(self):
        # joblin et al NGC7023 data
        # 12CO
        intensity = {}
        unit = u.Unit("W m-2 sr-1")
        intensity["COv0-0J4-3"] = Measurement(
            data=[2.8e-8], uncertainty=StdDevUncertainty(0.9e-8), identifier="COv0-0J4-3", unit=unit
        )
        intensity["COv0-0J5-4"] = Measurement(
            data=[5.3e-8], uncertainty=StdDevUncertainty(1.6e-8), identifier="COv0-0J5-4", unit=unit
        )
        intensity["COv0-0J6-5"] = Measurement(
            data=[1.1e-7], uncertainty=StdDevUncertainty(((0.5 + 0.2) / 2) * 1e-7), identifier="COv0-0J6-5", unit=unit
        )
        intensity["COv0-0J7-6"] = Measurement(
            data=[2.0e-7], uncertainty=StdDevUncertainty(0.6e-7), identifier="COv0-0J7-6", unit=unit
        )
        intensity["COv0-0J8-7"] = Measurement(
            data=[2.0e-7], uncertainty=StdDevUncertainty(((0.5 + 0.8) / 2) * 1e-7), identifier="COv0-0J8-7", unit=unit
        )
        intensity["COv0-0J9-8"] = Measurement(
            data=[3.1e-7], uncertainty=StdDevUncertainty(0.7e-7), identifier="COv0-0J9-8", unit=unit
        )
        intensity["COv0-0J10-9"] = Measurement(
            data=[2.5e-7], uncertainty=StdDevUncertainty(0.8e-7), identifier="COv0-0J10-9", unit=unit
        )
        intensity["COv0-0J11-10"] = Measurement(
            data=[3.5e-7], uncertainty=StdDevUncertainty(1.0e-7), identifier="COv0-0J11-10", unit=unit
        )
        intensity["COv0-0J12-11"] = Measurement(
            data=[2.7e-7], uncertainty=StdDevUncertainty(0.8e-7), identifier="COv0-0J12-11", unit=unit
        )
        intensity["COv0-0J13-12"] = Measurement(
            data=[2.4e-7], uncertainty=StdDevUncertainty(((0.9 + 0.5) / 2) * 1e-7), identifier="COv0-0J13-12", unit=unit
        )
        intensity["COv0-0J15-14"] = Measurement(
            data=[1.2e-7], uncertainty=StdDevUncertainty(0.2e-7), identifier="COv0-0J15-14", unit=unit
        )
        intensity["COv0-0J16-15"] = Measurement(
            data=[6.5e-8], uncertainty=StdDevUncertainty(1.3e-8), identifier="COv0-0J16-15", unit=unit
        )
        intensity["COv0-0J17-16"] = Measurement(
            data=[3.3e-8], uncertainty=StdDevUncertainty(0.8e-8), identifier="COv0-0J17-16", unit=unit
        )
        intensity["COv0-0J18-17"] = Measurement(
            data=[2.0e-8], uncertainty=StdDevUncertainty(0.5e-8), identifier="COv0-0J18-17", unit=unit
        )
        intensity["COv0-0J19-18"] = Measurement(
            data=[1.2e-8], uncertainty=StdDevUncertainty(0.7e-8), identifier="COv0-0J19-18", unit=unit
        )

        h = COExcitationFit(list(intensity.values()))

        h.run(components=1)
        assert h.thot.data == pytest.approx(116.4179, rel=1e-3)
        assert h.tcold == h.thot
        assert h.cold_colden.data == pytest.approx(1.423889e17, rel=1e-3)
        assert h.hot_colden == h.cold_colden

    def test_13co_fit(self):
        # joblin et al NGC7023 data
        # 13CO
        intensity = {}
        unit = u.Unit("W m-2 sr-1")

        intensity["13COv0-0J5-4"] = Measurement(
            data=[2.6e-8], uncertainty=StdDevUncertainty(((1 + 0.7) / 2) * 1e-8), identifier="13COv0-0J5-4", unit=unit
        )
        intensity["13COv0-0J6-5"] = Measurement(
            data=[2.5e-8], uncertainty=StdDevUncertainty(0.8e-8), identifier="13COv0-0J6-5", unit=unit
        )
        intensity["13COv0-0J7-6"] = Measurement(
            data=[3.7e-8], uncertainty=StdDevUncertainty(1e-8), identifier="13COv0-0J7-6", unit=unit
        )
        intensity["13COv0-0J8-7"] = Measurement(
            data=[3.8e-8], uncertainty=StdDevUncertainty(1.2e-8), identifier="13COv0-0J8-7", unit=unit
        )
        intensity["13COv0-0J9-8"] = Measurement(
            data=[5.8e-8], uncertainty=StdDevUncertainty(1.7e-8), identifier="13COv0-0J9-8", unit=unit
        )
        intensity["13COv0-0J10-9"] = Measurement(
            data=[4.0e-8], uncertainty=StdDevUncertainty(((1 + 1.7) / 2) * 1e-8), identifier="13COv0-0J10-9", unit=unit
        )

        h = C13OExcitationFit(list(intensity.values()))

        h.run(components=1)
        assert h.thot.data == pytest.approx(80.1933388, rel=1e-3)
        assert h.tcold == h.thot
        assert h.cold_colden.data == pytest.approx(8.75495e16, rel=1e-3)
        assert h.hot_colden == h.cold_colden
