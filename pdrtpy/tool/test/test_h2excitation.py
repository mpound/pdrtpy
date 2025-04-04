#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 26 16:01:56 2025

@author: mpound
"""
# h2excitation test code

import pytest
from astropy.nddata import StdDevUncertainty
from dust_extinction.parameter_averages import G23

from pdrtpy.measurement import Measurement
from pdrtpy.tool.excitation import H2ExcitationFit


class TestH2Excitation:
    """test the H2Excitation tool"""

    def setup_method(self):
        self._intensity = {}

    def test_fit_opr(self):
        self._intensity = {
                'H200S0': 3.00e-05,
                'H200S1': 5.16e-04,
                'H200S2': 3.71e-04,
                'H200S3': 1.76e-03,
                'H200S4': 5.28e-04,
                'H200S5': 9.73e-04,
        }
        a = []
        for i in self._intensity:
            # For this example, set a largish uncertainty on the intensity.
            m = Measurement(
                data=self._intensity[i],
                uncertainty=StdDevUncertainty(0.25 * self._intensity[i]),
                identifier=i,
                unit="erg cm-2 s-1 sr-1",
            )
            # print(m)
            a.append(m)
        h = H2ExcitationFit(a)
        #cd = h.column_densities(norm=False)

        # without opr
        h.run(fit_opr=False)
        assert h.thot.data == pytest.approx(692.82, rel=1e-3)
        assert h.tcold.data == pytest.approx(210.233,rel= 1e-3)
        assert h.cold_colden.data == pytest.approx(1.802324E21, rel=1e-3)
        assert h.hot_colden.data == pytest.approx(2.0493E20,rel= 1e-3)
        assert h.opr.data == pytest.approx(3.0, rel=1e-3)
        # with opr
        self._intensity = {
                'H200S0': 3.00e-05,
                'H200S1': 3.143E-4,
                'H200S2': 3.706e-04,
                'H200S3': 1.060e-03,
                'H200S4': 5.282e-04,
                'H200S5': 5.795e-04,
        }
        a = []
        for i in self._intensity:
            m = Measurement(
                data=self._intensity[i],
                uncertainty=StdDevUncertainty(0.25 * self._intensity[i]),
                identifier=i,
                unit="erg cm-2 s-1 sr-1",
            )
            a.append(m)
        h = H2ExcitationFit(a)

        h.run(fit_opr=True)
        assert h.thot.data == pytest.approx(687.467, rel=1e-3)
        assert h.tcold.data == pytest.approx(207.032,rel= 1e-3)
        assert h.cold_colden.data == pytest.approx(1.835378E+21, rel=1e-3)
        assert h.hot_colden.data == pytest.approx(2.07123E+20,rel= 1e-3)
        assert h.opr.data == pytest.approx(1.8615, rel=1e-3)

    def test_fit_av(self):
        # Previous data attenuated by Av=30 using Gordon 2023 extinction curve
        self._intensity = {
            "H200S0": 1.9604610972792677e-05,
            "H200S1": 0.00029530010810198534,
            "H200S2": 0.00021510356715925098,
            "H200S3": 0.0005620620764730837,
            "H200S4": 0.00030386531387721416,
            "H200S5": 0.000576178279762827,
        }
        a = []
        for i in self._intensity:
            # For this example, set a largish uncertainty on the intensity.
            m = Measurement(
                data=self._intensity[i],
                uncertainty=StdDevUncertainty(0.25 * self._intensity[i]),
                identifier=i,
                unit="erg cm-2 s-1 sr-1",
            )
            # print(m)
            a.append(m)
        h = H2ExcitationFit(a)
        g23 = G23(Rv=5.5)
        h.set_extinction_model(g23)
        h.run(fit_av=True)
        assert h.av.data == pytest.approx(28.16196941, 2e-8)
