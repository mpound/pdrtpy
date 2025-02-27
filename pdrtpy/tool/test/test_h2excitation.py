#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 26 16:01:56 2025

@author: mpound
"""
# h2excitation test code

from pdrtpy.measurement import Measurement
from pdrtpy.tool.h2excitation import H2ExcitationFit
from astropy.nddata import StdDevUncertainty
from dust_extinction.parameter_averages import G23
import pytest

class TestH2Excitation:
    """ test the H2Excitation tool"""

    def setup_method(self):
        self._intensity = {}
 
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
                data=self._intensity[i], uncertainty=StdDevUncertainty(0.5 * self._intensity[i]), identifier=i, unit="erg cm-2 s-1 sr-1"
            )
            # print(m)
            a.append(m)
        h = H2ExcitationFit(a)
        g23 = G23(Rv=5.5)
        h.set_extinction_model(g23)
        h.run(fit_av=True)
        assert h.av.data == pytest.approx(28.16196941,2e-8)
