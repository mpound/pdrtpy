#!/usr/bin/env python3
"""
Created on Thu Oct 23 13:04:46 2025

@author: mpound
"""

import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np
import piecewise_regression
from astropy.nddata import StdDevUncertainty
from astropy.table import Table
from astropy.units.quantity import Quantity
from dust_extinction.parameter_averages import G23
from matplotlib.patches import Rectangle

import pdrtpy.pdrutils as utils

# test of piece wise regression on h2 fitting
from pdrtpy.measurement import Measurement
from pdrtpy.plot.excitationplot import ExcitationPlot
from pdrtpy.tool.h2excitation import H2ExcitationFit

intensity = dict()
intensity["H200S0"] = 3.00e-05
intensity["H200S1"] = 5.16e-04
intensity["H200S2"] = 3.71e-04
intensity["H200S3"] = 1.76e-03
intensity["H200S4"] = 5.28e-04
intensity["H200S5"] = 9.73e-04


a = []
for i in intensity:
    # For this example, set a largish uncertainty on the intensity.
    m = Measurement(
        data=intensity[i], uncertainty=StdDevUncertainty(intensity[i]), identifier=i, unit="erg cm-2 s-1 sr-1"
    )
    print(m)
    a.append(m)
h = H2ExcitationFit(a)
cd = h.column_densities(line=False, norm=True)
energies = h.energies(line=False)

cdl = np.squeeze(np.array([v.data for v in cd.values()]))
logcdl = np.log10(cdl)
el = np.squeeze(np.array(list(energies.values())))
logcdl

pw_fit = piecewise_regression.Fit(el, logcdl, n_breakpoints=1)
pw_fit.summary()

pw_fit.plot_data(color="grey", s=20)
pw_fit.plot_fit(color="red", linewidth=4)
pw_fit.plot_breakpoints()
pw_fit.plot_breakpoint_confidence_intervals()
