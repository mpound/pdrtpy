#!/usr/bin/env python3
"""
Created on Thu Oct 23 13:04:46 2025

@author: mpound
"""

# test of piece wise regression on h2 fitting
from pdrtpy.measurement import Measurement
from pdrtpy.tool.h2excitation import H2ExcitationFit
from pdrtpy.plot.excitationplot import ExcitationPlot
import pdrtpy.pdrutils as utils
from astropy.nddata import StdDevUncertainty
import astropy.units as u
from astropy.table import Table
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from dust_extinction.parameter_averages import G23
from astropy.units.quantity import Quantity
import piecewise_regression
import numpy as np
intensity = dict()
intensity['H200S0'] = 3.00e-05
intensity['H200S1'] = 5.16e-04
intensity['H200S2'] = 3.71e-04
intensity['H200S3'] = 1.76e-03
intensity['H200S4'] = 5.28e-04
intensity['H200S5'] = 9.73e-04


a = []
for i in intensity:
    # For this example, set a largish uncertainty on the intensity.
    m = Measurement(data=intensity[i],uncertainty=StdDevUncertainty(intensity[i]),
                    identifier=i,unit="erg cm-2 s-1 sr-1")
    print(m)
    a.append(m)
h = H2ExcitationFit(a)    
cd=h.column_densities(line=False, norm=True)
energies=h.energies(line=False)

cdl=np.squeeze(np.array([v.data for v in cd.values()]))
logcdl=np.log10(cdl)
el=np.squeeze(np.array(list(energies.values())))
logcdl

pw_fit = piecewise_regression.Fit(el, logcdl, n_breakpoints=1)
pw_fit.summary()

pw_fit.plot_data(color="grey", s=20)
pw_fit.plot_fit(color="red", linewidth=4)
pw_fit.plot_breakpoints()
pw_fit.plot_breakpoint_confidence_intervals()


