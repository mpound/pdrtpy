#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 15 10:28:10 2025

@author: mpound
"""

# test exciation fitting
from pdrtpy.measurement import Measurement
from pdrtpy.tool.excitation import H2ExcitationFit
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

intensity = dict()
intensity_unc = dict()
intensity['H210Q1'] = 2.6573285650884875e-06
intensity_unc['H210Q1']=StdDevUncertainty(8.50427181247555e-08)
intensity['H210Q2'] = 1.4695275488577075e-06
intensity_unc['H210Q2']=StdDevUncertainty(8.768404267642483e-08)
intensity['H210Q3'] = 1.7223957895745199e-06
intensity_unc['H210Q3']=StdDevUncertainty(8.840028773309878e-08)
intensity['H210Q4'] = 6.113823432536416e-07
intensity_unc['H210Q4']=StdDevUncertainty(8.045947243103051e-08)
intensity['H210Q5'] = 7.23797463431074e-07
intensity_unc['H210Q5']=StdDevUncertainty(7.645048691544922e-08)
intensity['H210Q7'] = 4.378732107085015e-07
intensity_unc['H210Q7']=StdDevUncertainty(8.279870926278172e-08)


a = []
for i in intensity:
    # For this example, set a largish uncertainty on the intensity.
    m = Measurement(data=intensity[i],uncertainty=StdDevUncertainty(intensity_unc[i]),
                    identifier=i,unit="erg cm-2 s-1 sr-1")
    print(m)
    a.append(m)
    
h = H2ExcitationFit(a)
h.column_densities(line=False, norm=False)
h.run(components=1)