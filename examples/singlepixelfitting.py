#############################################################################
###  Listing A.2: Fitting intensity ratios for single-pixel observations  ###
#############################################################################
from pdrtpy.measurement import Measurement
from pdrtpy.modelset import ModelSet
from pdrtpy.tool.lineratiofit import LineRatioFit
from pdrtpy.plot.lineratioplot import LineRatioPlot
import pdrtpy.pdrutils as utils
from astropy.nddata import StdDevUncertainty
from lmfit import fit_report

# Example using single-beam observations of [OI] 163 micron, [CI] 609 micron, CO(J=4-3), 
# and [CII] 158 micron lines. You create `Measurements` for these using the constructor
# which takes the value, error, line identifier string, and units.   The value and the error 
# must be in the same units.  You can mix units  in different Measurements; note we use 
# K km/s  for the CO observation below.   The PDR Toolbox will convert all `Measurements` 
# to a common unit before using them in a fit. You can also add optional beam size 
# (bmaj,bmin,bpa), however the tools requires all `Measurements` have the same beam size 
# before calculations can be performed.  (If you don't provide beam parameters for any of 
# your Measurements,  the Toolbox will assume they are all the same).

myunit = "erg s-1 cm-2 sr-1" # default unit for value and error
m1 = Measurement(data=3.6E-4,uncertainty = StdDevUncertainty(1.2E-4),
                 identifier="OI_63",unit=myunit)
m2 = Measurement(data=1E-6,uncertainty = StdDevUncertainty([3E-7]),
                 identifier="CI_609",unit=myunit)
m3 = Measurement(data=26,uncertainty = StdDevUncertainty([5]),
                 identifier="CO_43",restfreq="461.04077 GHz", unit="K km/s")
m4 = Measurement(data=8E-5,uncertainty = StdDevUncertainty([8E-6]),
                 identifier="CII_158",unit=myunit)
observations = [m1,m2,m3,m4]

ms = ModelSet("wk2020",z=1)

# Instantiate the LineRatioFit tool giving it the ModelSet and Measurements
p = LineRatioFit(ms,measurements=observations) 
p.run()
# Print the fitted quantities using Python f-strings and the fit report via lmfit
print(f"n={p.density:.2e}\nX={utils.to('Draine',p.radiation_field):.2e}")
print(fit_report(p.fit_result[0]))

# Create the plotting tool for the LineRatioPlot, 
# then make plots of the observed ratios overlayed on the model ratios
plot = LineRatioPlot(p)
plot.ratios_on_models(yaxis_unit="Draine",colorbar=True,norm='log',
                      cmap='cividis',label=True,ncols=3,figsize=(23,7))
plot.savefig('example2_figure.pdf')

plot.overlay_all_ratios(yaxis_unit="Draine",figsize=(6,7))
plot.savefig("example3_figure.pdf")
# Plot the reduced chisquare, with only contours and legend
plot.chisq(image=False,colors='k',label=True,legend=True,yaxis_unit='Draine',figsize=(6,7))
plot.savefig("example4_figure.pdf")