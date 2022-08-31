#######################################################################################
###  Listing A.3:  Fitting intensity ratios for single-pixel observations with MCMC ###
#######################################################################################

from pdrtpy.measurement import Measurement
from pdrtpy.modelset import ModelSet
from pdrtpy.tool.lineratiofit import LineRatioFit
from pdrtpy.plot.lineratioplot import LineRatioPlot
import pdrtpy.pdrutils as utils
from astropy.nddata import StdDevUncertainty
from copy import deepcopy
import corner
import numpy as np

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
p.run(method='emcee',steps=2000)

res = p.fit_result[0]

# the value of the Draine unit in cgs
scale = utils.draine_unit.cgs.scale
# copy the results table
rescopy = deepcopy(res.flatchain)

# scale the radiation_field column of the table to Draine since it is in cgs
rescopy['radiation_field'] /= scale # = np.log10(rescopy['radiation_field']/scale)
#rescopy['density'] = np.log10(rescopy['density'])
# now copy and scale the "best fit" values where the cross hairs are plotted.
truths=np.array(list(res.params.valuesdict().values()))
truths[1] /=scale
#truths = np.log10(truths)

fig = corner.corner(rescopy, bins=20,range=[(1E4,1.2E5),(10,500)],
                    labels=[r"$n~{\rm [cm^{-3}]}$",r"$\chi~{\rm [Draine]}$"],
                    truths=truths)
fig.savefig("example5_figure.pdf",facecolor='white',transparent=False)
