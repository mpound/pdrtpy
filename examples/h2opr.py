############################################################################
###  Listing A.5:  Creating and fitting H2 excitation diagrams,          ###
###                including ortho-to-para ratio (OPR)                   ###
############################################################################
from pdrtpy.measurement import Measurement
from pdrtpy.tool.h2excitation import H2ExcitationFit
from pdrtpy.plot.excitationplot import ExcitationPlot
from astropy.nddata import StdDevUncertainty
intensity = dict()
intensity['H200S0'] = 3.003e-05
intensity['H200S1'] = 3.143e-04
intensity['H200S2'] = 3.706e-04
intensity['H200S3'] = 1.060e-03
intensity['H200S4'] = 5.282e-04
intensity['H200S5'] = 5.795e-04
observations = []
for i in intensity:
    m = Measurement(data=intensity[i],
                    uncertainty=StdDevUncertainty(0.75*intensity[i]),
                    identifier=i,unit="erg cm-2 s-1 sr-1")
    observations.append(m)

# Create the tool to run the fit
hopr = H2ExcitationFit(observations)
# Instantiate the plotter
hplot = ExcitationPlot(hopr,"H_2")

# Set some plot parameters appropriate for manuscript figure; 
# these pass through to matplotlib
hplot._plt.rcParams["xtick.major.size"] = 7
hplot._plt.rcParams["xtick.minor.size"] = 4
hplot._plt.rcParams["ytick.major.size"] = 7
hplot._plt.rcParams["ytick.minor.size"] = 4
hplot._plt.rcParams['font.size'] = 14
hplot._plt.rcParams['axes.linewidth'] =1.5
hplot.ex_diagram(ymax=21)
hplot.savefig('example9_figure.png',dpi=300)

# Fit a two temperature model allowing OPR to vary
hopr.run(fit_opr=True)
hplot.ex_diagram(show_fit=True,ymax=21)
hplot.savefig('example10_figure.png',dpi=300)
