###########################################################################
###            Listing A.1:  models, ModelSet, and ModelPlot            ###
###########################################################################

from pdrtpy.modelset import ModelSet
from pdrtpy.plot.modelplot import ModelPlot
from pdrtpy.measurement import Measurement
import pdrtpy.pdrutils as utils
import astropy.units as  u
from astropy.nddata import StdDevUncertainty

# Get the Wolfire-Kaufman 2020 Z=1 models 
ms = ModelSet("wk2020",z=1)

# Get KOSMA-tau R=3.1 model
mskt = ModelSet("kt2013wd01-7",z=1,mass=100,medium='clumpy')

# Example of how to fetch a given model, the [OI] 63 micron/[CII] 158 micron intensity ratio.
# The returned model type is pdrtpy.measurement.Measurement.

model = ms.get_model("OI_63/CII_158")
modelkt = mskt.get_model("OI_63/CII_158")

# Find all the models that use some combination of CO(J=1-0), [C II] 158 micron, 
# [O I] 145 micron,  and far-infrared intensity. This example gets both intensity 
# and ratio models, though one can specify model_type='intensity' 
# or model_type='ratio' to get one or the other. 
# The models are returned as a dictionary with the keys set to the model IDs.

mods = ms.get_models(["CII_158","OI_145", "CO_10", "FIR"],model_type='both')
print(list(mods.keys()))
# Output of above: ['OI_145', 'CII_158', 'CO_10', 'CII_158/OI_145', 'CII_158/CO_10', 
#                   'CII_158/FIR', 'OI_145+CII_158/FIR']

# Plot a selected model and save it to a PDF file.  Note in this example,
# we request Habing units for the FUV field.

# WK
mp = ModelPlot(ms)
mp.plot('OI_145+CII_158/FIR',yaxis_unit='Habing',
        label=True, cmap='viridis', colors='k',norm='log')
mp.savefig("example1a_figure.pdf")
# KT
mpkt = ModelPlot(mskt)
mpkt.plot('OI_145+CII_158/FIR',yaxis_unit='Habing',
        label=True, cmap='viridis', colors='k',norm='log')
mpkt.savefig("example1b_figure.pdf")

rcw49 = []
label = ["shell","pillar","northern cloud","ridge"]
format_ = ["k+","b+","g+","r+"]
# The data files are in the testdata directory of the pdrtpy installation
for region in ["shell","pil","nc","ridge"]:
    f1 = utils.get_testdata(f"cii-fir-{region}.tab")
    f2 = utils.get_testdata(f"cii-co-{region}.tab")
    rcw49.append(Measurement.from_table(f1))
    rcw49.append(Measurement.from_table(f2))
    
mp.phasespace(['CII_158/FIR','CII_158/CO_32'],nax1_clip=[1E2,1E5]*u.Unit("cm-3"),
               nax2_clip=[1E1,1E6]*utils.habing_unit, measurements=rcw49,label=label,
               fmt=format_,title="RCW 49 Regions")
mp.savefig("example1c_figure.pdf")

# Example ionized gas line diagnostic diagram
i1 = Measurement(identifier='FEII_1.60/FEII_1.64',
                 data=[0.1,0.05,0.2],
                 uncertainty=StdDevUncertainty([0.025,0.005,0.05]),unit="")
i2 = Measurement(identifier='FEII_1.64/FEII_5.34',
                 data=[0.3,0.1,1.0],
                 uncertainty=StdDevUncertainty([0.1,0.05,0.25]),unit="")
mp.phasespace(['FEII_1.60/FEII_1.64','FEII_1.64/FEII_5.34'],
              nax2_clip=[10,1E6]*u.Unit("cm-3"),nax1_clip=[1E3,8E3]*u.Unit("K"),
             measurements=[i1,i2],errorbar=True)
mp.savefig("example1d_figure.pdf")