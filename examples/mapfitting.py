############################################################################
###  Listing A.4:  Fitting intensity ratios for map observations         ###
############################################################################
from pdrtpy.measurement import Measurement
from pdrtpy.modelset import ModelSet
from pdrtpy.tool.lineratiofit import LineRatioFit
from pdrtpy.plot.lineratioplot import LineRatioPlot
import pdrtpy.pdrutils as utils

# Get the input filenames of the FITS files in the testdata directory
# utils.get_testdata() is a special method to locate files there.
# These are maps from Jameson et al 2018.
print("Test FITS files are in: %s"%utils.testdata_dir())
cii_flux = utils.get_testdata("n22_cii_flux.fits")  # [C II] flux
cii_err = utils.get_testdata("n22_cii_error.fits")  # [C II] error
oi_flux = utils.get_testdata("n22_oi_flux.fits")    # [O I] flux 
oi_err = utils.get_testdata("n22_oi_error.fits")    # [O I] error
FIR_flux = utils.get_testdata("n22_FIR.fits")       # FIR flux

# Output file names
cii_combined = "n22_cii_flux_error.fits"
oi_combined = "n22_oi_flux_error.fits"
FIR_combined = "n22_FIR_flux_error.fits"

# create the Measurements and write them out as FITS files with two HDUs.
Measurement.make_measurement(cii_flux, cii_err, 
                             outfile=cii_combined, overwrite=True)
Measurement.make_measurement(oi_flux, oi_err, 
                             outfile=oi_combined, overwrite=True)
# Assign a 10% error in FIR flux
Measurement.make_measurement(FIR_flux, error='10%',
                             outfile=FIR_combined, overwrite=True)

# Read back in the FITS files to Measurements
cii_meas = Measurement.read(cii_combined, identifier="CII_158")
FIR_meas = Measurement.read(FIR_combined, identifier="FIR")
oi_meas  = Measurement.read(oi_combined, identifier="OI_63")

# Here we will use the Small Magellanic Cloud ModelSet that have Z=0.1
# These are a limited set of models with just a few lines covered.
smc_ms = ModelSet("smc",z=0.1)
p = LineRatioFit(modelset=smc_ms, measurements=[cii_meas,FIR_meas,oi_meas])
p.run()
plot = LineRatioPlot(p)
plot.density(contours=True,norm="log",cmap='cividis')
plot.savefig("example6_n_figure.pdf")
plot.radiation_field(units="Habing",contours=True,norm="simple",cmap='cividis')
plot.savefig('example6_g0_figure.pdf')

# Save the results to FITS files.
p.density.write("N22_density_map.fits",overwrite=True)
p.radiation_field.write("N22_G0_map.fits",overwrite=True)