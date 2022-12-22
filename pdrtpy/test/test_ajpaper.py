import unittest
import os

from pdrtpy.modelset import ModelSet
from pdrtpy.plot.modelplot import ModelPlot
from pdrtpy.measurement import Measurement
from pdrtpy.tool.lineratiofit import LineRatioFit
from pdrtpy.plot.lineratioplot import LineRatioPlot
from pdrtpy.tool.h2excitation import H2ExcitationFit
from pdrtpy.plot.excitationplot import ExcitationPlot
import pdrtpy.pdrutils as utils
import astropy.units as  u
from astropy.nddata import StdDevUncertainty
from lmfit import fit_report
from copy import deepcopy
import corner
import numpy as np


class TestAJPaperListings(unittest.TestCase):
    def setUp(self):
        self._files = []

    def test_listingA1(self):
        print("### LISTING A.1")
        ###########################################################################
        ###            Listing A.1:  models, ModelSet, and ModelPlot            ###
        ###########################################################################

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
        self._files.append("example1a_figure.pdf")
        # KT
        mpkt = ModelPlot(mskt)
        mpkt.plot('OI_145+CII_158/FIR',yaxis_unit='Habing',
                label=True, cmap='viridis', colors='k',norm='log')
        mpkt.savefig("example1b_figure.pdf")
        self._files.append("example1b_figure.pdf")

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
        self._files.append("example1c_figure.pdf")


    def test_listingA2(self):
        print("### LISTING A.2")
        #############################################################################
        ###  Listing A.2: Fitting intensity ratios for single-pixel observations  ###
        #############################################################################

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
        plot.ratios_on_models(yaxis_unit="Draine",colorbar=True,norm='log',cmap='cividis',label=True,ncols=3,figsize=(23,7))
        # This is Figure 2 in the paper.
        plot.savefig('example2_figure.pdf')
        self._files.append('example2_figure.pdf')

        plot.overlay_all_ratios(yaxis_unit="Draine",figsize=(6,7))
        plot.savefig("example3_figure.pdf")
        self._files.append('example3_figure.pdf')
        # Plot the reduced chisquare, with only contours and legend
        plot.chisq(image=False,colors='k',label=True,legend=True,yaxis_unit='Draine',figsize=(6,7))
        plot.savefig("example4_figure.pdf")
        self._files.append('example4_figure.pdf')


# In[1]:

    def test_listingA3(self):
        print("### LISTING A.3")
        #######################################################################################
        ###  Listing A.3:  Fitting intensity ratios for single-pixel observations with MCMC ###
        #######################################################################################


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
        self._files.append("example5_figure.pdf")

    def test_listingA4(self):
        print("### LISTING A.4")

        ############################################################################
        ###  Listing A.4:  Fitting intensity ratios for map observations         ###
        ############################################################################

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
        self._files.append(cii_combined)
        self._files.append(oi_combined)
        self._files.append(FIR_combined)
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
        self._files.append('example6_g0_figure.pdf')
        self._files.append('example6_n_figure.pdf')

        # Save the results to FITS files.
        p.density.write("N22_density_map.fits",overwrite=True)
        p.radiation_field.write("N22_G0_map.fits",overwrite=True)
        self._files.append('N22_density_map.fits')
        self._files.append('N22_G0_map.fits')



    def test_listingA5(self):
        print("### LISTING A.5")

        ############################################################################
        ###  Listing A.5:  Creating and fitting H2 excitation diagrams,          ###
        ###                including ortho-to-para ratio (OPR)                   ###
        ############################################################################

        intensity = dict()
        intensity['H200S0'] = 3.003e-05
        intensity['H200S1'] = 3.143e-04
        intensity['H200S2'] = 3.706e-04
        intensity['H200S3'] = 1.060e-03
        # Add a point for J=6 so that the fit is not overconstrained.
        intensity['H200S4'] = 5.282e-04
        intensity['H200S5'] = 5.795e-04
        observations = []
        for i in intensity:
            m = Measurement(data=intensity[i],
                            uncertainty=StdDevUncertainty(0.75*intensity[i]),
                            identifier=i,unit="erg cm-2 s-1 sr-1")
            observations.append(m)

        hopr = H2ExcitationFit(observations)
        hopr.run(fit_opr=True)
        hplot = ExcitationPlot(hopr,"H_2")
        hplot._plt.rcParams["xtick.major.size"] = 7
        hplot._plt.rcParams["xtick.minor.size"] = 4
        hplot._plt.rcParams["ytick.major.size"] = 7
        hplot._plt.rcParams["ytick.minor.size"] = 4
        hplot._plt.rcParams['font.size'] = 14
        hplot._plt.rcParams['axes.linewidth'] =1.5
        hopr.run(fit_opr=True)
        hplot.ex_diagram(show_fit=True,ymax=21)
        hplot.savefig('example7_figure.pdf')
        self._files.append('example7_figure.pdf')



    def tearDown(self):
        print('cleaning up')
        for f in self._files:
            try:
                os.remove(f)
            except OSError:
                pass

if __name__ == '__main__':
    unittest.main()
