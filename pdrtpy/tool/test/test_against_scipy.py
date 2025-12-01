from pdrtpy.measurement import Measurement
from pdrtpy.tool.h2excitation import H2ExcitationFit
from astropy.nddata import StdDevUncertainty
import numpy as np
from scipy.optimize import curve_fit
import pytest
from astropy.constants import c,h
import astropy.units as u

# Regression test for issue 191 
# https://github.com/mpound/pdrtpy/issues/191
class TestExcitation:
    def func2T(self,x, m1, c1, m2, c2):
        return np.log(np.exp(c1 - m1*x) + np.exp(c2 - m2*x))
    def test_fit(self):
        # pdrtpy fit
        a = []
        Jfluxes_pdrtpy_demo = {0: 3.00e-05, 1: 5.16e-04, 2: 3.71e-04, 3: 1.76e-03, 4: 5.28e-04, 5: 9.73e-04} # values from pdrtpy notebook
        Jerrs_pdrtpy_demo = {0: 1.00e-05, 1: 1.00e-05, 2: 1.00e-05, 3: 1.00e-05, 4: 1.00e-05, 5: 1.00e-05} # choose something that will not have equal fractional error, equal error is simplest
        for J in Jfluxes_pdrtpy_demo.keys():
            m = Measurement(data=Jfluxes_pdrtpy_demo[J],uncertainty=StdDevUncertainty(Jerrs_pdrtpy_demo[J]),
            identifier='H200S{}'.format(J),unit="erg cm-2 s-1 sr-1")
            print("Input for J =",J, m)
            a.append(m)
        he = H2ExcitationFit(a)
        he.run(components=2, method='leastsq')
        print("\n Results of pdrtpy fit:")
        print(f'T(cold) = {he.tcold}')
        print("T(hot) = {:>8.3f}".format(he.thot))

        # manual fit
        clight = c.to("cm/s")
        hplanck = h.to(u.erg*u.s)

        # Roueff et al. (2019): https://ui.adsabs.harvard.edu/abs/2019A%26A...630A..58R/abstract
        wav_H2 = {0: 28.218843793, 1: 17.034845756, 2: 12.278611991, 3: 9.664910918, 4: 8.025041036, 5: 6.909508549, 6: 6.108563840, 7: 5.511183259, 8: 5.053115155, 9: 4.694613923}
        Aul_H2 = {0: 2.943e-11, 1: 4.761e-10, 2: 2.755e-9, 3: 9.836e-9, 4: 2.643e-8, 5: 5.879e-8, 6: 1.142e-7, 7: 2.001e-7, 8: 3.236e-7, 9: 4.90e-07}
        Tu_H2 = {2: 509.9, 3: 1015.1, 4: 1681.6, 5: 2503.7, 6: 3474.5, 7: 4586.1, 8: 5829.8, 9: 7196.7, 10: 8677.1, 11: 10261.4}
        gu_H2 = {2: 5, 3: 21, 4: 9, 5: 33, 6: 13, 7: 45, 8: 17, 9: 57, 10: 21, 11: 69}
        rotDiag_x = np.array([Tu_H2[Jl+2] for Jl in Jfluxes_pdrtpy_demo.keys()])
        rotDiag_y = 4*np.pi * np.array([f for f in Jfluxes_pdrtpy_demo.values()]) / (hplanck*clight*1e4) * (np.array([wav_H2[Jl]/Aul_H2[Jl]/gu_H2[Jl+2] for Jl in Jfluxes_pdrtpy_demo.keys()]))
        rotDiag_e = 4*np.pi * np.array([f for f in Jerrs_pdrtpy_demo.values()]) / (hplanck*clight*1e4) * (np.array([wav_H2[Jl]/Aul_H2[Jl]/gu_H2[Jl+2] for Jl in Jerrs_pdrtpy_demo.keys()]))
        sigma = rotDiag_e/rotDiag_y
        fit2, err2 = curve_fit(self.func2T, rotDiag_x, np.log(rotDiag_y), p0=(1/700,20,1/200,21), bounds=(0,np.inf), max_nfev=8192, absolute_sigma=False, sigma=sigma)
        T1fit = 1/fit2[0]
        T1fit_err = np.sqrt(err2[0,0])/fit2[0]**2
        T2fit = 1/fit2[2]
        T2fit_err = np.sqrt(err2[2,2])/fit2[2]**2
        print("\n Results of own fit:")
        print("T(cold) = {}+-{}".format(T1fit,T1fit_err))
        print("T(hot) = {}+-{}".format(T2fit,T2fit_err))
        
        assert T1fit == pytest.approx(he.tcold.value,rel=1E-3)
        assert T2fit == pytest.approx(he.thot.value,rel=1E-3)
        assert T1fit_err == pytest.approx(he.tcold.error,rel=1E-3)
        assert T2fit_err == pytest.approx(he.thot.error,rel=1E-3)


