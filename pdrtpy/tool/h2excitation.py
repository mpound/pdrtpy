from astropy.nddata import Cutout2D
import astropy.units as u
import astropy.constants as constants
from astropy.nddata import StdDevUncertainty
import math
import numpy as np
from scipy.optimize import curve_fit
#from statsmodels.stats.weightstats import DescrStatsW

from .toolbase import ToolBase
from .. import pdrutils as utils
from ..measurement import Measurement

class ExcitationFit(ToolBase):
    """Base class for creating excitation fitting tools for various species.

    :param measurements: Input measurements to be fit.  
    :type measurements: array or dict `~pdrtpy.measurement.Measurement`. If dict, the keys should be the Measurement *identifiers*.  
    """
    def __init__(self,measurements=None,constantsfile=None):
        # must be set before call to init_measurements
        self._intensity_units = "erg cm^-2 s^-1 sr^-1"
        self._cd_units = 'cm^-2'
        #print("ExcitationFIt contstructor")
        if type(measurements) == dict or measurements is None:
            self._measurements = measurements
        else:
            self._init_measurements(measurements)
        if constantsfile is not None:
            # set up atomic constants table, default intensity units
            self._ac = constantsfile
            self._ac.add_index("Line")
            self._ac.add_index("J_u")
        self._column_density = dict()
        self._column_densities = dict()
        # Most recent cutout selected by user for computation area.
        self._cutout = None
        

    
    def _init_measurements(self,m):
        '''Initialize measurements dictionary given a list.

           :param m: list of intensity :class:`~pdrtpy.measurement.Measurement`s in units equivalent to :math:`{\\rm erg~cm^{-2}~s^{-1}~sr^{-1}}`
           :type m: list of :class:`~pdrtpy.measurement.Measurement`
        '''
        self._measurements = dict()
        for mm in m:
            if not utils.check_units(mm.unit,self._intensity_units):
                raise TypeError("Measurement " +mm.id + " units "+mm.unit.to_string()+" are not in intensity units equivalent to "+self._intensity_units)
            self._measurements[mm.id] = mm
        # re-initialize column densities
        
    def add_measurement(self,m):
        '''Add an intensity Measurement to internal dictionary used to 
           compute the excitation diagram.   This method can also be used
           to safely replace an existing intensity Measurement.

           :param m: A :class:`~pdrtpy.measurement.Measurement` instance containing intensity in units equivalent to :math:`{\\rm erg~cm^{-2}~s^{-1}~sr^{-1}}`
        '''
        if not utils.check_units(m.unit,self._intensity_units):
            raise TypeError("Measurement " +m.id + " must be in intensity units equivalent to "+self._intensity_units)

        if self._measurements:
            self._measurements[m.id] = m
            # if there is an existing column density with this ID, remove it
            self._column_densities.pop(m.id,None)
        else:
            self._init_measurements(m)

    def replace_measurement(self,m):
        '''Safely replace an existing intensity Measurement.  Do not 
           change a Measurement in place, use this method. 
           Otherwise, the column densities will be inconsistent.

           :param m: A :class:`~pdrtpy.measurement.Measurement` instance containing intensity in units equivalent to :math:`{\\rm erg~cm^{-2}~s^{-1}~sr^{-1}}`
        '''
        self.add_measurement(self,m)

# use?
class FitParams(object):
    def __init__(self,params,pcov):
        self._params = params
        self._pcov = pcov
        self._perr = np.sqrt(np.diag(pcov))
        if len(params)==5:
            self._opr=params[4]
            self.opr_fitted = True
        else:
            self._opr=3
            self.opr_fitted = False
        self._tunit = u.Unit("K")
        self._cdunit = u.Unit("cm-2")
        self._total_colden = dict()
        self._temperature = dict()
        self._compute_quantities()

    def _compute_quantities(self):
        loge=math.log10(math.e)
        ln10 = math.log(10)
        if self._params[2] <  self._params[0]:
            # cold and hot temperatures
            uc = self._perr[2]/self._params[2]
            tc = -loge/self._params[2]
            ucc = StdDevUncertainty(np.abs(tc*uc))
            self._temperature["cold"]=Measurement(data=tc,unit=self._tunit,uncertainty=ucc)
            uh = self._perr[0]/self._params[0]
            th = -loge/self._params[0]
            uch = StdDevUncertainty(np.abs(th*uh))
            self._temperature["hot"]=Measurement(data=th,unit=self._tunit,uncertainty=uch)
            # cold and hot total column density
            nc = 10**self._params[1]
            uc = ln10*self._perr[1]*nc
            ucn = StdDevUncertainty(np.abs(uc))
            self._total_colden["cold"] = Measurement(nc,unit=self._cdunit,uncertainty=ucn)
            nh = 10**self._params[3]
            uh = ln10*self._perr[3]*nh
            uhn = StdDevUncertainty(np.abs(uh))
            self._total_colden["hot"] = Measurement(nh,unit=self._cdunit,uncertainty=uhn)
        else:
            uc = self._perr[0]/self._params[0]
            tc = -loge/self._params[0]
            ucc = StdDevUncertainty(np.abs(tc*uc))
            self._temperature["cold"]=Measurement(data=tc,unit=self._tunit,uncertainty=ucc)
            uh = self._perr[2]/self._params[2]
            th = -loge/self._params[2]
            uch = StdDevUncertainty(np.abs(th*uh))
            self._temperature["hot"]=Measurement(data=th,unit=self._tunit,uncertainty=uch)
            # cold and hot total column density
            uc = ln10*self._perr[3]*self._params[3]
            nc = 10**self._params[3]
            uc = ln10*self._perr[3]*nc
            ucn = StdDevUncertainty(np.abs(uc))
            self._total_colden["cold"] = Measurement(nc,unit=self._cdunit,uncertainty=ucn)
            nh = 10**self._params[1]
            uh = ln10*self._perr[1]*nh
            uhn = StdDevUncertainty(np.abs(uh))
            self._total_colden["hot"] = Measurement(nh,unit=self._cdunit,uncertainty=uhn)
    
    def residuals(self,residuals):
        self._residuals = residuals
    
    @property
    def thot(self):
        return self._temperature["hot"]
    @property
    def tcold(self):
        return self._temperature["cold"]
    @property
    def coldenhot(self):
        return self._total_colden["hot"]
    @property 
    def coldencold(self):
        return self._total_colden["cold"]
    @property 
    def total_colden(self):
        return self._total_colden["hot"]+self._total_colden["cold"]
    @property
    def opr(self):
        return self._opr
    @property 
    def slopes(self):
        return [self._params[0],self._params[2]]
    @property 
    def intercepts(self):
        return [self._params[1],self._params[3]]
        
    def report(self):
        s = "Fit parameters\n"
        s += "Slopes :" + str(self.slopes) +"\n"
        s += "Intercepts :" + str(self.intercepts) +"\n"
        tcold = f'T_cold = {self.tcold.data:0.1f} +/-{ self.tcold.error:0.1f} {self._tunit:FITS}\n'
        thot = f'T_hot = {self.thot.data:0.1f} +/- {self.thot.error:0.1f} {self._tunit:FITS}\n'
        s += tcold
        s += thot
        ncold = f'N_cold = {self.coldencold.data:0.2e} +/- {self.coldencold.error:0.2e} {self._cdunit:FITS}\n'
        nhot = f'N_hot = {self.coldenhot.data:0.2e} +/- {self.coldenhot.error:0.2e} {self._cdunit:FITS}\n'
        ntot = f'N_total = {self.total_colden.data:0.2e} +/- {self.total_colden.error:0.2e} {self._cdunit:FITS}\n'
        s += ncold
        s += nhot
        s += ntot
        s += "OPR = "+str(self._opr)
        print(s)
        
class H2ExcitationFit(ExcitationFit):
    """Tool for fitting temperatures to :math:`H_2` Excitation Diagrams

       **This tool is still under development**
    """
    def __init__(self,measurements=None,
                 constantsfile=utils.get_table("atomic_constants.tab")):
        #print("H2ExcitationFit constructor")
        super().__init__(measurements,constantsfile)
        self._canonical_opr = 3.0
        self._opr_mask = None 
        self._fitparams = None

    @property
    def fit_params(self):
        return self._fitparams
    @property
    def opr_fitted(self):
        if self._fitparams is None: 
            return False
        return self._fitparams.opr_fitted
    @property
    def opr(self):
        if self.opr_fitted:
            return self._fitparams.opr
        else:
            return self._canonical_opr
        
    @property 
    def intensities(self):
        '''The stored intensities. See :meth:`add_measurement`
         
           :rtype: list of :class:`~pdrtpy.measurement.Measurement`
        '''
        return self._measurements  
    @property
    def total_colden(self):
        '''The fitted total column density
        
        :rtype: :class:`~pdrtpy.measurement.Measurement` 
        '''
        return self._fitparams.total_colden

    @property 
    def tcold(self):
        '''The fitted cold gas temperature
        
        :rtype: :class:`~pdrtpy.measurement.Measurement` 
        '''        
        return self._fitparams.tcold

    @property
    def thot(self):
        '''The fitted hot gas temperature
        
        :rtype: :class:`~pdrtpy.measurement.Measurement` 
        '''   
        return self._fitparams.thot
    
    def column_densities(self,norm=False,unit=utils._CM2):
        '''The computed upper state column densities of stored intensities

           :param norm: if True, normalize the column densities by the 
                       statistical weight of the upper state, :math:`g_u`.  
                       Default: False
           :type norm: bool

           :returns: dictionary of column densities indexed by Line name
           :rtype: dict
        '''
        # Compute column densities if needed. 
        # Note: this has a gotcha - if user changes an existing intensity 
        # Measurement in place, rather than replaceMeasurement(), the colden 
        # won't get recomputed. But we warned them!
        if not self._column_density or (len(self._column_density) != len(self._measurements)):
            self._compute_column_densities(unit=unit)
        if norm:
            cdnorm = dict()
            for cd in self._column_density:
                # This fails with complaints about units:
                #self._column_density[cd] /= self._ac.loc[cd]["g_u"]
                #gu = Measurement(self._ac.loc[cd]["g_u"],unit=u.dimensionless_unscaled)
                #print("1 normalizing")
                cdnorm[cd] = self._column_density[cd]/self._ac.loc[cd]["g_u"]
            return cdnorm
        else:
            return self._column_density

    def energies(self,line=False):
        '''Upper state energies of stored intensities, in K. 

           :param line: if True, the dictionary index is the Line name, 
                     otherwise it is the upper state :math:`J` number.  Default: False
           :type line: bool
           :returns: dictionary indexed by upper state :math:`J` number or Line name. 
           :rtype: dict
        '''
        t = dict()
        if line:
            for m in self._measurements:
                t[m] = self._ac.loc[m]["E_upper/k"]
        else:
            for m in self._measurements:
                t[self._ac.loc[m]["J_u"]] = self._ac.loc[m]["E_upper/k"]
        return t

    def run(self):
        cdavg = self.average_column_density(norm=True)
        energy = self.energies(line=False)
        z=np.array([np.hstack([cdavg[key],energy[key]]) for key in cdavg.keys()])
        x = Measurement(z[:,0],unit="K")
        y = Measurement(z[:1],unit=utils._CM2)

    def intensity(self,colden):
        # colden is N_upper
        dE = self._ac.loc[colden.id]["dE/k"]*constants.k_B.cgs*self._ac["dE/k"].unit
        A = self._ac.loc[colden.id]["A"]*self._ac["A"].unit
        v = A*dE/(4.0*math.pi*u.sr)
        val = Measurement(data=v.value,unit=v.unit,identifier=colden.id)
        intensity = val*colden # error will get propagated
        #print(intensity.unit,self._intensity_units)
        i = intensity.convert_unit_to(self._intensity_units)
        i._identifier = val.id
        return i

    def colden(self,intensity,unit):
        '''Compute the column density in upper state :math:`N_u`, given an 
           intensity :math:`I` and assuming optically thin emission.  
           Units of :math:`I` need to be equivalent to 
           :math:`{\\rm erg~cm^{-2}~s^{-1}~sr^{-1}}`.

           .. math::
                 I &= {A \Delta E~N_u \over 4\pi}

                 N_u &= 4\pi {I\over A\Delta E}

           where :math:`A` is the Einstein A coefficient and :math:`\Delta E` is the energy of the transition.

           :param intensity: A :class:`~pdrtpy.measurement.Measurement` instance containing intensity in units equivalent to :math:`{\\rm erg~cm^{-2}~s^{-1}~sr^{-1}}`
           :type intensity: :class:`~pdrtpy.measurement.Measurement`
           :param unit: The units in which to return the column density. Default: :math:`{\\rm }cm^{-2}`
           :type unit: str or :class:`astropy.unit.Unit`
           :returns: a :class:`~pdrtpy.measurement.Measurement` of the column density.
           :rtype: :class:`~pdrtpy.measurement.Measurement` 
        '''
        #print("calling colden")
        dE = self._ac.loc[intensity.id]["dE/k"]*constants.k_B.cgs*self._ac["dE/k"].unit
        A = self._ac.loc[intensity.id]["A"]*self._ac["A"].unit
        v = 4.0*math.pi*u.sr/(A*dE)
        #print("dE ",dE.unit,' A ',A.unit,' v ',v.unit)
        val = Measurement(data=v.value,unit=v.unit)
        N_upper = intensity * val # error will get propagated
        return N_upper.convert_unit_to(unit)

    def _compute_column_densities(self,unit):
        '''Compute all column densities for stored intensity measurements
           :param unit: The units in which to return the column density. Default: cm-2
           :type unit: str or :class:`astropy.unit.Unit`
           :returns: a :class:`~pdrtpy.measurement.Measurement` of the column density.
        '''
        for m in self._measurements:
            self._column_density[m] = self.colden(self._measurements[m],unit)

    def gu(self,key,opr):
        if utils.isEven(self._ac.loc[key]["J_u"]):
            return self._ac.loc[key]["g_u"]
        else:
            print("Ju=%d scaling by [%.2f/%.2f]=%.2f"%(self._ac.loc[key]["J_u"],opr,self._canonical_opr,opr/self._canonical_opr))
            return self._ac.loc[key]["g_u"]*opr/self._canonical_opr
        
    def average_column_density(self,position,size,norm=True,
                               unit=utils._CM2,line=False,test=False):
        r'''Compute the average column density over a spatial box.  The box is created using :class:`astropy.nddata.utils.Cutout2D`.

        :param position: The position of the cutout array's center with respect to the data array. The position can be specified either as a `(x, y)` tuple of pixel coordinates or a :class:`~astropy.coordinates.SkyCoord`, which will use the :class:`~astropy.wcs.WCS` of the ::class:`~pdrtpy.measurement.Measurement`s added to this tool. See :class:`~astropy.nddata.utils.Cutout2D`.
        :type position: tuple or :class:`astropy.coordinates.SkyCoord`
        :param size: The size of the cutout array along each axis. If size is a scalar number or a scalar :class:`~astropy.units.Quantity`, then a square cutout of size will be created. If `size` has two elements, they should be in `(ny, nx)` order. Scalar numbers in size are assumed to be in units of pixels. `size` can also be a :class:`~astropy.units.Quantity` object or contain :class:`~astropy.units.Quantity` objects. Such :class:`~astropy.units.Quantity` objects must be in pixel or angular units. For all cases, size will be converted to an integer number of pixels, rounding the the nearest integer. See the mode keyword for additional details on the final cutout size.
        :type size: int, array_like, or :class:`astropy.units.Quantity`
        :param norm: if True, normalize the column densities by the 
                       statistical weight of the upper state, :math:`g_u`.  For ortho-$H_2$ $g_u = OPR \times (2J+1)$, for para-$H_2$ $g_u=2J+1$. In LTE, $OPR = 3$.
        :type norm: bool
        :param unit: The units in which to return the column density. Default: :math:`{\rm cm}^{-2}` 
        :type unit: str or :class:`astropy.unit.Unit`
        :param line: if True, the returned dictionary index is the Line name, otherwise it is the upper state :math:`J` number.  
        :type line: bool
        :returns: dictionary of column density Measurements, with keys as :math:`J` number or Line name
        :rtype:  dict
        '''

        # Possibly modify error calculation, see https://pypi.org/project/statsmodels/
        # https://stackoverflow.com/questions/2413522/weighted-standard-deviation-in-numpy
        # Doesn't come out too different from np.sqrt(np.cov(values, aweights=weights))
        
        # Set norm=False because we normalize below if necessary.
        cdnorm = self.column_densities(norm=False,unit=unit)
        cdmeas = dict()
        for cd in cdnorm:
            ca = cdnorm[cd]
            self._cutout = Cutout2D(ca.data, position, size, ca.wcs, mode='partial', fill_value=np.nan)
            w= Cutout2D(ca.uncertainty.array, position, size, ca.wcs, mode='partial', fill_value=np.nan)
            cddata = np.ma.masked_array(self._cutout.data,np.isnan(self._cutout.data))
            weights = np.ma.masked_array(w.data,np.isnan(w.data))
            #print("W Shape %s data shape %s"%(w.shape,cddata.shape))
            cdavg = np.average(cddata,weights=weights)
            #error = np.sqrt(np.cov(cddata.flatten(),aweights=weights.flatten()))
            error = np.nanmean(ca.error)
            #weighted_stats = DescrStatsW(cddata.flatten(), weights=weights.flatten(), ddof=0)
            #print("NP %e %e"%(cdavg, error))
            #print("Stats %e %e %e"%(weighted_stats.mean, weighted_stats.std ,  weighted_stats.std_mean))
            if line:
                index = cd
            else:
                index = self._ac.loc[cd]["J_u"]
            if norm:
                #print("2 normalizing")
                #print("dividing by %f"%self._ac.loc[cd]["g_u"])
                if test:
                    cdavg /= self.gu(cd,self.opr)
                    error /= self.gu(cd,self.opr)
                else:
                    cdavg /= self._ac.loc[cd]["g_u"]
                    error /= self._ac.loc[cd]["g_u"]
            cdmeas[index] = Measurement(data=cdavg, uncertainty=StdDevUncertainty(error),
                                        unit=ca.unit, identifier=cd)
             
        return cdmeas

    def _set_opr_mask(self,ids):
        # need to figure out which measurements are odd J and set mask=True for those, False for even J
        # Do this by lookup in atomic_constants.tab
        self._opr_mask = self._ac.loc[ids]["J_u"]%2!=0

    def _slopesfromguess(self,guess):
        if guess[0]<guess[1]:
            thot = guess[1]
            tcold = guess[2]
        else:
            tcold=guess[2]
            thot = guess[1]
        slope = []
        slope[0] = -utils.LOGE/tcold
        slope[1] = -utils.LOGE/thot
        return slope

    def _first_guess(self,x,y):
        r"""The first guess at the fit parameters is done by finding the line between the first two (lowest energy) points to determine $T_{cold}and between the last two (highest energy) points to determine $T_{hot}. The first guess is needed to ensure the final fit converges.  The guess doesn't need to be perfect, just in the ballpark.
        
        :param x: array of energies, $E/k$
        :type x: numpy array
        :param y: array of normalized column densities $N_u/g_u$
        :type y: numpy array
        """
        slopecold = (y[1]-y[0])/(x[1]-x[0])
        slopehot = (y[-1]-y[-2])/(x[-1]-x[-2])
        intcold = y[1] - slopecold*x[1]
        inthot  = y[-1] - slopehot*x[-1]
        return [slopecold, intcold, slopehot, inthot]
    
    def fit_excitation(self,position,size,fit_opr=False,**kwargs):
        """Fit the :math:`log N_u-E` diagram with two excitation temperatures,
        a ``hot`` :math:`T_{ex}` and a ``cold`` :math:`T_{ex}`.  A first
        pass guess is initially made using data partitioning and two
        linear fits.
        :param energy: Eu/k
        :type energy: :class:`~pdrtpy.measurement.Measurement` containing list of values.
        :param colden: list of log(Nu/gu)
        :type colden: :class:`~pdrtpy.measurement.Measurement` containing list of values and errors

        :returns: The fit parameters as in :mod:`scipy.optimize.curve_fit`
        :rtype: list
        """
        energy = self.energies(line=True)
        _ee = np.array([c for c in energy.values()])
        _energy = Measurement(_ee,unit="K")
        _ids = list(energy.keys())
        #if type(energy) is dict:
        #    _ee = np.array([c for c in energy.values()])
        #    _energy = Measurement(_ee,unit="K")
        #    _ids = list(energy.keys())
        #else:
        #    _energy = energy
        #    _ids = []
        #     for e in _energy:
        #        _ids.append(e.id)
        self._set_opr_mask(_ids)
        # Get Nu/gu 
        colden=self.average_column_density(norm=True,position=position,size=size,line=True,test=True)
        #if type(colden) is dict:
        # Need to stuff the data into a single vector
        _cd = np.array([c.data for c in colden.values()])
        _er  = np.array([c.error for c in colden.values()])
        _colden = Measurement(_cd,uncertainty=StdDevUncertainty(_er),unit="cm-2")
        #else:
        #    _colden = colden
        #    _error = error
        x = _energy.data
        y = np.log10(_colden.data)
        kwargs_opts = {"guess": self._first_guess(x,y)}
        kwargs_opts.update(kwargs)
        sigma = utils.LOGE*_colden.error/_colden.data
        if False:
            print("Energy = ",x,type(x))
            print("Colden = ",y,type(y))
            print("SIGMA = ",sigma,type(sigma))
            print("MASK = ",self._opr_mask)
            #print("GUESS = ",kwargs_opts["guess"])
        if kwargs_opts["guess"] is None: 
            fit_param, pcov = curve_fit(self._two_lines, xdata=x, ydata=y,sigma=sigma,maxfev=100000)
            #print("FIT_PARAM [slope,intercept,slope,intercept] :",fit_param)
        else:
            fit_param = kwargs_opts["guess"]
            #print("FIT PARAM GUESS [slope,intercept,slope,intercept]:",fit_param) 
        
        # There is no guarantee of order in fit_param except it will be [slope, intercept, slope, intercept].
        # Thus we have to check which slope is actually more steeply negative to discover which is T_cold.
        if fit_param[2] < fit_param[0]:
            tcold=(-utils.LOGE/fit_param[2])*u.Unit("K")
            thot=(-utils.LOGE/fit_param[0])*u.Unit("K")
        else:
            tcold=(-utils.LOGE/fit_param[0])*u.Unit("K")
            thot=(-utils.LOGE/fit_param[2])*u.Unit("K")
        print("First guess at excitation temperatures: T_cold = %.1f, T_hot = %.1f "%(tcold.value,thot.value))
        #if m1 != m2:
        #    x_intersect = (n2 - n1) / (m1 - m2)
        #    print(x_intersect)
        #else:
        #    print("did not find two linear components")

        # Now do second pass fit to full curve using first pass as initial guess
        if fit_opr:
            fit_param=np.append(fit_param,[3]) # add opr
            bounds = ([-np.inf,-np.inf,-np.inf,-np.inf,0],[np.inf,np.inf,np.inf,np.inf,3])
            fit_param2,pcov2 = curve_fit(self._exc_func_opr, xdata=x, ydata=y, p0=fit_param, 
                                         sigma=sigma ,maxfev=1000000, bounds=bounds)
        else:
            fit_param2,pcov2 = curve_fit(self._exc_func, x, y, p0=fit_param, sigma=sigma)
        self._fitparams = FitParams(fit_param2,pcov2)
        #self._fitparams.report()

        if False:
            if fit_param2[2] < fit_param2[0]:
                _a =  [fit_param2[2], fit_param2[3], fit_param2[0], fit_param2[1]]
                if fit_opr:
                    _a.append(fit_param2[4])
                self._fitted_params = [ _a, pcov2 ]
                self._tcold=-utils.LOGE/fit_param2[2]*u.Unit("K")
                self._thot=-utils.LOGE/fit_param2[0]*u.Unit("K")
                self._total_colden["cold"] = 10**self._fitted_params[0][1]*u.Unit("cm-2")
                self._total_colden["hot"] = 10**self._fitted_params[0][3]*u.Unit("cm-2")
            else:
                self._fitted_params = [ fit_param2, pcov2]
                self._tcold=-utils.LOGE/fit_param2[0]*u.Unit("K")
                self._thot=-utils.LOGE/fit_param2[2]*u.Unit("K")
                self._total_colden["hot"] = 10**self._fitted_params[0][1]*u.Unit("cm-2")
            self._total_colden["cold"] = 10**self._fitted_params[0][3]*u.Unit("cm-2")         
      
            text = f'Fitted excitation temperatures:T_cold = {self.tcold.data:0.1f}+/-{self.tcold.error:0.1f} K, T_hot={self.thot.data:0.1f}+/-{self.thot.error:0.1f} K'
            print(text)
            if fit_opr:
                texto = f'Fitted Ortho-to-para ratio:{self.opr:0.2f}'
                print(texto)
            text2 = rf'Fitted total column density: N(H_2) = {self.total_colden.data:.1e}'
            print(text2)
            
        if fit_opr:
            r = y - self._exc_func_opr(x, *fit_param2)
        else:
            r = y - self._exc_func(x, *fit_param2)
        #print("Residuals: %.3e"%np.sum(np.square(r)))
        self._fitparams.residuals(r)
        return  self._fitparams

    def _two_lines(self,x, m1, n1, m2, n2):
        '''This function is used to partition a fit to data using two lines and 
           an inflection point.  Second slope is steeper because slopes are 
           negative in excitation diagram.

           :param x: array of x values
           :type x: :class:`numpy.ndarray` 
           :param m1: slope of first line
           :type m1: float
           :param n1: intercept of first line
           :type n1: float
           :param m2: slope of second line
           :type m2: float
           :param n2: intercept of second line
           :type n2: float

            See https://stackoverflow.com/questions/48674558/how-to-implement-automatic-model-determination-and-two-state-model-fitting-in-py
        ''' 
        return np.max([m1 * x + n1, m2 * x + n2], axis = 0)

    def _one_line(self,x,m1,n1):
        '''Return a line.

           :param x: array of x values
           :type x: :class:`numpy.ndarray` 
           :param m1: slope of first line
           :type m1: float
           :param n1: intercept of first line
           :type n1: float
        '''
        return m1*x+n1

    def _exc_func(self,x, m1, n1, m2, n2):
        '''Function for fitting the excitation curve as sum of two linear functions.

           :param x: array of x values
           :type x: :class:`numpy.ndarray` 
           :param m1: slope of first line
           :type m1: float
           :param n1: intercept of first line
           :type n1: float
           :param m2: slope of second line
           :type m2: float
           :param n2: intercept of second line
           :type n2: float

           :return: Sum of lines in log space: log10(10**(x*m1+n1) + 10**(x*m2+n2))
           :rtype: :class:`numpy.ndarray` 
        '''
        y1 = 10**(x*m1+n1)
        y2 = 10**(x*m2+n2)
        retval = np.log10(y1+y2)
        return retval

    def _exc_func_opr(self,x, m1, n1, m2, n2,opr):
#@TODO if enough values, compute separate OPRs Tcold and Thot. (See Neufeld et al 2006)
        '''Function for fitting the excitation curve as sum of two linear functions 
           and allowing ortho-to-para ratio to vary.  Para is even J, ortho is odd J.
           A single OPR value 

           :param x: masked array of x values where mask is TRUE indicates the x value is an odd J transition and the OPR can vary.  Even J transitions have OPR=1 always.
           :type x: :class:`numpy.ma.masked_array`
           :param m1: slope of first line
           :type m1: float
           :param n1: intercept of first line
           :type n1: float
           :param m2: slope of second line
           :type m2: float
           :param n2: intercept of second line
           :type n2: float
           :param opr: ortho-to-para ratio
           :type opr: float

           :return: Sum of lines in log space: log10(10**(x*m1+n1) + 10**(x*m2+n2))
           :rtype: :class:`numpy.ndarray` 
        '''
        # We assume that the column densities passed in have been normalized 
        # using the canonical OPR=3. Therefore what we are actually fitting is 
        # the ratio of the actual OPR to the canonical OPR.
        # For even J, input x = Nu/(3*(2J+1), so we back that out by multiplying 
        # by [canonical OPR]/[fit guess OPR]. 
        rat = self._canonical_opr/opr
        #print(opr)
        opr_array = np.ma.masked_array(rat*np.ones(x.size),mask=self._opr_mask)
        y1 = 10**(x.data*m1+n1)*opr_array
        #print(y1)
        y2 = 10**(x.data*m2+n2)*opr_array
        #print(y2)
        retval = np.log10(y1.data+y2.data)
        #print(retval,opr)
        return retval
