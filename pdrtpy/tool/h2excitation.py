from astropy.nddata import Cutout2D
import astropy.units as u
import astropy.constants as constants
from astropy.nddata import StdDevUncertainty
import math
import numpy as np
from lmfit import Parameters, fit_report
from lmfit.model import Model, ModelResult
from .toolbase import ToolBase
from .. import pdrutils as utils
from ..measurement import Measurement
import warnings

class ExcitationFit(ToolBase):
    """Base class for creating excitation fitting tools for various species.

    :param measurements: Input measurements to be fit.  
    :type measurements: array or dict `~pdrtpy.measurement.Measurement`. If dict, the keys should be the Measurement *identifiers*.  
    """
    def __init__(self,measurements=None,constantsfile=None):
        super().__init__()
        # must be set before call to init_measurements
        self._intensity_units = "erg cm^-2 s^-1 sr^-1"
        self._cd_units = 'cm^-2'
        self._t_units = "K"
        if type(measurements) == dict or measurements is None:
            self._measurements = measurements
        else:
            self._init_measurements(measurements)
        self._set_measurementnaxis()
        if constantsfile is not None:
            # set up atomic constants table, default intensity units
            self._ac = utils.get_table(constantsfile)
            self._ac.add_index("Line")
            self._ac.add_index("J_u")
        #@todo we don't really even use this.  CD's are computed on the fly in average_column_density()
        self._column_density = dict()      
       
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
            self._column_density.pop(m.id,None)
        else:
            self._init_measurements(m)
            
    def remove_measurement(self,id):
        '''Delete a measurement from the internal dictionary used to compute column densities. Any associated column density will also be removed.

           :param id: the measurement identifier
           :type id: str
           :raises KeyError: if id not in existing Measurements
        '''
        del self._measurements[id] # we want this to raise a KeyError if id not found
        self._column_density.pop(m.id,None) # but not this.

    def replace_measurement(self,m):
        '''Safely replace an existing intensity Measurement.  Do not 
           change a Measurement in place, use this method. 
           Otherwise, the column densities will be inconsistent.

           :param m: A :class:`~pdrtpy.measurement.Measurement` instance containing intensity in units equivalent to :math:`{\\rm erg~cm^{-2}~s^{-1}~sr^{-1}}`
        '''
        self.add_measurement(self,m)
        
class H2ExcitationFit(ExcitationFit):
    """Tool for fitting temperatures to :math:`H_2` Excitation Diagrams
    """
    def __init__(self,measurements=None,
                 constantsfile="atomic_constants.tab"):
        super().__init__(measurements,constantsfile)
        self._canonical_opr = 3.0
        self._init_params()
        self._init_model()
        self._fitresult = None
        self._temperature = None
        self._total_colden = None
        # position and size that was used for averaging/fit
        self._position = None  
        self._size = None
        
    def _init_params(self):
        #fit input parameters
        self._params = Parameters()
        # we have to have opr max be greater than 3 so that fitting will work.
        # the fit algorithm does not like when the initial value is pinned at one
        # of the limits
        self._params.add('opr',value=3.0,min=1.0,max=3.5,vary=False)
        self._params.add('m1',value=0,min=-1,max=1)
        self._params.add('n1',value=15,min=0,max=30)
        self._params.add('m2',value=0,min=-1,max=1)
        self._params.add('n2',value=15,min=0,max=30)
        
    def _residual(self,params,x,data,error,idx):
        # We assume that the column densities passed in have been normalized 
        # using the canonical OPR=3. Therefore what we are actually fitting is 
        # the ratio of the actual OPR to the canonical OPR.
        # For odd J, input x = Nu/(3*(2J+1) where 3=canonical OPR. 
        #
        # We want the model-data residual to be small, but if the opr 
        # is different from the  canonical value of 3, then data[idx] will 
        # be low by a factor of 3/opr.
        # So we must LOWER model[idx] artificially by dividing it by 
        # 3/opr, i.e. multiplying by opr/3.  This is equivalent to addition in log-space.
        p = params.valuesdict()
        y1 = 10**(x*p['m1']+p['n1'])            
        y2 = 10**(x*p['m2']+p['n2'])
        model = np.log10(y1+y2)
        if params['opr'].vary:
            model += np.log10(p['opr']/self._canonical_opr)
        return (model - data)/error
    
    def _modelfunc(self,x,m1,n1,m2,n2,opr,idx=[],fit_opr=False):
        '''Function for fitting the excitation curve as sum of two linear functions 
           and allowing ortho-to-para ratio to vary.  Para is even J, ortho is odd J.
           :param x: independent axis array
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
           :type idx: np.ndarray
           :param idx: list of indices that may have variable opr (odd J transitions)
           :param fit_opr: indicate whether opr will be fit, default False (opr fixed)
           :type fit_opr: False
           :return: Sum of lines in log space:log10(10**(x*m1+n1) + 10**(x*m2+n2)) + log10(opr/3.0)
           :rtype: :class:`numpy.ndarray` 
        '''
        y1 = 10**(x*m1+n1)            
        y2 = 10**(x*m2+n2)
        model = np.log10(y1+y2)
        # We assume that the column densities passed in have been normalized 
        # using the canonical OPR=3. Therefore what we are actually fitting is 
        # the ratio of the actual OPR to the canonical OPR.
        # For odd J, input x = Nu/(3*(2J+1) where 3=canonical OPR. 
        #
        # We want the model-data residual to be small, but if the opr 
        # is different from the  canonical value of 3, then data[idx] will 
        # be low by a factor of 3/opr.
        # So we must LOWER model[idx] artificially by dividing it by 
        # 3/opr, i.e. multiplying by opr/3.  This is equivalent to addition in log-space.
        if fit_opr:
            model[idx] += np.log10(opr/self._canonical_opr)
        return model
    
    def _init_model(self):
        #@todo make a separate class that subclasses Model. 
        # potentially allow users to change it.
        self._model=Model(self._modelfunc)
        for p,q in self._params.items():
            self._model.set_param_hint(p, value=q.value, 
                                       min=q.min, max=q.max, 
                                       vary=q.vary)
            self._model.make_params()
    
    def _compute_quantities(self,params):
        """Compute the temperatures and column densities for the hot and cold gas components.  This method will set class variables `_temperature` and `_colden`.
        
        :param params: The fit parameters returned from fit_excitation.
        :type params: :class:`lmfit.Parameters`
        """
        self._temperature = dict()
        # N(J=0) column density = intercept on y axis               
        self._j0_colden = dict()
        # total column density = N(J=0)*Z(T) where Z(T) is partition function
        self._total_colden = dict()

        if params['m2'] <  params['m1']:
            cold = '2'
            hot = '1'
        else:
            cold = '1'
            hot = '2'
        mcold = 'm'+cold
        mhot= 'm'+hot
        ncold = 'n'+cold
        nhot = 'n'+hot
        # cold and hot temperatures
        uc = params[mcold].stderr/params[mcold]
        tc = -utils.LOGE/params[mcold]
        ucc = StdDevUncertainty(np.abs(tc*uc))
        self._temperature["cold"]=Measurement(data=tc,unit=self._t_units,uncertainty=ucc)
        uh = params[mhot].stderr/params[mhot]
        th = -utils.LOGE/params[mhot]
        uch = StdDevUncertainty(np.abs(th*uh))
        self._temperature["hot"]=Measurement(data=th,unit=self._t_units,uncertainty=uch)
        # cold and hot total column density
        nc = 10**params[ncold]
        uc = utils.LN10*params[ncold].stderr*nc
        ucn = StdDevUncertainty(np.abs(uc))
        self._j0_colden["cold"] = Measurement(nc,unit=self._cd_units,uncertainty=ucn)
        nh = 10**params[nhot]
        uh = utils.LN10*params[nhot].stderr*nh
        uhn = StdDevUncertainty(np.abs(uh))
        self._j0_colden["hot"] = Measurement(nh,unit=self._cd_units,uncertainty=uhn)
        self._total_colden["cold"] = self._j0_colden["cold"] * self._partition_function(self.tcold)
        self._total_colden["hot"] = self._j0_colden["hot"] * self._partition_function(self.thot)      
        self._opr = Measurement(self._fitresult.params['opr'].value,
                                unit=u.dimensionless_unscaled, 
                        uncertainty=StdDevUncertainty(self._fitresult.params['opr'].stderr))
        
        
    @property
    def fit_result(self):
        '''The result of the fitting procedure which includes fit statistics, variable values and uncertainties, and correlations between variables.
        
        :rtype: :class:`lmfit.modelresult.ModelResult`
        '''
        return self._fitresult
    
    @property
    def opr_fitted(self):
        '''Was the ortho-to-para ratio fitted?
        
        :returns: True if OPR was fitted, False if canonical LTE value was used
        :rtype: bool
        '''
        if self._fitresult is None: 
            return False
        return self._fitresult.params['opr'].vary
    
    @property
    def opr(self):
        '''The ortho-to-para ratio (OPR)
        
        :returns: The fitted OPR is it was determined in the fit, otherwise the canonical LTE OPR
        :rtype: float
        '''
        if self.opr_fitted:
            return self._opr
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
        return self._total_colden["hot"]+self._total_colden["cold"]

    @property
    def hot_colden(self):
        '''The fitted hot gas total column density
        
        :rtype: :class:`~pdrtpy.measurement.Measurement`         
        '''
        return self._total_colden["hot"]
    
    @property 
    def cold_colden(self):
        '''The fitted cold gas total column density
        
        :rtype: :class:`~pdrtpy.measurement.Measurement`         
        '''
        return self._total_colden["cold"]

    @property 
    def tcold(self):
        '''The fitted cold gas excitation temperature
        
        :rtype: :class:`~pdrtpy.measurement.Measurement` 
        '''        
        return self._temperature['cold']#self._fitparams.tcold

    @property
    def thot(self):
        '''The fitted hot gas excitation temperature
        
        :rtype: :class:`~pdrtpy.measurement.Measurement` 
        '''   
        return self._temperature['hot']#self._fitparams.thot
    
    def column_densities(self,norm=False,unit=utils._CM2, line=False):
        r'''The computed upper state column densities of stored intensities

           :param norm: if True, normalize the column densities by the 
                       statistical weight of the upper state, :math:`g_u`.  
                       Default: False
           :type norm: bool
           :param unit: The units in which to return the column density. Default: :math:`{\\rm }cm^{-2}`
           :type unit: str or :class:`astropy.units.Unit`
           :param line: if True, the dictionary index is the Line name, 
                     otherwise it is the upper state :math:`J` number.  Default: False
           :type line: bool

           :returns: dictionary of column densities indexed by upper state :math:`J` number or Line name. Default: False means return indexed by :math:`J`.
           :rtype: dict
        '''
        # Compute column densities if needed. 
        # Note: this has a gotcha - if user changes an existing intensity 
        # Measurement in place, rather than replaceMeasurement(), the colden 
        # won't get recomputed. But we warned them!
        #if not self._column_density or (len(self._column_density) != len(self._measurements)):
        
        # screw it. just always compute them.  Note to self: change this if it becomes computationally intensive
        self._compute_column_densities(unit=unit,line=line)
        if norm:
            cdnorm = dict()
            for cd in self._column_density:
                if line:
                    denom = self._ac.loc[cd]["g_u"]
                else:
                    denom = self._ac.loc['J_u',cd]["g_u"]
                # This fails with complaints about units:
                #self._column_density[cd] /= self._ac.loc[cd]["g_u"]
                #gu = Measurement(self._ac.loc[cd]["g_u"],unit=u.dimensionless_unscaled)
                cdnorm[cd] = self._column_density[cd]/denom
            #return #self._column_density
            return cdnorm
        else:
            return self._column_density

    def energies(self,line=False):
        '''Upper state energies of stored intensities, in K. 

           :param line: if True, the dictionary index is the Line name, 
                     otherwise it is the upper state :math:`J` number.  Default: False
           :type line: bool
           :returns: dictionary indexed by upper state :math:`J` number or Line name. Default: False means return indexed by :math:`J`.
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

    def run(self,position=None,size=None,fit_opr=False,**kwargs):
        r'''Fit the :math:`log N_u-E` diagram with two excitation temperatures,
        a ``hot`` :math:`T_{ex}` and a ``cold`` :math:`T_{ex}`. 
        
        If ``position`` and ``size`` are given, the data will be averaged over a spatial box before fitting.  The box is created using :class:`astropy.nddata.utils.Cutout2D`.  If position or size is None, the data are averaged over all pixels.  If the Measurements are single values, these arguments are ignored.

        :param position: The position of the cutout array's center with respect to the data array. The position can be specified either as a `(x, y)` tuple of pixel coordinates.
        :type position: tuple 
        :param size: The size of the cutout array along each axis in pixels. If size is a scalar number or a scalar :class:`~astropy.units.Quantity`, then a square cutout of size will be created. If `size` has two elements, they should be in `(nx, ny)` order [*this is the opposite of Cutout2D signature*]. Scalar numbers in size are assumed to be in units of pixels.  Default value of None means use all pixels (position is ignored)
        :type size: int, array_like`
        :param fit_opr: Whether to fit the ortho-to-para ratio or not. If True, the OPR will be varied to determine the best value. If False, the OPR is fixed at the canonical LTE value of 3.
        :type fit_opr: bool
        :returns: The fit result which contains slopes, intercepts, the ortho to para ratio (OPR), and fit statistics
        :rtype:  :class:`lmfit.model.ModelResult`      
        '''
        return self._fit_excitation(position,size,fit_opr,**kwargs)

    def intensity(self,colden):
        '''Given an upper state column density :math:`N_u`, compute the intensity :math:`I`.  

           .. math::
                 I = {A \Delta E~N_u \over 4\pi}     
              
        where :math:`A` is the Einstein A coefficient and :math:`\Delta E` is the energy of the transition.     
        
        :param colden: upper state column density
        :type colden: :class:`~pdrtpy.measurement.Measurement`
        :returns: optically thin intensity 
        :rtype: :class:`~pdrtpy.measurement.Measurement`
        '''
        # colden is N_upper
        dE = self._ac.loc[colden.id]["dE/k"]*constants.k_B.cgs*self._ac["dE/k"].unit
        A = self._ac.loc[colden.id]["A"]*self._ac["A"].unit
        v = A*dE/(4.0*math.pi*u.sr)
        val = Measurement(data=v.value,unit=v.unit,identifier=colden.id)
        intensity = val*colden # error will get propagated
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
           :type unit: str or :class:`astropy.units.Unit`
           :returns: a :class:`~pdrtpy.measurement.Measurement` of the column density.
           :rtype: :class:`~pdrtpy.measurement.Measurement` 
        '''

        dE = self._ac.loc[intensity.id]["dE/k"] * constants.k_B.cgs * self._ac["dE/k"].unit
        A = self._ac.loc[intensity.id]["A"]*self._ac["A"].unit
        v = 4.0*math.pi*u.sr/(A*dE)
        val = Measurement(data=v.value,unit=v.unit)
        N_upper = intensity * val # error will get propagated
        return N_upper.convert_unit_to(unit)

    def _compute_column_densities(self,unit=utils._CM2,line=False):
        r'''Compute all upper level column densities for stored intensity measurements and puts them in a dictionary
           :param unit: The units in which to return the column density. Default: :math:`{\\rm }cm^{-2}`
           :type unit: str or :class:`astropy.units.Unit`
           :param line: if True, the dictionary index is the Line name, 
                     otherwise it is the upper state :math:`J` number.  Default: False
           :type line: bool
        
            # should we reutrn something here or just compute them and never store.
            # I'm beginning to think there is no reason to store them.
           #:returns: dictionary of column densities as:class:`~pdrtpy.measurement.Measurement  indexed by upper state :math:`J` number or Line name. Default: False means return indexed by :math:`J`.
           #:returns: a :class:`~pdrtpy.measurement.Measurement` of the column density.
        '''
        self._column_density = dict()
        for m in self._measurements:
            if line:
                index = m
            else:
                index = self._ac.loc[m]["J_u"]             
            self._column_density[index] = self.colden(self._measurements[m],unit)

    def gu(self,id,opr):
        r'''Get the upper state statistical weight $g_u$ for the given transition identifer, and, if the transition is odd-$J$, scale the result by the given ortho-to-para ratio.  If the transition is even-$J$, the LTE value is returned.
        
           :param id: the measurement identifier
           :type id: str
           :param opr:
           :type opr: float
           :raises KeyError: if id not in existing Measurements 
           :rtype: float
        '''
        if utils.isEven(self._ac.loc[id]["J_u"]):
            return self._ac.loc[id]["g_u"]
        else:
            #print("Ju=%d scaling by [%.2f/%.2f]=%.2f"%(self._ac.loc[id]["J_u"],opr,self._canonical_opr,opr/self._canonical_opr))
            return self._ac.loc[id]["g_u"]*opr/self._canonical_opr
        
    def average_column_density(self,position=None,size=None,norm=True,
                               unit=utils._CM2,line=False, clip=-1E40*u.Unit("cm-2")):
        r'''Compute the average column density over a spatial box.  The box is created using :class:`astropy.nddata.utils.Cutout2D`.

        :param position: The position of the cutout array's center with respect to the data array. The position can be specified either as a `(x, y)` tuple of pixel coordinates.
        :type position: tuple 
        :param size: The size of the cutout array along each axis. If size is a scalar number or a scalar :class:`~astropy.units.Quantity`, then a square cutout of size will be created. If `size` has two elements, they should be in `(nx,ny)` order [*this is the opposite of Cutout2D signature*]. Scalar numbers in size are assumed to be in units of pixels.  Default value of None means use all pixels (position is ignored)
        :type size: int, array_like`
        :param norm: if True, normalize the column densities by the 
                       statistical weight of the upper state, :math:`g_u`.  For ortho-$H_2$ $g_u = OPR \times (2J+1)$, for para-$H_2$ $g_u=2J+1$. In LTE, $OPR = 3$.
        :type norm: bool
        :param unit: The units in which to return the column density. Default: :math:`{\rm cm}^{-2}` 
        :type unit: str or :class:`astropy.units.Unit`
        :param line: if True, the returned dictionary index is the Line name, otherwise it is the upper state :math:`J` number.  
        :type line: bool
        :returns: dictionary of column density Measurements, with keys as :math:`J` number or Line name
        :rtype:  dict
        :param clip: Column density value at which to clip pixels. Pixels with column densities below this value will not be used in the average. Default: a large negative number, which translates to no clipping.  
        :type clip: :class:`astropy.units.Quantity`
        '''
        #@todo
        # - should default clip = None?
    
        # Set norm=False because we normalize below if necessary.
        if position is not None and size is None:
            print("WARNING: ignoring position keyword since no size given")
        if position is None and size is not None:
            raise Exception("You must supply a position in addition to size for cutout")
        if size is not None:
            if np.isscalar(size):
                size = np.array([size,size])
            else:
                #Cutout2D wants (ny,nx)
                size = np.array([size[1],size[0]])
        
        clip = clip.to("cm-2")
        cdnorm = self.column_densities(norm=norm,unit=unit,line=line)
        cdmeas = dict()
        for cd in cdnorm:
            ca = cdnorm[cd]
            if size is not None:
                if len(size) != len(ca.shape):
                    raise Exception(f"Size dimensions [{len(size)}] don't match measurements [{len(ca.shape)}]")


                #if size[0] > ca.shape[0] or size[1] > ca.shape[1]:
                #    raise Exception(f"Requested cutout size {size} exceeds measurement size {ca.shape}")
                cutout = Cutout2D(ca.data, position, size, ca.wcs, mode='trim', fill_value=np.nan)
                w= Cutout2D(ca.uncertainty.array, position, size, ca.wcs, mode='trim', fill_value=np.nan)
                cddata = np.ma.masked_array(cutout.data,mask=np.ma.mask_or(np.isnan(cutout.data),cutout.data<clip.value))
                weights = np.ma.masked_array(w.data,np.isnan(w.data))
                if False:
                    # save cutout as a test that we have the x,y correct in size param
                    t = Measurement(cddata,unit=ca.unit,uncertainty=StdDevUncertainty(weights),identifier=ca.id)
                    t.write("cutout.fits",overwrite=True)
                    
            else: 
                cddata = ca.data
                # handle corner case of measurment.data is shape = (1,)
                # and StdDevUncertainty.array is shape = ().
                # They both have only one value but StdDevUncertainty stores
                # its data in a peculiar way.
                # alternative: check that type(ca.uncertainty.array) == np.ndarray would also work.
                if np.shape(ca.data) == (1,) and np.shape(ca.uncertainty.array) == ():
                    weights = np.array([ca.uncertainty.array])
                else:
                    weights = ca.uncertainty.array
            cdavg = np.average(cddata,weights=weights)
            error = np.nanmean(ca.error)/np.sqrt(ca.error.size)#-1
            cdmeas[cd] = Measurement(data=cdavg, 
                                        uncertainty=StdDevUncertainty(error),
                                        unit=ca.unit, identifier=cd)            
        return cdmeas
        
    def _get_ortho_indices(self,ids):
        """Given a list of J values, return the indices of those that are ortho
        transitions (odd J)
        
        :param ids:
        :type ids: list of str 
        :returns: The array indices of the odd J values.
        :rtype: list of int
        """
        return np.where(self._ac.loc[ids]["J_u"]%2!=0)[0]
    
    def _get_para_indices(self,ids):
        """Given a list of J values, return the indices of those that are para
        transitions (even J)
        
        :param ids:
        :type ids: list of str 
        :returns: The array indices of the even J values.
        :rtype: list of int
        """
        return np.where(self._ac.loc[ids]["J_u"]%2==0)[0]
    
    # currently unused.  in future may allow users to give first guesses at the temperatures, though not clear these will be better than _firstguess().  plus this does nothing for the intercepts  
    def _slopesfromguess(self,guess):
        """given a guess of two temperatures, compute slopes from them"""
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

    def _fit_excitation(self,position,size,fit_opr=False,**kwargs):
        """Fit the :math:`log N_u-E` diagram with two excitation temperatures,
        a ``hot`` :math:`T_{ex}` and a ``cold`` :math:`T_{ex}`.  A first
        pass guess is initially made using data partitioning and two
        linear fits. 

        If ``position`` and ``size`` are given, the data will be averaged over a spatial box before fitting.  The box is created using :class:`astropy.nddata.utils.Cutout2D`.  If position or size is None, the data are averaged over all pixels.  If the Measurements are single values, these arguments are ignored.

        :param position: The position of the cutout array's center with respect to the data array. The position can be specified either as a `(x, y)` tuple of pixel coordinates or a :class:`~astropy.coordinates.SkyCoord`, which will use the :class:`~astropy.wcs.WCS` of the ::class:`~pdrtpy.measurement.Measurement`s added to this tool. See :class:`~astropy.nddata.utils.Cutout2D`.
        :type position: tuple or :class:`astropy.coordinates.SkyCoord` 
        :param size: The size of the cutout array along each axis. If size is a scalar number or a scalar :class:`~astropy.units.Quantity`, then a square cutout of size will be created. If `size` has two elements, they should be in `(ny, nx)` order. Scalar numbers in size are assumed to be in units of pixels. `size` can also be a :class:`~astropy.units.Quantity` object or contain :class:`~astropy.units.Quantity` objects. Such :class:`~astropy.units.Quantity` objects must be in pixel or angular units. For all cases, size will be converted to an integer number of pixels, rounding the the nearest integer. See the mode keyword for additional details on the final cutout size. Default value of None means use all pixels (position is ignored)
        :type size: int, array_like, or :class:`astropy.units.Quantity`
        :param fit_opr: Whether to fit the ortho-to-para ratio or not. If True, the OPR will be varied to determine the best value. If False, the OPR is fixed at the canonical LTE value of 3.
        :type fit_opr: bool
        :returns: The fit result which contains slopes, intercepts, the ortho to para ratio (OPR), and fit statistics
        :rtype:  :class:`lmfit.model.ModelResult`  
        """

        energy = self.energies(line=True)
        _ee = np.array([c for c in energy.values()])
        _energy = Measurement(_ee,unit="K")
        _ids = list(energy.keys())
        # Get Nu/gu.  Canonical opr will be used. 
        colden = self.average_column_density(norm=True, position=position,
                                             size=size, line=True)

        # Need to stuff the data into a single vector
        _cd = np.array([c.data for c in colden.values()])
        _er  = np.array([c.error for c in colden.values()])
        _colden = Measurement(_cd,uncertainty=StdDevUncertainty(_er),unit="cm-2")

        x = _energy.data
        y = np.log10(_colden.data)
        #kwargs_opts = {"guess": self._first_guess(x,y)}
        #kwargs_opts.update(kwargs)
        sigma = utils.LOGE*_colden.error/_colden.data
        slopecold, intcold, slopehot, inthot = self._first_guess(x,y)
        tcold=(-utils.LOGE/slopecold)
        thot=(-utils.LOGE/slopehot)
        print("First guess at excitation temperatures:\n T_cold = %.1f K\n T_hot = %.1f K"%(tcold,thot))
        
        # update Parameter hints based on first guess.
        self._model.set_param_hint('m1',value=slopecold,vary=True)
        self._model.set_param_hint('n1',value=intcold,vary=True)
        self._model.set_param_hint('m2',value=slopehot,vary=True)
        self._model.set_param_hint('n2',value=inthot,vary=True)
        # update whether opr is allowed to vary or not.
        self._model.set_param_hint('opr',vary = fit_opr)
        p=self._model.make_params()
        #p.pretty_print()
        
        wts = 1/(sigma*sigma)
        idx=self._get_ortho_indices(_ids)
        # Suppress the incorrect warning about model parameters
        warnings.simplefilter('ignore',category=UserWarning)
        self._fitresult = self._model.fit(data=y, weights=wts, x=x, idx=idx,fit_opr=fit_opr)
        warnings.resetwarnings()
        self._compute_quantities(self._fitresult.params)
        print("Fitted excitation temperatures and column densities:")
        print(f" T_cold = {self.tcold.value:3.0f}+/-{self.tcold.error:.1f} {self.tcold.unit}\n T_hot = {self.thot.value:3.0f}+/-{self.thot.error:.1f} {self.thot.unit}")
        print(f" N_cold = {self.cold_colden.value:.2e}+/-{self.cold_colden.error:.1e} {self.cold_colden.unit}\n N_hot = {self.hot_colden.value:.2e}+/-{self.hot_colden.error:.1e} {self.hot_colden.unit}")
        print(f" N_total = {self.total_colden.value:.2e}+/-{self.total_colden.error:.1e} {self.total_colden.unit}")
        
        #if successful, set the used position and size
        self._position = position
        self._size = size
        return self._fitresult

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
    

    def _partition_function(self,tex):
        '''Calculate the H2 partition function given an excitation temperature
        
        :param tex: the excitation temperature
        :type tex: :class:`~pdrtpy.measurement.Measurement` or :class:`astropy.units.quantity.Quantity`
        :returns: the partition function value
        :rtype: numpy.float
        '''
        # See Herbst et al 1996
        # http://articles.adsabs.harvard.edu/pdf/1996AJ....111.2403H
        # Z(T) =  = 0.0247T * [1—exp(—6000/T)]^-1
        
        # This is just being defensive.  I know the temperatures used internally are in K.
        t = (tex.value*u.Unit(tex.unit)).to("K",equivalencies=u.temperature()).value
        z = 0.0247*t/(1.0 - np.exp(-6000.0/t))
        return z

