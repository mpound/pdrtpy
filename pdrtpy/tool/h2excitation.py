from astropy.nddata import Cutout2D
import astropy.units as u
import astropy.constants as constants
from astropy.nddata import StdDevUncertainty
import math
import numpy as np
from lmfit import Parameters#, fit_report
from lmfit.model import Model#, ModelResult
from emcee.pbar import get_progress_bar
import cProfile 
import pstats
import io

from .toolbase import ToolBase
from .fitmap import FitMap
from .. import pdrutils as utils
from ..measurement import Measurement
import warnings

class ExcitationFit(ToolBase):
    """Base class for creating excitation fitting tools for various species.

    :param measurements: Input measurements to be fit.
    :type measurements: list of :class:`~pdrtpy.measurement.Measurement`.
    """
    def __init__(self,measurements=None,constantsfile=None):
        super().__init__()
        # must be set before call to init_measurements
        self._intensity_units = "erg cm^-2 s^-1 sr^-1"
        self._cd_units = 'cm^-2'
        self._t_units = "K"
        self._valid_components = ['hot','cold','total']
        if type(measurements) == dict or measurements is None:
            self._measurements = measurements
        else:
            self._init_measurements(measurements)
        self._set_measurementnaxis()
        if constantsfile is not None:
            # set up atomic constants table, default intensity units
            self._ac = utils.get_table(constantsfile)
            self._ac.add_index("Line")
            self._ac.add_index("Ju")
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
                raise TypeError(f"Measurement {mm.id} units {mm.unit.to_string()} are not in intensity units equivalent to {self._intensity_units}")
            self._measurements[mm.id] = mm

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

    def remove_measurement(self,identifier):
        '''Delete a measurement from the internal dictionary used to compute column densities. Any associated column density will also be removed.

           :param identifier: the measurement identifier
           :type identifier: str
           :raises KeyError: if identifier not in existing Measurements
        '''
        del self._measurements[identifier] # we want this to raise a KeyError if id not found
        self._column_density.pop(identifier,None) # but not this.

    def replace_measurement(self,m):
        '''Safely replace an existing intensity Measurement.  Do not
           change a Measurement in place, use this method.
           Otherwise, the column densities will be inconsistent.

           :param m: A :class:`~pdrtpy.measurement.Measurement` instance containing intensity in units equivalent to :math:`{\\rm erg~cm^{-2}~s^{-1}~sr^{-1}}`
        '''
        self.add_measurement(m)

class H2ExcitationFit(ExcitationFit):
    r"""Tool for fitting temperatures, column densities, and ortho-to-para ratio(`OPR`) from an :math:`H_2` excitation diagram. It takes as input a set of :math:`H_2` rovibrational line observations with errors represented as :class:`~pdrtpy.measurement.Measurement`.

Often, excitation diagrams show evidence of both "hot" and "cold" gas components, where the cold gas dominates the intensity in the low `J` transitions and the hot gas dominates in the high `J` transitions. Given data over several transitions, one can fit for :math:`T_{cold}, T_{hot}, N_{total} = N_{cold}+ N_{hot}`, and optionally `OPR`. One needs at least 5 points to fit the temperatures and column densities (slope and intercept :math:`\times 2`), though one could compute (not fit) them with only 4 points. To additionally fit `OPR`, one should have 6 points (5 degrees of freedom).

Once the fit is done, :class:`~pdrtpy.plot.ExcitationPlot` can be used to view the results.

:param measurements: Input :math:`H_2` measurements to be fit.
:type measurements: list of :class:`~pdrtpy.measurement.Measurement`.
    """
    def __init__(self,measurements=None,
                 constantsfile="RoueffEtAl.tab"):
        super().__init__(measurements,constantsfile)
        self._canonical_opr = 3.0
        self._opr = Measurement(data=[self._canonical_opr],uncertainty=None)
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
        self._params.add('m1',value=0,min=-1,max=0)
        self._params.add('n1',value=15,min=10,max=30)
        self._params.add('m2',value=0,min=-1,max=0)
        self._params.add('n2',value=15,min=10,max=30)

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

    def _compute_quantities(self,fitmap):
        """Compute the temperatures and column densities for the hot and cold gas components.  This method will set class variables `_temperature` and `_colden`.

        :param params: The fit parameters returned from fit_excitation.
        :type params: :class:`lmfit.Parameters`
        """
        self._temperature = dict()
        # N(J=0) column density = intercept on y axis
        self._j0_colden = dict()
        # total column density = N(J=0)*Z(T) where Z(T) is partition function
        self._total_colden = dict()
        size = fitmap.data.size
        # create default arrays in which calculated values will be stored.
        # Use nan as fill value because there may be nans in fitmapdata, in which
        # case nothing need be done to arrays.
        # tc, th = cold and hot temperatures
        # utc, utc = uncertainties in cold and hot temperatures
        # nc, nh = cold and hot column densities
        # unc, unh = uncertainties in cold and hot temperatures
        # opr = ortho to para ratio
        # uopr = uncertainty in OPR
        tc = np.full(shape=size,fill_value=np.nan,dtype=float)
        th = np.full(shape=size,fill_value=np.nan,dtype=float)
        utc = np.full(shape=size,fill_value=np.nan,dtype=float)
        uth = np.full(shape=size,fill_value=np.nan,dtype=float)
        nc = np.full(shape=size,fill_value=np.nan,dtype=float)
        nh = np.full(shape=size,fill_value=np.nan,dtype=float)
        unh = np.full(shape=size,fill_value=np.nan,dtype=float)
        unc = np.full(shape=size,fill_value=np.nan,dtype=float)
        opr = np.full(shape=size,fill_value=np.nan,dtype=float)
        uopr = np.full(shape=size,fill_value=np.nan,dtype=float)
        ff = fitmap.data.flatten()
        ffmask = fitmap.mask.flatten()
        for i in range(size):
            if ffmask[i]:
                continue
            params = ff[i].params
            for p in params:
                if params[p].stderr is None:
                    print("AT pixel i [mask]",i,ffmask[i])
                    params.pretty_print()
                    raise Exception("Something went wrong with the fit and it was unable to calculate errors on the fitted parameters. It's likely that a two-temperature model is not appropriate for your data. Check the fit_result report and plot.")


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
            utc[i] = params[mcold].stderr/params[mcold]
            tc[i] = -utils.LOGE/params[mcold]
            uth[i] = params[mhot].stderr/params[mhot]
            th[i] = -utils.LOGE/params[mhot]
            nc[i] = 10**params[ncold]
            unc[i] = utils.LN10*params[ncold].stderr*nc[i]
            nh[i] = 10**params[nhot]
            unh[i] = utils.LN10*params[nhot].stderr*nh[i]
            opr[i] = params['opr'].value
            uopr[i] = params['opr'].stderr

        # now reshape them all back to map shape
        tc = tc.reshape(fitmap.data.shape)
        th = th.reshape(fitmap.data.shape)
        utc = utc.reshape(fitmap.data.shape)
        uth = uth.reshape(fitmap.data.shape)
        nc = nc.reshape(fitmap.data.shape)
        nh = nh.reshape(fitmap.data.shape)
        unh = unh.reshape(fitmap.data.shape)
        unc = unc.reshape(fitmap.data.shape)
        opr = opr.reshape(fitmap.data.shape)
        uopr = uopr.reshape(fitmap.data.shape)

        mask = fitmap.mask | np.logical_not(np.isfinite(tc))
        ucc= StdDevUncertainty(np.abs(tc*utc))
        self._temperature["cold"]=Measurement(data=tc,unit=self._t_units,
                                              uncertainty=ucc,wcs=fitmap.wcs, mask = mask)
        mask = fitmap.mask | np.logical_not(np.isfinite(th))
        uch = StdDevUncertainty(np.abs(th*uth))
        self._temperature["hot"]=Measurement(data=th,unit=self._t_units,
                                             uncertainty=uch,wcs=fitmap.wcs, mask = mask)
        # cold and hot total column density
        ucn = StdDevUncertainty(np.abs(unc))
        mask = fitmap.mask | np.logical_not(np.isfinite(nc))
        self._j0_colden["cold"] = Measurement(nc,unit=self._cd_units,
                                              uncertainty=ucn,wcs=fitmap.wcs, mask=mask)
        mask = fitmap.mask | np.logical_not(np.isfinite(nh))
        uhn = StdDevUncertainty(np.abs(unh))
        self._j0_colden["hot"] = Measurement(nh,unit=self._cd_units,
                                             uncertainty=uhn,wcs=fitmap.wcs, mask = mask)
        #
        self._total_colden["cold"] = self._j0_colden["cold"] * self._partition_function(self.tcold)
        self._total_colden["hot"] = self._j0_colden["hot"] * self._partition_function(self.thot)
        mask = fitmap.mask | np.logical_not(np.isfinite(opr))
        self._opr = Measurement(opr, unit=u.dimensionless_unscaled,
                                uncertainty=StdDevUncertainty(uopr),wcs=fitmap.wcs, mask=mask)

    @property
    def fit_result(self):
        '''The result of the fitting procedure which includes fit statistics, variable values and uncertainties, and correlations between variables.

        :rtype:  :class:`lmfit.model.ModelResult`
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
        return self._params['opr'].vary

    @property
    def opr(self):
        '''The ortho-to-para ratio (OPR)

        :returns: The fitted OPR is it was determined in the fit, otherwise the canonical LTE OPR
        :rtype: :class:`~pdrtpy.measurement.Measurement`
        '''
        return self._opr

    @property
    def intensities(self):
        '''The stored intensities. See :meth:`add_measurement`

           :rtype: list of :class:`~pdrtpy.measurement.Measurement`
        '''
        return self._measurements

    def colden(self,component):#,log=False):
        '''The column density of hot or cold gas component, or total column density.

        :param component: 'hot', 'cold', or 'total
        :type component: str

        :rtype: :class:`~pdrtpy.measurement.Measurement`
        '''
        #:param log: take the log10 of the column density
        cl = component.lower()
        if cl not in self._valid_components:
            raise KeyError(f"{cl} not a valid component. Must be one of {self._valid_components}")
        #print(f'returning {cl}')
        if cl == 'total':
            return self.total_colden
        else:
            return self._total_colden[cl]

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

    @property
    def temperature(self):
        '''The fitted gas temperatures, returned in a dictionary with keys 'hot' and 'cold'.
        :rtype: dict
        '''
        return self._temperature

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
                    denom = self._ac.loc[cd]["gu"]
                else:
                    denom = self._ac.loc['Ju',cd]["gu"]
                    if(len(denom)>0): 
                        denom=denom[0] #ARGH kluge.  Need to get rid of line=False option as Ju is no longer unique
                #print("CD ",cd,"DENOM ",denom)
                # This fails with complaints about units:
                #self._column_density[cd] /= self._ac.loc[cd]["gu"]
                #gu = Measurement(self._ac.loc[cd]["gu"],unit=u.dimensionless_unscaled)
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
                t[m] = self._ac.loc[m]["Tu"]
        else:
            for m in self._measurements:
                t[self._ac.loc[m]["Ju"]] = self._ac.loc[m]["Tu"]
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
        '''
        kwargs_opts = { 'mask': None,
                        'method': 'leastsq',
                        'nan_policy': 'raise',
                        'test':False,
                        'profile':False,
                      }
        kwargs_opts.update(kwargs)
        return self._fit_excitation(position,size,fit_opr,**kwargs_opts)

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
        dE = self._ac.loc[colden.id]["dE"]*constants.k_B.cgs*self._ac["dE"].unit
        A = self._ac.loc[colden.id]["A"]*self._ac["A"].unit
        v = A*dE/(4.0*math.pi*u.sr)
        val = Measurement(data=v.value,unit=v.unit,identifier=colden.id)
        intensity = val*colden # error will get propagated
        i = intensity.convert_unit_to(self._intensity_units)
        i._identifier = val.id
        return i

    def upper_colden(self,intensity,unit):
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

        dE = self._ac.loc[intensity.id]["dE"] * constants.k_B.cgs * self._ac["dE"].unit
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
                index = self._ac.loc[m]["Ju"]
            self._column_density[index] = self.upper_colden(self._measurements[m],unit)

    def gu(self,id,opr):
        r'''Get the upper state statistical weight $g_u$ for the given transition identifer, and, if the transition is odd-$J$, scale the result by the given ortho-to-para ratio.  If the transition is even-$J$, the LTE value is returned.

           :param id: the measurement identifier
           :type id: str
           :param opr:
           :type opr: float
           :raises KeyError: if id not in existing Measurements
           :rtype: float
        '''
        if utils.is_even(self._ac.loc[id]["Ju"]):
            return self._ac.loc[id]["gu"]
        else:
            #print("Ju=%d scaling by [%.2f/%.2f]=%.2f"%(self._ac.loc[id]["Ju"],opr,self._canonical_opr,opr/self._canonical_opr))
            return self._ac.loc[id]["gu"]*opr/self._canonical_opr

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
        return np.where(self._ac.loc[ids]["Ju"]%2!=0)[0]

    def _get_para_indices(self,ids):
        """Given a list of J values, return the indices of those that are para
        transitions (even J)

        :param ids:
        :type ids: list of str
        :returns: The array indices of the even J values.
        :rtype: list of int
        """
        return np.where(self._ac.loc[ids]["Ju"]%2==0)[0]

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
        #print("FG ",type(slopecold),type(slopehot),type(intcold),type(inthot))
        return np.array([slopecold, intcold, slopehot, inthot])

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
     ,dtype=object   :returns: The fit result which contains slopes, intercepts, the ortho to para ratio (OPR), and fit statistics
        :rtype:  :class:`lmfit.model.ModelResult`
        """
        profile = kwargs.pop('profile')
        self._stats = None
        if profile:
            pr = cProfile.Profile()
            pr.enable()
        if fit_opr:
            min_points = 5
        else:
            min_points = 4
            self._opr = Measurement(data=[self._canonical_opr],uncertainty=None)

        self._params['opr'].vary = fit_opr
        energy = self.energies(line=True)
        _ee = np.array([c for c in energy.values()])
        #@ todo: allow fitting of one-temperature model
        if len(_ee) < min_points:
            raise Exception(f"You need at least {min_points:d} data points to determine two-temperature model")
        if len(_ee) == min_points:
            warnings.warn(f"Number of data points is equal to number of free parameters ({min_points:d}). Fit will be over-constrained")
        _energy = Measurement(_ee,unit="K")
        _ids = list(energy.keys())
        idx=self._get_ortho_indices(_ids)
        # Get Nu/gu.  Canonical opr will be used.
        if position is None or size is None:
            colden = self.column_densities(norm=True,line=True)
        else:

            colden = self.average_column_density(norm=True, position=position,
                                             size=size, line=True)

        # Need to stuff the data into a single vector
        _cd = np.squeeze(np.array([c.data for c in colden.values()]))
        _er = np.squeeze(np.array([c.error for c in colden.values()]))
        _colden = Measurement(_cd,uncertainty=StdDevUncertainty(_er),unit="cm-2")
        fk = utils.firstkey(colden)
        x = _energy.data
        y = np.log10(_colden.data)
        #print("SHAPE Y LEN(SHAPE(Y) ",y.shape,len(y.shape))
        #kwargs_opts = {"guess": self._first_guess(x,y)}
        #kwargs_opts.update(kwargs)
        sigma = utils.LOGE*_colden.error/_colden.data
        slopecold, intcold, slopehot, inthot = self._first_guess(x,y)
        #print(slopecold,slopehot,intcold,inthot)
        tcold=(-utils.LOGE/slopecold)
        thot=(-utils.LOGE/slopehot)
        if np.shape(tcold) == ():
            tcold = np.array([tcold])
            thot = np.array([thot])
        saveshape = tcold.shape
        #print("TYPE COLD SIT",type(slopecold),type(intcold),type(tcold))
        #print("SHAPES: colden/sigma/slope/int/temp/cd: ",np.shape(_colden),np.shape(sigma),np.shape(slopecold),np.shape(intcold),np.shape(tcold),np.shape(_cd))
        #print("First guess at excitation temperatures:\n T_cold = %.1f K\n T_hot = %.1f K"%(tcold,thot))
        fmdata = np.empty(tcold.shape,dtype=object).flatten()
        #fm = FitMap(data=fmdata,wcs=colden[fk].wcs,uncertainty=None,unit=None)
        tcold = tcold.flatten()
        thot = thot.flatten()
        slopecold = slopecold.flatten()
        slopehot = slopehot.flatten()
        inthot = inthot.flatten()
        intcold = intcold.flatten()
        #sigma = sigma.flatten()
        # flatten any dimensions past 0
        shp = y.shape
        #print("NS ",shp[0],shp[1:])
        if len(shp) == 1:
            #print("adding new axis")
            y = y[:,np.newaxis]
            shp = y.shape
        yr = y.reshape((shp[0],np.prod(shp[1:])))
        sig = sigma.reshape((shp[0],np.prod(shp[1:])))
        #print("YR, SIG SHAPE",yr.shape,sig.shape)
        count = 0
        #print("LEN(TCOLD)",len(tcold))
        total = len(tcold)
        fm_mask = np.full(shape=tcold.shape,fill_value=False)
        # Suppress the incorrect warning about model parameters
        warnings.simplefilter('ignore',category=UserWarning)
        excount = 0
        badfit = 0
        # update whether opr is allowed to vary or not.
        self._model.set_param_hint('opr',vary = fit_opr)
        # use progress bar if more than one pixel
        if total > 1:
            progress = kwargs.pop("progress",True)
        else:
            progress = False
        with get_progress_bar(progress,total,leave=True,position=0) as pbar:
            for i in range(total):
                if np.isfinite(yr[:,i]).all() and np.isfinite(sig[:,i]).all():
                    # update Parameter hints based on first guess.
                    self._model.set_param_hint('m1',value=slopecold[i],vary=True)
                    self._model.set_param_hint('n1',value=intcold[i],vary=True)
                    self._model.set_param_hint('m2',value=slopehot[i],vary=True)
                    self._model.set_param_hint('n2',value=inthot[i],vary=True)
                    p=self._model.make_params()
                    wts = 1.0/(sig[:,i]*sig[:,i])
                    try:
                        fmdata[i] = self._model.fit(data=yr[:,i], weights=wts, x=x,
                                                      idx=idx,fit_opr=fit_opr,method=kwargs['method'],
                                                      nan_policy = kwargs['nan_policy'])
                        if fmdata[i].success and fmdata[i].errorbars:
                            count = count+1
                        else:
                            fmdata[i] = None
                            fm_mask[i] = True
                            badfit = badfit + 1
                    except ValueError:
                        fmdata[i] = None
                        fm_mask[i] = True
                        excount = excount+1
                else:
                    fmdata[i] = None
                    fm_mask[i] = True
                pbar.update(1)
        warnings.resetwarnings()
        fmdata = fmdata.reshape(saveshape)
        fm_mask = fm_mask.reshape(saveshape)
        self._fitresult = FitMap(fmdata,wcs=colden[fk].wcs,mask=fm_mask,name="result")
        # this will raise an exception if the fit was bad (fit errors == None)
        self._compute_quantities(self._fitresult)
        print(f"fitted {count} of {slopecold.size} pixels")
        print(f'got {excount} exceptions and {badfit} bad fits')
        #if successful, set the used position and size
        self._position = position
        self._size = size
        if profile:
            pr.disable()
            s = io.StringIO()
            sortby = pstats.SortKey.CUMULATIVE
            ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
            ps.print_stats()
            self._stats = s

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
        :rtype: numpy.ndarray
        '''
        # See Herbst et al 1996
        # http://articles.adsabs.harvard.edu/pdf/1996AJ....111.2403H
        # Z(T) =  = 0.0247T * [1—exp(—6000/T)]^-1

        # This is just being defensive.  I know the temperatures used internally are in K.
        t = np.ma.masked_invalid((tex.value*u.Unit(tex.unit)).to("K",equivalencies=u.temperature()).value)
        t.mask = np.logical_or(t.mask,np.logical_not(np.isfinite(t)))
        z = 0.0247*t/(1.0 - np.exp(-6000.0/t))
        return z
