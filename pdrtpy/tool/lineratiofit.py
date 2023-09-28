from copy import deepcopy

import numpy as np
import numpy.ma as ma

#from astropy.io import fits
from astropy.io.fits.header import Header
#import astropy.wcs as wcs
import astropy.units as u
import astropy.stats as astats
from astropy.table import Table, Column
from astropy.nddata import CCDData
import warnings
from lmfit import Parameters, Minimizer#, fit_report
from scipy.interpolate import interpn
from pdrtpy.pbar import get_progress_bar

import cProfile
import pstats 
import io

from .toolbase import ToolBase
from .fitmap import FitMap
from .. import pdrutils as utils
from ..modelset import ModelSet
#from ..measurement import Measurement

class LineRatioFit(ToolBase):
    """LineRatioFit is a tool to fit observations of intensity ratios to a set of PDR models. It takes as input a set of observations with errors represented as :class:`~pdrtpy.measurement.Measurement` and  :class:`~pdrtpy.modelset.ModelSet` for the models to which the data will be fitted. The observations should be spectral line or continuum intensities.  They can be spatial maps or single pixel values. They should have the same spatial resolution.

The models to be fit are stored as intensity ratios. The input observations will be use to create ratios that correspond to models. From there a minimization fit is done to determine the density and radiation field that best fit the data. At least 3 observations are needed in order to make at least 2 ratios. With fewer ratios, no fitting can be done. More ratios generally means better determined density and radiation field, assuming the data are consistent with each other.

Once the fit is done, :class:`~pdrtpy.plot.LineRatioPlot` can be used to view the results.


:param modelset: The set of PDR models to use for fitting.
:type modelset: :class:`~pdrtpy.modelset.ModelSet`

:param measurements: Input measurements to be fit.
:type measurements: list or dict of :class:`~pdrtpy.measurement.Measurement`. If dict, the keys should be the Measurement *identifiers*.
    """
    def __init__(self,modelset=ModelSet("wk2006",z=1),measurements=None):
        super().__init__() # needed?
        if type(modelset) == str:
            # may need to disable this
            self._initialize_modelTable(modelset)
        self._modelset = modelset
        self._init_measurements(measurements)
        self._set_measurementnaxis()
        self._modelratios = None
        self._modelnaxis = None
        self._set_model_files_used()
        self._observedratios = None
        self._chisq = None
        self._reduced_chisq = None
        self._likelihood = None
        self._radiation_field = None
        self._density = None
        self.radiation_field_unit = None
        self.radiation_field_type = None
        self.density_unit = None
        self.density_type = None
        self._fitresult = None
        self._fitparam = None
        self._minimizer = None
        self._deltasq = None
        self._ratiocount = None

    @property
    def fit_result(self):
        '''The result of the fitting procedure which includes fit statistics, variable values and uncertainties, and correlations between variables.  For each pixel, there will be one instance of :class:`lmfit.minimizer.MinimizerResult`.

        :rtype:  :class:`~pdrtpy.tool.FitMap`
        '''
        return self._fitresult

    @property
    def modelset(self):
        """The underlying :class:`~pdrtpy.modelset.ModelSet`"""
        return self._modelset

    @property
    def measurements(self):
        """The stored :class:`measurements <pdrtpy.measurement.Measurement>` as a dictionary with Measurement IDs as keys

        :rtype: dict of :class:`~pdrtpy.measurement.Measurement`
        """
        return self._measurements

    @property
    def measurementIDs(self):
        '''The stored measurement IDs, which are strings.

        :rtype: :class:`dict_keys`
        '''

        if self._measurements is None: 
            return None
        return self._measurements.keys()

    @property
    def observed_ratios(self):
        '''The list of the observed line ratios that have been input so far.

        :rtype: list of str
        '''
        return list(self._observedratios.keys())

    @property
    def ratiocount(self):
        '''The number of ratios that match models available in the current :class:`~pdrtpy.modelset.ModelSet` given the current set of measurements

        :rtype: int
        '''
        # call to  modelset._get_ratio_MA 01938elements is expensive. So set it once for each run.
        if self._ratiocount is None:
            self._ratiocount = self._modelset.ratiocount(self.measurementIDs)
        return self._ratiocount

    @property
    def density(self):
        '''The computed hydrogen nucleus density value(s).

        :rtype: :class:`~pdrtpy.measurement.Measurement`
        '''
        return self._density

    @property
    def radiation_field(self):
        '''The computed radiation field value(s).

        :rtype: :class:`~pdrtpy.measurement.Measurement`
        '''
        return self._radiation_field

    def chisq(self,min=False):
        '''The computed chisquare value(s).

        :type min: bool
        :param min: If `True` return the minimum reduced :math:`\chi^2`.  In the case of map inputs this will be a spatial map of mininum :math:`\chi^2`.  If `False` with map inputs the entire :math:`\chi^2` hypercube is returned.  If `True` with single pixel inputs, a single value is returned.  If `False` with single pixel inputs, :math:`\chi^2` as a function of density and radiation field is returned.

        :rtype: :class:`~pdrtpy.measurement.Measurement`
        '''
        if min:
            return self._chisq_min
        else:
            return self._chisq

    def reduced_chisq(self,min=False):
        r'''The computed reduced chisquare value(s).

        :type min: bool
        :param min: If `True` return the minimum reduced :math:`\chi_\nu^2`.  In the case of map inputs this will be a spatial map of mininum :math:`\chi_\nu^2`.  If `False` with map inputs the entire :math:`\chi_\nu^2` hypercube is returned.  If `True` with single pixel inputs, a single value is returned.  If `False` with single pixel inputs, :math:`\chi_\nu^2` as a function of density and radiation field is returned.

        :rtype: :class:`~pdrtpy.measurement.Measurement`
        '''
        if min:
            return self._reduced_chisq_min
        else:
            return self._reduced_chisq

    def _init_measurements(self,m):
        """Initialize the measurements from an input list or dict. If a dict, the dictionary keys must be valid measurement identifiers.

        :param m: the input list of Measurements
        :type m: list, tuple, or dict
        """
        self._masks = dict() # need to save these so they can be reset later
        if m is None:
            self._measurements = None
        elif type(m) == list or type(m) == tuple:
            self._measurements = dict()
            for mm in m:
                self._measurements[mm.id] = mm
                self._masks[mm.id] = deepcopy(mm.mask)
        elif type(m) == dict:
            self._measurements = deepcopy(m)
            for key in m:
                self._masks[key] = deepcopy(m[key].mask)
        else:
            raise ValueError("Input measurements must be list, tuple, or dict")

    def _set_model_files_used(self):
        self._model_files_used = dict()
        if self._measurements is None: 
            return
        for x in self._modelset.find_files(self.measurementIDs):
            self._model_files_used[x[0]]=x[1]

    def _check_shapes(self,d):
       #ugly
       s1 = d[utils.firstkey(d)].shape
       return np.all([m.shape == s1 for m in d.values()])

    def _check_header(self,kw,value=NotImplemented):
       """Check to see if any of the given keyword values differ for
          the input measurements.

          :param kw: the keyword to check
          :type kw: str
          :param value: If given and not NotImplemented, then check that all *kw* values of the input measurements requal this value.  The default is the special value NotImplemented rather than None which allows us to check against None if needed.
          :type kw: any
       """
       d = self._measurements
       fk = utils.firstkey(d)
       in_wcs = False
       try:
           if value == NotImplemented:
               h  =  d[fk].header
               if kw not in h:
                  #CTYPES etc can also be in WCS so check that too
                  s1 =  d[fk].wcs.to_header()[kw]
                  in_wcs = True
               else:
                  s1 = h[kw]
           else:
               s1 = value
           if in_wcs:
               return np.all([m.wcs.to_header()[kw] == s1 for m in d.values()])
           else:
               return np.all([m.header[kw] == s1 for m in d.values()])
       except KeyError:
           print("WARNING: %s keyword not present in all Measurements"%kw)
           return False

    def _check_measurement_shapes(self):
       if self._measurements is None: 
           return False
       return self._check_shapes(self._measurements)

    def _check_ratio_shapes(self):
       if self._observedratios is None: 
           return False
       return self._check_shapes(self._observedratios)

    def _check_model_shapes(self):
       if self._modelratios is None: 
           return False
       return self._check_shapes(self._modelratios)

    def add_measurement(self,m):
        r'''Add a Measurement to internal dictionary used to compute ratios. This measurement may be intensity units (erg :math:`{\rm s}^{-1}` :math:`{\rm cm}^{-2}`) or integrated intensity (K km/s).

           :param m: a Measurement instance to be added to this tool
           :type m: :class:`~pdrtpy.measurement.Measurement`.

        '''
        if self._measurements:
            self._measurements[m.id] = m
        else:
            self._init_measurements(m)
        self._set_model_files_used()

    def remove_measurement(self,id):
        '''Delete a measurement from the internal dictionary used to compute ratios.

           :param id: the measurement identifier
           :type id: str
           :raises KeyError: if id not in existing Measurements
        '''
        del self._measurements[id]
        self._set_model_files_used()

    def read_models(self,unit=u.dimensionless_unscaled):
        """Given a list of measurement IDs, find and open the FITS files that have matching ratios
        and populate the _modelratios dictionary.  Uses :class:`pdrtpy.measurement.Measurement` as
        a storage mechanism.

           :param  m: list of measurement IDS (string)
           :type m: list
           :param unit: units of the data
           :type unit: string or astropy.Unit
        """
        self._modelratios = self._modelset.get_models(self.measurementIDs,model_type='ratio')
        if self.ratiocount < 2:
            msg = f"Not enough ratios.  You need to provide at least 3 observations that can be used to compute 2 ratios that are covered by the ModelSet. From your observations, {self.ratiocount:d} ratio(s)"
            if self.ratiocount>0:
                msg += f" {list(self._modelratios.keys())}"
            msg += " can be computed."
            raise Exception(msg)

        k = utils.firstkey(self._modelratios)
        self._modelnaxis = self._modelratios[k].wcs.naxis
        self._modelshape = np.array(self._modelratios[k].data.shape)
        if not self.density_unit:
            self.density_unit = self._modelratios[k].wcs.wcs.cunit[0]
            self.density_type = self._modelratios[k].wcs.wcs.ctype[0]
        if not self.radiation_field_unit:
            self.radiation_field_unit = self._modelratios[k].wcs.wcs.cunit[1]
            self.radiation_field_type = self._modelratios[k].wcs.wcs.ctype[1]
            # for wk2006 models, Habing units in the Y axis cause problems when trying to assign them
            # to a WCS (see note in modelset.py).  So get from header instead.
            if self.radiation_field_unit.to_string() == '':
                try:
                    self.radiation_field_unit = u.Unit(self._modelratios[k].header["CUNIT2"])
                except KeyError:
                    raise Exception("Keyword CUNIT2 is required in file %s FITS header to describe units of interstellar radiation field"%self._model_files_used[k])
        if not self._check_model_shapes():
            warnings.warn("Trimming all model grids to match H2 grid: log(n) = 1-5, log(G0) = 1-5")
            utils._trim_all_to_H2(self._modelratios)

    def _check_compatibility(self):
        """Check that all Measurements are compatible (beams, coordinate systems, shapes) so that the computation make commence.

          :raises Exception: if headers and shapes don't match, warns if no beam present
        """

        if not self._check_measurement_shapes():
           raise Exception("Your input Measurements have different dimensions")

        # Check the beam sizes
        # @Todo do the convolution ourselves if requested.
        if not self._check_header("BMAJ"):
           raise Exception("Beam major axis (BMAJ) of your input Measurements do not match.  Please convolve all maps to the same beam size")
        if not self._check_header("BMIN"):
           raise Exception("Beam minor axis (BMIN) of your input Measurements do not match.  Please convolve all maps to the same beam size")
        if not self._check_header("BPA"):
           raise Exception("Beam position angle (BPA) of your input Measurements do not match.  Please convolve all maps to the same beam size")


        # Check the coordinate systems only if there is more than one pixel
        m1 = self._measurements[utils.firstkey(self._measurements)]
        if utils.is_image(m1):
            if not self._check_header("CTYPE1"):
               raise Exception("CTYPE1 of your input Measurements do not match. Please ensure coordinates of all Measurements are the same.")
            if not self._check_header("CTYPE2"):
               raise Exception("CTYPE2 of your input Measurements do not match. Please ensure coordinates of all Measurements are the same.")

        #Only allow beam = None if single value measurements.
        if not utils.is_image(m1):
            if self._check_header("BMAJ",None) or self._check_header("BMIN",None) or self._check_header("BPA",None):
               utils.warn(self,"No beam parameters in Measurement headers, assuming they are all equal!")
        #if not self._check_header("BUNIT") ...

    def run(self,**kwargs):
        '''Run the full computation using all the :class:`observations <pdrtpy.measurement.Measurement>` added.   This will
        check compatibility of input observations (e.g., beam parameters, coordinate types, axes lengths) and
        raise exceptions if the observations don't match each other.

           :param mask: Indicate how to mask image observations (Measurements) before computing the density
                        and radiation field. Possible values are:

             ['mad', multiplier]   - compute standard deviation using median absolute deviation (astropy.mad_std),
                                     and mask out values between +/- multiplier*mad_std.  Example: ['mad',1.0]

             ['data', (low,high)]  - mask based on data values, mask out data between low and high

             ['clip', (low,high)]  - mask based on data values, mask out data below low and above high

             ['error', (low,high)] - mask based on uncertainty plane, mask out data where the corresponding error pixel value
                                     is below low or above high

                              None - Don't mask data. This is the default.

           :type mask:  list or None
           :param method: the fitting method to be used. The default is 'leastsq', which is Levenberg-Marquardt least squares.  For other options see https://lmfit-py.readthedocs.io/en/latest/fitting.html#fit-methods-table
           :type method: str
           :param nan_policy: Specifies action if fit returns NaN values. One of:
                * ’raise’ : a ValueError is raised [Default]
                * ’propagate’ : the values returned from userfcn are un-altered
                * ’omit’ : non-finite values are filtered
           :type nan_policy: str

           :raises Exception: if no models match the input observations, observations are not compatible,
                              or on unrecognized parameters, or NaN encountered.
        '''
        #@todo global masking for 'data', 'clip', 'error' not entirely useful unless all data/error have same ranges.
        #need something like ['data',['key1':(low,hi), 'key2',(low,hi),...], which is very complicated.
        # or data/error cut based on histogram
        kwargs_opts = { 'mask': None,
                        'method': 'leastsq',
                        'nan_policy': 'raise',
                        'refine':True,
                       # for emcee
                        'burn': 0,
                        'steps': 1000,
                       # debugging
                        'test': False,
                        'profile': False,
        }
        kwargs_opts.update(kwargs)

        profile = kwargs_opts.pop('profile')
        self._stats = None
        if profile:
            pr = cProfile.Profile()
            pr.enable()
        self._check_compatibility()
        self._ratiocount = None
        self.read_models()
        self._reset_masks()
        self._mask_measurements(kwargs_opts['mask'])
        kwargs_opts.pop('mask')
        self._compute_valid_ratios()
        if self.ratiocount == 0 :
            raise Exception("No models were found that match your data. Check ModelSet.supported_ratios.")


        # eventually need to check that the maps overlap in real space.
        self._compute_residual()
        self._minimizer= Minimizer(self._residual_single_pixel,
                                   params=None, nan_policy=kwargs_opts['nan_policy'])
        #need to pop nan_policy and test so that it does not get passed to Minimzer.minimize()
        kwargs_opts.pop('nan_policy',None)
        kwargs_opts.pop('test',None)
        self._compute_chisq()
        self._coarse_density_radiation_field()
        if kwargs_opts['refine']:
            kwargs_opts.pop('refine')
            self._refine_density_radiation_field2(**kwargs_opts)
        if profile:
            pr.disable()
            s = io.StringIO()
            sortby = pstats.SortKey.CUMULATIVE
            ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
            ps.print_stats()
            self._stats = s

    def _reset_masks(self):
        for m in self._measurements:
            self._measurements[m].mask = deepcopy(self._masks[m])

    def _mask_measurements(self,mask):
        ''' Set the mask on the measurements based on noise characteristics.  This is so that
            we don't compute garbage n,G0 where observed ratios are noise divided by noise.

           :param mask: Indicate how to mask image observations before computing the density and radiation field. See run()>
        '''
        if mask is None:
            return
        if self._measurementnaxis == 0:
            utils.warn(self,"Ignoring 'mask' parameter for single pixel observations")
            return
        if mask[0] == 'mad':
            for k,v in self._measurements.items():
                sigcut = mask[1]*astats.mad_std(v.data,ignore_nan=True)
                print("Masking %s data between [%.1e,%.1e]"%(k,-sigcut,sigcut))
                masked_data = ma.masked_inside(v.data,-sigcut,sigcut,copy=True)
                # CCDData/NDData do not use MaskArrays underneath but two nddata.arrays. Why??
                # Make a copy so we are keeping references to data copies lying around.
                v.mask = masked_data.mask.copy()
        elif mask[0] == 'data':
            for k,v in self._measurements.items():
                masked_data=ma.masked_inside(v.data,mask[1][0],mask[1][1],copy=True)
                v.mask = masked_data.mask.copy()
                print("Masking %s data between [%.1e,%.1e]"%(k,mask[1][0],mask[1][1]))
        elif mask[0] == 'clip':
            for k,v in self._measurements.items():
                masked_data=ma.masked_outside(v.data,mask[1][0],mask[1][1],copy=True)
                v.mask = masked_data.mask.copy()
                print("Masking %s data outside [%.1e,%.1e]"%(k,mask[1][0],mask[1][1]))
        elif mask[0] == 'error':
            for k,v in self._measurements.items():
                # error is StdDevUncertainty so must use _array to get at raw values
                indices = np.where((v.error <= mask[1][0]) | (v.error >= mask[1][1]))
                if v.mask is not None:
                    v.mask[indices] = True
                else:
                    v.mask = np.full(v.data.shape,False)
                    v.mask[indices] = True
                print("Masking %s data where error outside [%.1e,%.1e]"%(k,mask[1][0],mask[1][1]))
        else:
            raise ValueError("Unrecognized mask parameter %s. Valid values are 'mad','data','error'"%mask[0])


    def _compute_valid_ratios(self):
        '''Compute the valid observed ratio maps for the available model data'''
        if not self._check_measurement_shapes():
            raise Exception("Measurement maps have different dimensions")

        # Note _find_ratio_elements does not handle case of OI+CII/FIR so
        # we have to deal with that separately below.
        z = self._modelset._find_ratio_elements(self.measurementIDs)
        self._observedratios = dict()
        for p in z:
            label = p["numerator"]+"/"+p["denominator"]
            # deepcopy workaround for bug: https://github.com/astropy/astropy/issues/9006
            num = utils.convert_if_necessary(self._measurements[p["numerator"]])
            denom = utils.convert_if_necessary(self._measurements[p["denominator"]])
            self._observedratios[label] = deepcopy(num/denom)
            #@TODO create a meaningful header for the ratio map
            self._ratioHeader(p["numerator"],p["denominator"],label)
            self._observedshape = self._observedratios[label].data.shape
        self._observedshape = np.array(self._observedshape)
        self._add_oi_cii_fir()


    def _add_oi_cii_fir(self):
        '''add special case ([O I] 63 micron + [C II] 158 micron)/IFIR to observed ratios'''
        m = self.measurementIDs
        if "CII_158" in m and "FIR" in m:
            if "OI_63" in m:
                lab="OI_63+CII_158/FIR"
                oi = utils.convert_if_necessary(self._measurements["OI_63"])
                cii = utils.convert_if_necessary(self._measurements["CII_158"])
                a = deepcopy(oi+cii)
                b = deepcopy(self._measurements["FIR"])
                self._observedratios[lab] = a/b
                self._observedratios[lab].meta = deepcopy(b.header)
                self._ratioHeader("OI_63+CII_158","FIR",lab)
            if "OI_145" in m:
                lab="OI_145+CII_158/FIR"
                oi = utils.convert_if_necessary(self._measurements["OI_145"])
                cii = utils.convert_if_necessary(self._measurements["CII_158"])
                aa = deepcopy(oi+cii)
                bb = deepcopy(self._measurements["FIR"])
                self._observedratios[lab] = aa/bb
                self._observedratios[lab].meta = deepcopy(bb.header)
                self._ratioHeader("OI_145+CII_158","FIR",lab)

    # function to minimize in single-pixel case
    def _residual_single_pixel(self,params,index):
        parvals = params.valuesdict()
        mvalue = np.empty(self.ratiocount)
        dvalue = np.empty(self.ratiocount)
        evalue = np.empty(self.ratiocount)
        i=0
        #@todo create temporary model, data, error as np arrays so the loop over k dictionary key
        # isn't necessary.
        # OR save residuals from computeDelta so you don't do this twice.
        for k in self._modelratios:
            mvalue[i] = self._modelratios[k].get(parvals['density'],parvals['radiation_field'])
            dvalue[i] = self._observedratios[k].data.flatten()[index]
            evalue[i] = self._observedratios[k].uncertainty.array.flatten()[index]
            i = i+1
        return  (dvalue - mvalue)/evalue

    def _residual_multi_pixel(self,params,index):
        # this is currently slower than the 'dumb' way of residual_single_pixel!
        parvals = params.valuesdict()
        return self._interp_resid(parvals['density'],parvals['radiation_field'],index)

    def _compute_residual(self):
        '''Compute the residual values from the observed ratios and models
        '''

        if self.ratiocount < 2:
            msg = f"Not enough ratios.  You need to provide at least 3 observations that can be used to compute 2 ratios that are covered by the ModelSet. From your observations, {self.ratiocount:d} ratio(s)"
            if self.ratiocount>0:
                msg += f" {list(self._modelratios.keys())}"
            msg += " can be computed."
            raise Exception(msg)

        if not self._check_ratio_shapes():
            raise Exception("Observed ratio maps have different dimensions")

        self._residual = dict()
        for r in self._observedratios:
            modelpix = self._modelratios[r].data.flatten()
            residuals = list()
            mdata = ma.masked_invalid(self._observedratios[r].value)
            merror = ma.masked_invalid(self._observedratios[r].error)
            for pix in modelpix:
                #optional fractional error correction for log likelihood.
                _q = (mdata - pix)/merror
                _q = ma.masked_invalid(_q)
                residuals.append(_q)
            # result order is g0,n,y,x

            # Catch the case of a single pixel
            if self._observedratios[r].is_single_pixel():
                newshape = np.hstack((self._modelratios[r].shape))
                _meta = deepcopy(self._modelratios[r].meta)
                _wcs = deepcopy(self._modelratios[r].wcs)
                #print("META",_meta)
                # clean potential crap
                _meta.pop("",None)
                _meta.pop("TITLE",None)
            else:
                newshape = np.hstack((self._modelratios[r].shape,self._observedratios[r].shape))
                _meta = deepcopy(self._observedratios[r].meta)
                _wcs = deepcopy(self._observedratios[r].wcs)
            # result order is g0,n,y,x
            _qq = np.squeeze(np.reshape(residuals,newshape))
            self._residual[r] = CCDData(_qq,unit="adu",wcs=_wcs,meta=_meta)
        self._fancy_index_residual()

    def _fancy_index_residual(self):
        # create a custom numpy array for interpolating the residuals
        # during fitting.
        fk = utils.firstkey(self._modelratios)
        resid_index = np.array(np.arange(len(self._residual)))
        # use linear interpolation grid from model
        rad_index = self._modelratios[fk]._world_axis_lin[1]
        density_index = self._modelratios[fk]._world_axis_lin[0]
        image_index = np.arange(self._observedratios[fk].size)
        self._interpgrid = (resid_index,rad_index,density_index,image_index)
        # put the residual data into a single numpy array
        self._interpvalues = np.stack([self._residual[r].data for r in self._residual.keys()])
        # the flatten the spatial indices, so indices are then [radiation_field,density,pixel]
        self._interpvalues = self._interpvalues.reshape(*self._interpvalues.shape[:-2], -1)

    def _interp_resid(self,density,radiation_field,pixel):
        # density and radiation field must be linear not logarithmic values.
        # pixel is 0-based pixel index into flattened map data array.
        vals = list()
        for i in range(len(self._residual)):
            vals.append(interpn(self._interpgrid,self._interpvalues,(i,radiation_field,density,pixel)))
        return np.array(vals)

    def _compute_chisq(self):
        '''Compute the chi-squared values from observed ratios and models'''
        if self.ratiocount < 2 :
            raise Exception("Not enough ratios to compute chisq.  Need 2, got %d"%self.ratiocount)
        sumary = sum((self._residual[r]._data**2 for r in self._residual))
        self._dof = len(self._residual) - 1
        k = utils.firstkey(self._residual)
        _wcs = deepcopy(self._residual[k].wcs)
        _meta = deepcopy(self._residual[k].meta)
        self._chisq = CCDData(sumary,unit='adu',wcs=_wcs,meta=_meta)
        self._reduced_chisq =  self._chisq.divide(self._dof)
        # must make a copy here otherwise the header is an OrderDict
        # instead of astropy.io.fits.header.Header
        self._reduced_chisq.header =  Header(deepcopy(self._chisq.header))
        self._fixheader(self._chisq)
        self._fixheader(self._reduced_chisq)
        utils.comment("Chi-squared",self._chisq)
        utils.comment(("Reduced Chi-squared (DOF=%d)"%self._dof),self._reduced_chisq)
        self._makehistory(self._chisq)
        self._makehistory(self._reduced_chisq)

    def write_chisq(self,chi="chisq.fits",rchi="rchisq.fits",overwrite=True):
        '''Write the chisq and reduced-chisq data to a file

           :param chi: FITS file to write the chisq map to.
           :type  chi: str
           :param rchi: FITS file to write the reduced chisq map to.
           :type rchi: str
        '''
        self._chisq.write(chi,overwrite=overwrite,hdu_mask='MASK')
        self._reduced_chisq.write(rchi,overwrite=overwrite,hdu_mask='MASK')

    def _refine_density_radiation_field2(self,**kwargs):
        if kwargs['method'] != 'emcee':
            kwargs.pop('steps')
            kwargs.pop('burn')
            progress = kwargs.pop("progress",True) # progress bar
        else:
            progress = kwargs.get("progress",False) #keep the progress keyword for emcee, get vs pop
        # First get the range of density n and radiation field FUV from the
        # model space, in order to provide them to the Parameters object.
        # Since the wk2006 H2 models have a smaller model space,
        # we have to check if H2 is in one of the models used.
        keys = list(self._modelratios.keys())
        if utils._has_H2(keys):
            # this will get the index for the first modelratio that has H2 in it
            # https://stackoverflow.com/questions/2170900/get-first-list-index-containing-sub-string
            i =[idx for idx, s in enumerate(keys) if 'H2' in s][0]
            fk = keys[i]
        else:
            fk = keys[0]
        # this will work regardless if Y axis is Habing, CGS, Draine, etc
        # We do this in linear space because we want the fit report to be in linear space
        # which is what observers care about.
        x,y=utils.get_xy_from_wcs(self._modelratios[fk],linear=True)
        minn=x[0]
        maxn=x[-1]
        minfuv=y[0]
        maxfuv=y[-1]
        if self._radiation_field is None or self._density is None:
            startn = x[int(len(x)/2)]
            startfuv=y[int(len(y)/2)]
        else:
            startn = np.nanmean(self._density.value)
            startfuv = np.nanmean(self._radiation_field.value)
        self._fitparam = Parameters()
    #@todo let user limit these ranges. either they pass in Parameters or
    # limits = {'density':[low,hi], 'radiation_field':[low,hi]}
    # in any unit and convert
        self._fitparam.add('density',min=minn,max=maxn,value=startn)
        self._fitparam.add('radiation_field',min=minfuv,max=maxfuv,value=startfuv)
        #self._fitparam.pretty_print()
        rf = np.empty(self._observedratios[fk].size)
        den = np.empty(self._observedratios[fk].size)
        rfe = np.empty(self._observedratios[fk].size)
        dene = np.empty(self._observedratios[fk].size)
        chi = np.empty(self._observedratios[fk].size)
        rchi = np.empty(self._observedratios[fk].size)
        dflat = self._density.value.flatten()
        rflat = self._radiation_field.value.flatten()
        fmdata = np.empty(self._observedratios[fk].size,dtype=object)
        fm_mask = np.full(shape=self._observedratios[fk].data.shape,fill_value=False).flatten()
        count = 0
        excount = 0
        # turn off progress bar for single pixel or emcee prints out multiple bars.
        if self._observedratios[fk].size == 1:
            progress = False
        with get_progress_bar(progress,self._observedratios[fk].size,leave=True,position=0) as pbar:
            for j in range(self._observedratios[fk].size):
                #use previous coarse fit as first guess
                #print("doing pixel ",j, " of ",self._observedratios[fk].size)
                self._fitparam['density'].value = dflat[j]
                self._fitparam['radiation_field'].value = rflat[j]
                if np.isnan(dflat[j]) or np.isnan(rflat[j]):
                    fmdata[j] = None
                    fm_mask[j] = True
                    den[j] = np.nan
                    dene[j] = np.nan
                    rfe[j] = np.nan
                    rf[j] = np.nan
                    chi[j] = np.nan
                    rchi[j] = np.nan
                else:
                    try:
                        self._minimizer.userargs=(j,)
                        fmdata[j] = self._minimizer.minimize(params=self._fitparam,**kwargs)
                        #if hasattr(fmdata[j],"success")   ugh.  not guaranteed
                        #if fmdata[j].errorbars:
                        count = count+1
                        den[j] = fmdata[j].params['density'].value
                        dene[j] = fmdata[j].params['density'].stderr
                        rf[j] = fmdata[j].params['radiation_field'].value
                        rfe[j] = fmdata[j].params['radiation_field'].stderr
                        chi[j] = fmdata[j].chisqr
                        rchi[j] = fmdata[j].redchi
                        #else:
                        #fmdata[j] = None
                        #fm_mask[j] = True
                    except ValueError as exc:
                        #print("At pixel %d, got valuerror %s with fitparams %s" %(j, exc,self._fitparam))
                        excount = excount+1
                        fmdata[j] = None
                        fm_mask[j] = True
                        den[j] = np.nan
                        dene[j] = np.nan
                        rfe[j] = np.nan
                        rf[j] = np.nan
                        chi[j] = np.nan
                        rchi[j] = np.nan
                pbar.update(1)
        fmdata = fmdata.reshape(self._observedratios[fk].data.shape)
        fm_mask = fm_mask.reshape(self._observedratios[fk].data.shape)
        #ff_mask = ff_mask | np.logical_not(np.isfinite(/*something*/))
        self._fitresult = FitMap(fmdata,wcs=self._observedratios[fk].wcs,mask=fm_mask,name="result")
        print(f"fitted {count} of {self._observedratios[fk].size} pixels")
        print(f'got {excount} exceptions')
        if False:
            self._rf2 = deepcopy(self._radiation_field)
            self._rf2.data = rf.reshape(self._rf2.data.shape)
            self._rf2.uncertainty.array = rfe.reshape(self._rf2.uncertainty.array.shape)
            self._d2 = deepcopy(self._density)
            self._d2.data = den.reshape(self._d2.data.shape)
            self._d2.uncertainty.array = dene.reshape(self._d2.uncertainty.array.shape)
            self._chi2 = deepcopy(self._chisq_min)
            self._rchi2 = deepcopy(self._reduced_chisq_min)
            self._chi2.data = chi.reshape(self._chi2.data.shape)
            self._rchi2.data = rchi.reshape(self._rchi2.data.shape)
        rshape = self._radiation_field.data.shape
        dshape = self._density.data.shape
        self._radiation_field.data = rf.reshape(rshape)
        self._radiation_field.uncertainty.array = rfe.reshape(rshape)
        self._density.data = den.reshape(dshape)
        self._density.uncertainty.array = dene.reshape(dshape)
        self._chisq_min.data = chi.reshape(dshape)
        self._reduced_chisq_min.data = rchi.reshape(dshape)

    def _coarse_density_radiation_field(self):
        '''Compute the best-fit density and radiation field spatial maps
           by searching for the minimum chi-squared at each spatial pixel.'''
        if self._chisq is None or self._reduced_chisq is None: 
            return

        # get the chisq minima of each pixel along the g,n axes
        fk = utils.firstkey(self._modelratios)
        mshape = self._modelratios[fk].shape
        # Wolfire 2006 models have NAXIS=2, while 2020+ have NAXIS=3.
        # Deal with it.
        # @see Measurement squeeze parameter. This should no longer be needed
        if self._modelnaxis == 2:
            firstindex = 0
            secondindex = 1
            thirdindex = 2
            fourthindex = 3
        elif self._modelnaxis == 3:
            if mshape[0] != 1:
                raise Exception("Unexpected NAXIS3 != 1 in model %s" %fk)
            firstindex = 1
            secondindex = 2
            thirdindex = 3
            fourthindex = 4
        rchi_min=np.amin(self._reduced_chisq.data,(firstindex,secondindex))
        chi_min=np.amin(self._chisq.data,(firstindex,secondindex))
        gnxy = np.where(self._reduced_chisq==rchi_min)
        gi = gnxy[firstindex]
        ni = gnxy[secondindex]
        # spatial_idx is which indices are the spatial axes
        if len(gnxy) >= 4:
            # astronomical spatial indices
            spatial_idx = (gnxy[thirdindex],gnxy[fourthindex])
        else:
            spatial_idx = 0
        # model n,g0 indices
        model_idx   = np.transpose(np.array([ni,gi]))
        if self._modelnaxis == 3:
            # add 3rd axis to model_idx
            model_idx = np.insert(model_idx,0,[0],axis=1)
        fk2 = utils.firstkey(self._observedratios)
        newshape = self._observedratios[fk2].shape
        g0 =10**(self._modelratios[fk].wcs.wcs_pix2world(model_idx,0))[:,1]
        n =10**(self._modelratios[fk].wcs.wcs_pix2world(model_idx,0))[:,0]
        self._radiation_field=deepcopy(self._observedratios[fk2])
        if spatial_idx == 0 and newshape == (1,):
            self._radiation_field.data=g0
            self._radiation_field.uncertainty.array=np.array([np.nan])
        else:
            if self.has_vectors:
                self._radiation_field.data = g0
            else: #Measurement with image
                # note this will reshape g0 in radiation_field for us!
                self._radiation_field.data[spatial_idx]=g0
            # We cannot mask nans because numpy does not support writing
            # MaskedArrays to a file. Will get a not implemented error.
            # Therefore just copy the nans over from the input observations.
            self._radiation_field.data[np.isnan(self._observedratios[fk2])] = np.nan
            # kluge because we dont know how to properly calcultate uncertainty on this.
            #self._radiation_field.uncertainty.array=np.zeroes(self._radiation_field.uncertainty.array)
            self._radiation_field.uncertainty.array[:] = np.nan

        self._radiation_field.unit = self.radiation_field_unit
        self._radiation_field.uncertainty.unit = self.radiation_field_unit

        self._density=deepcopy(self._observedratios[fk2])
        if spatial_idx == 0 and newshape == (1,):
            self._density.data=n
            self._density.uncertainty.array=np.array([np.nan])
        else:
            if self.has_vectors: # Measurement with data vector
                self._density.data = n
            else: #Measurement with image
                # note this will reshape g0 in radiation_field for us!
                self._density.data[spatial_idx]=n
                self._density.data[np.isnan(self._observedratios[fk2])] = np.nan
            # kluge because we dont know how to properly calcultate undertainty on this.
            #self._density.uncertainty.array=np.zeroes(self._density.uncertainty.array)
            self._density.uncertainty.array[:] = np.nan

        self._density.unit = self.density_unit
        self._density.uncertainty.unit = self.density_unit
        #this raises exception, CCDData enforces both units the same
        #self._density.uncertainty.unit =  u.dimensionless_unscaled

        #fix the headers
        self._density_radiation_field_header()

        # now save copies of the 2D min chisquares
        self._chisq_min=deepcopy(self._observedratios[fk2])
        if spatial_idx == 0 and newshape == (1,):
            self._chisq_min.data = np.array([chi_min])
        else:
            if self._modelnaxis == 2:
                #print("modelnaxis 2")
                self._chisq_min.data = chi_min
            else:
                #print("modelnaxis!= 2")
                self._chisq_min.data=chi_min[0,:,:]
                self._chisq_min.data[np.isnan(self._observedratios[fk2])] = np.nan
        self._chisq_min.unit = u.dimensionless_unscaled
        self._chisq_min.uncertainty.array = [0.0]
        self._chisq_min.uncertainty.unit = u.dimensionless_unscaled

        self._reduced_chisq_min=deepcopy(self._observedratios[fk2])
        if spatial_idx == 0 and newshape == (1,):
            self._reduced_chisq_min.data = np.array([rchi_min])
        else:
            if self._modelnaxis == 2:
                self._reduced_chisq_min.data=rchi_min
            else:
                self._reduced_chisq_min.data=rchi_min[0,:,:]
            self._reduced_chisq_min.data[np.isnan(self._observedratios[fk2])] = np.nan
        self._reduced_chisq_min.unit = u.dimensionless_unscaled
        self._reduced_chisq_min.uncertainty.array = [0.0]
        self._reduced_chisq_min.uncertainty.unit = u.dimensionless_unscaled

        # update histories
        utils.setkey("BUNIT","Minimum Chi-squared",self._chisq_min)
        utils.setkey("BUNIT",("Minimum Reduced Chi-squared (DOF=%d)"%self._dof),self._reduced_chisq_min)
        self._makehistory(self._reduced_chisq_min)
        self._makehistory(self._chisq_min)

    def _makehistory(self,image):
        '''Add information to HISTORY keyword indicating how the density and radiation field were computed (measurements given, ratios used)

       :param image: The image which to add the history to.
       :type image: :class:`astropy.io.fits.ImageHDU`, :class:`astropy.nddata.CCDData`, or :class:`~pdrtpy.measurement.Measurement`.
        '''
        s = "Measurements provided: " + str(list(self._measurements.keys()))
        utils.history(s,image)
        s = "Ratios used: " + str(list(self._residual.keys()))
        utils.history(s,image)
        utils.signature(image)
        utils.dataminmax(image)

    def _ratioHeader(self,numerator,denominator,label):
        '''Add the RATIO identifier to the appropriate image

           :param numerator:  numerator key of the line ratio
           :type numerator: str
           :param denominator:  denominator key of the line ratio
           :type denominator: str
           :param label:  ratio key indicating which observation image (Measuremnet) to use
           :type label: str
        '''
        utils.addkey("RATIO",label,self._observedratios[label])
        utils.dataminmax(self._observedratios[label])
        utils.signature(self._observedratios[label])

    def _fixheader(self,image):
        '''Put additional axis and header values into an image

        :param image: The image to which to add the header values
        :type image: :class:`astropy.io.fits.ImageHDU`, :class:`astropy.nddata.CCDData`, or :class:`~pdrtpy.measurement.Measurement`.
        '''
        if self._modelnaxis == 2:
            naxis = len(image.shape)
        else:
            naxis = len(image.shape)-1
        ax1=str(naxis-1)
        ax2=str(naxis)
        if "NAXIS" not in image.header:
            utils.setkey("NAXIS",naxis,image)
            utils.setkey("NAXIS"+ax1,image.shape[1],image)
            utils.setkey("NAXIS"+ax2,image.shape[0],image)
        utils.setkey("CTYPE"+ax1,self.density_type,image)
        utils.setkey("CTYPE"+ax2,self.radiation_field_type,image)
        utils.setkey("CUNIT"+ax1,str(self.density_unit),image)
        utils.setkey("CUNIT"+ax2,str(self.radiation_field_unit),image)

        fk = utils.firstkey(self._modelratios)
        mod = self._modelratios[fk]
        utils.setkey("CDELT"+ax1,mod.wcs.wcs.cdelt[0],image)
        utils.setkey("CDELT"+ax2,mod.wcs.wcs.cdelt[1],image)
        utils.setkey("CRVAL"+ax1,mod.wcs.wcs.crval[0],image)
        utils.setkey("CRVAL"+ax2,mod.wcs.wcs.crval[1],image)
        utils.setkey("CRPIX"+ax1,mod.wcs.wcs.crpix[0],image)
        utils.setkey("CRPIX"+ax2,mod.wcs.wcs.crpix[1],image)


    def _density_radiation_field_header(self):
        '''Common header items in the density and radiation field FITS files'''
        self._density.header.pop('RATIO')
        self._radiation_field.header.pop('RATIO')
        # note: must use to_string() here or astropy.io.fits.Card complains
        # about the value being a Unit.  Oddly it doesn't complain for the
        # data units.  Go figure.
        utils.setkey("BUNIT",self.density_unit.to_string(),self._density)
        utils.comment("Best-fit H2 volume density",self._density)
        utils.setkey("BUNIT",self.radiation_field_unit.to_string(),self._radiation_field)
        utils.comment("Best-fit interstellar radiation field",self._radiation_field)
        self._makehistory(self._density)
        self._makehistory(self._radiation_field)
        # convert from OrderedDict to astropy.io.fits.header.Header
        self._density.header = Header(self._density.header)
        self._radiation_field.header = Header(self._radiation_field.header)
        self._density._identifier = "H2 Volume Density"
        self._radiation_field._identifier = "Radiation Field"

    @property
    def table(self):
        #@TODO: make this work for map data ?
        '''Construct the table of input Measurements, and if the fit has been run, the density, radiation field, and :math:`\chi^2` values

        :rtype: :class:`astropy.table.Table`
        '''
        v = self._measurements.values()
        # This only works for astropy version >= 4.1
        # and for some reason requirements.txt did not install it
        # for a test user.
        #t = Table(self._measurements,
        #          units=[m.unit for m in v]
        #          )
        t = Table()
        cols = [Column(data=d,unit=d.unit) for d in v]
        t.add_columns(cols=cols, names=[m.id for m in v])
        if self._observedratios is not None:
            v = self._observedratios.values()
            cols = [Column(data=d,unit=d.unit) for d in v]
            t.add_columns(cols=cols, names=[m.id for m in v])

        if self.radiation_field is not None:
            t.add_column(col=Column(self.radiation_field, unit=self.radiation_field_unit), name=self.radiation_field.id)

        if self.density is not None:
            t.add_column(col=Column(self.density, unit=self.density_unit), name=self.density.id)

        if self._chisq_min is not None:
            t.add_column(col=Column(self._chisq_min, unit=None), name="Chi-square")

        for j in t.columns:
            t[j].format = '3.2E'
        return t

    #========= Below is deprecated
    def _compute_delta_sq(self):
        '''Compute the difference-squared values from the observed ratios
           and models - multi-pixel version and store in _deltasq member'''
        self._deltasq = self._computeDelta(f=0)

    def _computeDelta(self,f):
        '''Compute the difference-squared values from the observed ratios
           and models - multi-pixel version

           :param f: fractional amount by which the variance is underestimated.
           For traditional chi-squared calculation f is zero.
           For log-likelihood calculation f is positive and less than 1.
           See, e.g. https://emcee.readthedocs.io/en/stable/tutorials/line/#maximum-likelihood-estimation
           :type f: float
        '''
        if not self._modelratios: # empty list or None
            raise Exception("No model data ready.  Was read_models() called?")

        if self.ratiocount < 2 :
            raise Exception("Not enough ratios.  You need to provide at least 3 observations that can be used to compute 2 ratios that are covered by the ModelSet. From your observations, only %d ratios can be computed."%self.ratiocount)

        if not self._check_ratio_shapes():
            raise Exception("Observed ratio maps have different dimensions")

        returnval = dict()
        for r in self._observedratios:
            sz = self._modelratios[r].size
            modelpix = np.reshape(self._modelratios[r],sz)

            residuals = list()
            mf = ma.masked_invalid(self._observedratios[r].value)
            me = ma.masked_invalid(self._observedratios[r].error)
            #frac_error = f*modelpix  # this is actually slower than looping over modelpix
            s2 = me**2
            add_term = 0
            for pix in modelpix:
                #optional fractional error correction for log likelihood.
                if f != 0:
                   #term is actually log(2*pi*s2) but addition of
                   #constant makes no difference in likelihood.
                    frac_error  = f*pix
                    s2 += frac_error**2
                    add_term += np.log(s2)
                _q = (mf - pix)**2/s2 + add_term
                _q = ma.masked_invalid(_q)
                residuals.append(_q)
            # result order is g0,n,y,x

            # Catch the case of a single pixel
            if len(self._observedratios[r].shape) == 0:
                newshape = np.hstack((self._modelratios[r].shape))
                _meta= deepcopy(self._modelratios[r].meta)
                # clean potential crap
                _meta.pop("",None)
                _meta.pop("TITLE",None)
            else:
                newshape = np.hstack((self._modelratios[r].shape,self._observedratios[r].shape))
                _meta= deepcopy(self._observedratios[r].meta)
            # result order is y,x,g0,n
            #newshape = np.hstack((self._observedratios[r].shape,self._modelratios[r].shape))
            _qq = np.squeeze(np.reshape(residuals,newshape))
            #print("QQ SHAPE ",_qq.shape)
            # WCS will be None for single pixel
            _wcs = deepcopy(self._observedratios[r].wcs)
            returnval[r] = CCDData(_qq,unit="adu",wcs=_wcs,meta=_meta)

        return returnval
