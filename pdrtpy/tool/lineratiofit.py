import itertools
import collections
from copy import deepcopy
import datetime

import numpy as np
import numpy.ma as ma
import scipy.stats as stats

from astropy.io import fits
from astropy.io.fits.header import Header
import astropy.wcs as wcs
import astropy.units as u
from astropy.table import Table
from astropy.nddata import NDDataArray, CCDData, NDUncertainty, StdDevUncertainty, VarianceUncertainty, InverseVariance

from ..tool.toolbase import ToolBase
from ..plot.lineratioplot import LineRatioPlot
from .. import pdrutils as utils
from ..modelset import ModelSet

class LineRatioFit(ToolBase):
    """Tool to fit observations of flux ratios to a set of models"""
    def __init__(self,modelset=ModelSet.WolfireKaufman(),measurements=None):
        if type(modelset) == str:
            # may need to disable this
            self._initialize_modelTable(modelset)
        self._modelset = modelset

        if type(measurements) == dict or measurements is None:
            self._measurements = measurements
        else:
            self._init_measurements(measurements)

        self._modelratios = None
        self._set_model_files_used()
        self._observedratios = None
        self._chisq = None
        self._deltasq = None
        self._reduced_chisq = None
        self._likelihood = None
        self.radiation_field_unit = None
        self.density_unit = None
        self._plotter = LineRatioPlot(self)
    
    @property
    def modelset(self):
        """The underlying :class:`ModelSet`"""
        return self._modelset

    @property
    def measurements(self):
        """The stored :class:`measurements <Measurement>` as dictionary with Measurement IDs as keys
   
        :rtype: dict
        """
        return self._measurements
    
    @property
    def measurementIDs(self):
        '''The stored measurement IDs.

        :rtype: :class:`dict_keys`
        '''
         
        if self._measurements is None: return None
        return self._measurements.keys()

    @property
    def observed_ratios(self):
        '''The list of the observed line ratios that have been input so far.
 
        :rtype: list of str
        '''
        return list(self._observedratios.keys())

    @property
    def ratiocount(self):
        '''The number of ratios that match models available in the 
           current ModelSet given the current set of measurements
 
        :rtype: int
        '''
        return self._modelset.ratiocount(self.measurementIDs)

    @property
    def density(self):
        '''The computed density value(s).

        :rtype: :class:`~pdrtpy.measurement.Measurement`
        '''
        return self._density

    @property
    def radiation_field(self):
        '''The computed radiation field value(s).

        :rtype: :class:`~pdrtpy.measurement.Measurement`
        '''
        return self._radiation_field

    def _init_measurements(self,m):
        """Initialize the measurements from an input list
        
        :param m: the input list of Measurements
        :type m: list
        """
        self._measurements = dict()
        for mm in m:
            self._measurements[mm.id] = mm

    #@deprecated
    #def _initialize_modelTable(self,filename):
    #    """initialize models from an IPAC format ASCII file"""
    #    self._modelTable=Table.read(filename,format="ascii.ipac")

    def _set_model_files_used(self):
        self._model_files_used = dict()
        if self._measurements is None: return
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
       try:
           if value == NotImplemented:
               s1 = d[utils.firstkey(d)].header[kw]
           else: 
               s1 = value
           return np.all([m.header[kw] == s1 for m in d.values()])
       except KeyError:
           print("WARNING: %s keyword not present in all Measurements"%kw)
           return False
       
    def _check_measurement_shapes(self):
       if self._measurements == None: return False
       return self._check_shapes(self._measurements)

    def _check_ratio_shapes(self):
       if self._observedratios == None: return False
       return self._check_shapes(self._observedratios)
            
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
   
    
    def read_models(self,unit):
        """Given a list of measurement IDs, find and open the FITS files that have matching ratios
           and populate the _modelratios dictionary.  Use astropy's CCDdata as a storage mechanism. 

           :param  m: list of measurement IDS (string)
           :type m: list
           :param unit: units of the data 
           :type unit: string or astropy.Unit
        """
        d = utils.model_dir()

        self._modelratios = dict()
        for (k,p) in self._modelset.find_files(self.measurementIDs):
            thefile = d+p
            self._modelratios[k] = CCDData.read(thefile,unit=unit)
            if True:
                self._modelratios[k].header["CUNIT1"] = "cm-3"
                #self._modelratios[k].header["CUNIT2"] = "erg cm-2 s-1"
                self._modelratios[k].header["CUNIT2"] = "Habing"
            if not self.density_unit:
                try:
                    self.density_unit = u.Unit(self._modelratios[k].header["CUNIT1"])
                except KeyError:
                    raise Exception("Keyword CUNIT1 is required in file %s FITS header to describe units of density" % thefile)
            if not self.radiation_field_unit:
                try:
                    self.radiation_field_unit    = u.Unit(self._modelratios[k].header["CUNIT2"])
                except KeyError:
                    raise Exception("Keyword CUNIT2 is required in file %s FITS header to describe units of interstellar radiation field"%thefile)
    
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
        if len(m1) != 0:
            if not self._check_header("CTYPE1"):
               raise Exception("CTYPE1 of your input Measurements do not match. Please ensure coordinates of all Measurements are the same.")
            if not self._check_header("CTYPE2"):
               raise Exception("CTYPE2 of your input Measurements do not match. Please ensure coordinates of all Measurements are the same.")

        #Only allow beam = None if single value measurements.
        if len(m1) == 0 :
            if self._check_header("BMAJ",None) or self._check_header("BMIN",None) or self._check_header("BPA",None):
               utils.warn(self,"No beam parameters in Measurement headers, assuming they are all equal!")
        #if not self._check_header("BUNIT") ...

    def run(self):
        '''Run the full computation'''
        self._check_compatibility()
        self.read_models(unit='erg s-1 cm-2 sr-1')
        self._compute_valid_ratios()
        # eventually need to check that the maps overlap in real space.
        self._compute_delta_sq()
        self._compute_chisq()
        self._write_chisq()
        self.compute_density_radiation_field()
     

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
            num = self._convert_if_necessary(self._measurements[p["numerator"]])
            denom = self._convert_if_necessary(self._measurements[p["denominator"]])
            self._observedratios[label] = deepcopy(num/denom)
            #@TODO create a meaningful header for the ratio map
            self._ratioHeader(p["numerator"],p["denominator"],label)
        self._add_oi_cii_fir()

    def _add_oi_cii_fir(self):
        '''add special case ([O I] 63 micron + [C II] 158 micron)/IFIR to observed ratios'''
        m = self.measurementIDs
        if "CII_158" in m and "FIR" in m:
            if "OI_63" in m:
                lab="OI_63+CII_158/FIR"
                #print(l)
                oi = self._convert_if_necessary(self._measurements["OI_63"])
                cii = self._convert_if_necessary(self._measurements["CII_158"])
                a = deepcopy(oi+cii)
                b = deepcopy(self._measurements["FIR"])
                self._observedratios[lab] = a/b
                self._ratioHeader("OI_63+CII_158","FIR",lab)
            if "OI_145" in m:
                lab="OI_145+CII_158/FIR"
                #print(ll)
                oi = self._convert_if_necessary(self._measurements["OI_145"])
                cii = self._convert_if_necessary(self._measurements["CII_158"])
                aa = deepcopy(oi+cii)
                bb = deepcopy(self._measurements["FIR"])
                self._observedratios[lab] = aa/bb
                self._ratioHeader("OI_145+CII_158","FIR",lab)
                    
    #deprecated
    def __computeDeltaSq(self):
        '''Compute the difference-squared values from the observed ratios and models - single pixel version'''
        if not self._modelratios: # empty list or None
            raise Exception("No model data ready.  You need to call read_fits")
        if self.ratiocount < 2 :
            raise Exception("Not enough ratios to compute deltasq.  Need 2, got %d"%self.ratiocount)
        self._deltasq = dict()
        for r in self._observedratios:
            _z = self._modelratios[r].multiply(-1.0)
            _z._data = _z._data + self._observedratios[r].flux
            _q = _z.divide(self._observedratios[r].error)
            self._deltasq[r] = _q.multiply(_q)

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
            raise Exception("No model data ready.  You need to call read_fits")
            
        if self.ratiocount < 2 :
            raise Exception("Not enough ratios to compute deltasq.  Need 2, got %d"%self.ratiocount)

        if not self._check_ratio_shapes():
            raise Exception("Observed ratio maps have different dimensions")
            
        returnval = dict()
        for r in self._observedratios:
            sz = self._modelratios[r].size
            _z = np.reshape(self._modelratios[r],sz)

            ff = list()
            for pix in _z:
                mf = ma.masked_invalid(self._observedratios[r].flux)
                me = ma.masked_invalid(self._observedratios[r].error)  
                #optional fractional error correction for log likelihood.
                #
                if f == 0:
                    s2 = me**2
                    add_term = 0
                else:
                   #term is actually log(2*pi*s2) but addition of 
                   #constant makes no difference in likelihood.
                    frac_error  = f*pix
                    s2 = me**2+frac_error**2
                    add_term = np.log(s2)
                #_q = (self._observedratios[r].flux - pix)/self._observedratios[r].error
                _q = (mf - pix)**2/s2 + add_term
                ff.append(_q)
            # result order is g0,n,y,x
            #print("Shape mr1 ",self._modelratios[r].shape,type(self._modelratios[r].shape[0]))
            #print("Shape or2 ",self._observedratios[r].shape)#,type(self._observedratios[r].shape[0]))

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
            #print("newshape type(newshape)",newshape,type(newshape),type(newshape[0]))
            #print("ff.shape ",np.shape(ff))
            # result order is y,x,g0,n
            #newshape = np.hstack((self._observedratios[r].shape,self._modelratios[r].shape))
            _qq = np.reshape(ff,newshape)
            # WCS will be None for single pixel
            _wcs = deepcopy(self._observedratios[r].wcs)
            returnval[r] = CCDData(_qq,unit="adu",wcs=_wcs,meta=_meta)
    
        return returnval

    def _compute_log_likelihood(self,f):
        """***Experimental***

           :param f: fractional amount by which the variance is underestimated. 
           For traditional chi-squared calculation f is zero.  
           For log-likelihood calculation f is positive and less than 1.
           See, e.g. https://emcee.readthedocs.io/en/stable/tutorials/line/#maximum-likelihood-estimation

           :type f: float
        """
        l = self._computeDelta(f)
        sumary = -0.5* sum((l[r]._data for r in l))
        k = utils.firstkey(self._deltasq)
        _wcs = deepcopy(self._deltasq[k].wcs)
        _meta = deepcopy(self._deltasq[k].meta)
        self._likelihood = CCDData(sumary,unit='adu',wcs=_wcs,meta=_meta)
        self._fixheader(self._likelihood)
        self._makehistory(self._likelihood)


#    def log_likelihood(self, theta):
#       ratio, log_f = theta
#       l = self._likelihood(f=np.exp(log_f))
#       sumary = -0.5* sum((self._likelihood[r]._data for r in self._likelihood))
#       return sumary

    def _compute_chisq(self):
        '''Compute the chi-squared values from observed ratios and models'''
        if self.ratiocount < 2 :
            raise Exception("Not enough ratios to compute chisq.  Need 2, got %d"%self.ratiocount)
        sumary = sum((self._deltasq[r]._data for r in self._deltasq))
        self._dof = len(self._deltasq) - 1
        k = utils.firstkey(self._deltasq)
        _wcs = deepcopy(self._deltasq[k].wcs)
        _meta = deepcopy(self._deltasq[k].meta)
        self._chisq = CCDData(sumary,unit='adu',wcs=_wcs,meta=_meta)
        self._reduced_chisq =  self._chisq.divide(self._dof)
        # must make a copy here otherwise the header is an OrderDict
        # instead of astropy.io.fits.header.Header
        self._reduced_chisq.header =  Header(self._chisq.header)
        self._fixheader(self._chisq)
        self._fixheader(self._reduced_chisq)
        utils.setkey("BUNIT","Chi-squared",self._chisq)
        utils.setkey("BUNIT",("Reduced Chi-squared (DOF=%d)"%self._dof),self._reduced_chisq)
        self._makehistory(self._chisq)
        self._makehistory(self._reduced_chisq)
        
    def _write_chisq(self,file="chisq.fits",rfile="rchisq.fits"):
        '''Write the chisq and reduced-chisq data to a file
         
           :param file: FITS file to write the chisq map to.
           :type  file: str
           :param rfile: FITS file to write the reduced chisq map to.
           :type rfile: str
        '''
        self._chisq.write(file,overwrite=True,hdu_mask='MASK')
        self._reduced_chisq.write(rfile,overwrite=True,hdu_mask='MASK')  

    def _compute_likeliest(self):
        """***Experimental*** 
        Compute the likeliest density n and radiation field spatial maps
        """
        if self._likelihood is None: return
        
        # get the likelihood maxima of each pixel along the g,n axes
        z=np.amax(self._likelihood,(0,1))
        gi,ni,yi,xi=np.where(self._likelihood==z)
        #print(gi)
        #print(len(gi),len(ni),len(xi),len(yi))
        spatial_idx = (yi,xi)
        model_idx   = np.transpose(np.array([ni,gi]))
        # qq[:,:2] takes the first two columns of qq
        # [:,[1,0]] swaps those columns
        # np.flip would also swap them.
        #print(qq[:,:2][:,[1,0]])
        #print(np.flip(qq[:,:2]))
        fk = utils.firstkey(self._modelratios)
        fk2 = utils.firstkey(self._observedratios)
        newshape = self._observedratios[fk2].shape
        g0=10**(self._modelratios[fk].wcs.wcs_pix2world(model_idx,0))[:,1]
        n =10**(self._modelratios[fk].wcs.wcs_pix2world(model_idx,0))[:,0]
        self.L_radiation_field=self._observedratios[fk2].copy()
        self.L_radiation_field.data[spatial_idx]=g0
        self.L_radiation_field.unit = self.radiation_field_unit
        self.L_radiation_field.uncertainty.unit = self.radiation_field_unit
        self.L_density=self._observedratios[fk2].copy()      
        self.L_density.data[spatial_idx]=n
        self.L_density.unit = self.density_unit
        self.L_density.uncertainty.unit = self.density_unit
        #fix the headers
        #self._density_radiation_field_header() 
        
    def compute_density_radiation_field(self):
        '''Compute the best-fit density n and radiation field spatial maps 
           by searching for the minimum chi-squared at each spatial pixel.'''
        if self._chisq is None or self._reduced_chisq is None: return
        
        # get the chisq minima of each pixel along the g,n axes
        z=np.amin(self._reduced_chisq,(0,1))
        gnxy = np.where(self._reduced_chisq==z)
        gi = gnxy[0]
        ni = gnxy[1]
        if len(gnxy) == 4:
            # astronomical spatial indices
            spatial_idx = (gnxy[2],gnxy[3])
        else:
            spatial_idx = 0
        # model n,g0 indices
        model_idx   = np.transpose(np.array([ni,gi]))
        # qq[:,:2] takes the first two columns of qq
        # [:,[1,0]] swaps those columns
        # np.flip would also swap them.
        #print(qq[:,:2][:,[1,0]])
        #print(np.flip(qq[:,:2]))
        fk = utils.firstkey(self._modelratios)
        fk2 = utils.firstkey(self._observedratios)
        newshape = self._observedratios[fk2].shape
        g0 =10**(self._modelratios[fk].wcs.wcs_pix2world(model_idx,0))[:,1]
        n =10**(self._modelratios[fk].wcs.wcs_pix2world(model_idx,0))[:,0]
        #print("G ",g0)
        #print("N ",n)

        self._radiation_field=deepcopy(self._observedratios[fk2])
        if spatial_idx == 0:
            self._radiation_field.data=g0[0]
        else:
            # note this will reshape g0 in radiation_field for us!
            self._radiation_field.data[spatial_idx]=g0
            # We cannot mask nans because numpy does not support writing
            # MaskedArrays to a file. Will get a not implemented error.
            # Therefore just copy the nans over from the input observations.
            self._radiation_field.data[np.isnan(self._observedratios[fk2])] = np.nan
        self._radiation_field.unit = self.radiation_field_unit
        self._radiation_field.uncertainty.unit = self.radiation_field_unit

        self._density=deepcopy(self._observedratios[fk2])
        if spatial_idx == 0:
            self._density.data=n[0]
        else:
            self._density.data[spatial_idx]=n
            self._density.data[np.isnan(self._observedratios[fk2])] = np.nan

        self._density.unit = self.density_unit
        self._density.uncertainty.unit = self.density_unit

        #fix the headers
        self._density_radiation_field_header() 
            
                 
    def _makehistory(self,image):
        '''Add information to HISTORY keyword indicating how the density and radiation field were computed (measurements given, ratios used)

       :param image: The image which to add the history to.
       :type image: :class:`astropy.io.fits.ImageHDU`, :class:`astropy.nddata.CCDData`, or :class:`~pdrtpy.measurement.Measurement`.
        '''
        s = "Measurements provided: "
        for k in self._measurements.keys():
            s = s + k + ", "
        utils.history(s,image)
        s = "Ratios used: "
        for k in self._deltasq.keys():
            s = s + k + ", "
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
        # @TODO make these headers compliant with inputs (e.g. requested units)
        naxis = len(image.shape)
        ax1=str(naxis-1)
        ax2=str(naxis)
        utils.setkey("CTYPE"+ax1,"Log(Volume Density)",image)
        utils.setkey("CTYPE"+ax2,"Log(Radiation Field)",image)
        utils.setkey("CUNIT"+ax1,str(self.density_unit),image)
        utils.setkey("CUNIT"+ax2,str(self.radiation_field_unit),image)
        #@TODO this cdelts will change with new models.  make this flexible
        utils.setkey("CDELT"+ax1,0.25,image)
        utils.setkey("CDELT"+ax2,0.25,image)
        utils.setkey("CRVAL"+ax1,-0.5,image)
        utils.setkey("CRVAL"+ax2,1.0,image)
        utils.setkey("CRPIX"+ax1,1.0,image)
        utils.setkey("CRPIX"+ax2,1.0,image)
        
         
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

       
    def _convert_if_necessary(self,image):
        """If the input image has units of K km/s convert it to intensity

        :param image: The image to which to add the header values
        :type image: :class:`astropy.io.fits.ImageHDU`, :class:`astropy.nddata.CCDData`, or :class:`~pdrtpy.measurement.Measurement`.
        """
        if image.header["BUNIT"] == "K km/s":
            return utils.convert_integrated_intensity(image)
        else:
            return image

#if __name__ == "__main__":
#    from ..measurement import Measurement 
#    from .. import pdrutils as utils
#    m1 = Measurement(data=30,uncertainty = StdDevUncertainty([5.]),identifier="OI_145",unit="adu")
#    m2 = Measurement(data=10.,uncertainty = StdDevUncertainty(2.),identifier="CI_609",unit="adu")
#    m3 = Measurement(data=10.,uncertainty = StdDevUncertainty(1.5),identifier="CO_21",unit="adu")
#    m4 = Measurement(data=100.,uncertainty = StdDevUncertainty(10.),identifier="CII_158",unit="adu")
#
#    p = LineRatioFit(measurements = [m1,m2,m3,m4])
#    print("num ratios:", p.ratiocount)
#    print("modelfiles used: ", p._model_files_used)
#    p.run()
#    p._modelset.get_ratio_elements(p.measurements)
#    p.makeRatioOverlays(cmap='gray')
#    p.plotRatiosOnModels()
#    p._observedratios
