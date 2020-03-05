#!/usr/bin/env python


#todo: Look into seaborn https://seaborn.pydata.org
# Also https://docs.bokeh.org/en
# especially for coloring and style
import itertools
import collections
from copy import deepcopy
import datetime

import numpy as np
import numpy.ma as ma
import scipy.stats as stats

from astropy.io import fits
import astropy.wcs as wcs
import astropy.units as u
from astropy.table import Table
from astropy.nddata import NDDataArray, CCDData, NDUncertainty, StdDevUncertainty, VarianceUncertainty, InverseVariance

from tool import Tool
from plot import LineRatioPlot
import pdrutils as utils
from modelset import ModelSet


# potential new structure
# PDRToolbox
#   utils
#   tool.py
#    class Tool(object)
#      self._plotter = None
#      def run(self) { return;} // all subclass tools must implement run
#   lineratiofit(Tool)
#   h2excitation(Tool)
#   plot
class LineRatioFit(Tool):
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
        self._set_modelfilesUsed()
        self._observedratios = None
        self._chisq = None
        self._deltasq = None
        self._reducedChisq = None
        self._likelihood = None
        self.radiation_field_unit = None
        self.density_unit = None
        self._plotter = LineRatioPlot(self)
    
    @property
    def modelset(self):
        return self._modelset

    @property
    def measurements(self):
        '''return stored measurements as dictionary with Measurement IDs as keys'''
        return self._measurements
    
    @property
    def measurementIDs(self):
        '''Return stored measurement IDs as `dict_keys` iterator'''
        if self._measurements is None: return None
        return self._measurements.keys()

    @property
    def ratiocount(self):
        '''Return number of ratios that match models available in the 
           current ModelSet given the current set of measurements'''
        return self._modelset.ratiocount(self.measurementIDs)

    def _init_measurements(self,m):
        self._measurements = dict()
        for mm in m:
            self._measurements[mm.id] = mm

    #@deprecated
    def _initialize_modelTable(self,filename):
        """initialize models from an IPAC format ASCII file"""
        self._modelTable=Table.read(filename,format="ascii.ipac")

    def _set_modelfilesUsed(self):
        self._modelfilesUsed = dict()
        if self._measurements is None: return
        for x in self._modelset.find_files(self.measurementIDs):
            self._modelfilesUsed[x[0]]=x[1]

    def _check_shapes(self,d):
       #ugly
       s1 = d[list(d.keys())[0]].shape
       return np.all([m.shape == s1 for m in d.values()])

    def _check_measurement_shapes(self):
       if self._measurements == None: return False
       return self._check_shapes(self._measurements)

    def _check_ratio_shapes(self):
       if self._observedratios == None: return False
       return self._check_shapes(self._observedratios)
            
    def add_measurement(self,m):
        '''Add a Measurement to internal dictionary used to compute ratios

           Parameters:
              m - a Measurement instance
        '''
        if self._measurements:
            self._measurements[m.id] = m
        else:
            self._init_measurements(m)
        self._set_modelfilesUsed()
        
    def remove_measurement(self,id):
        '''Delete a measurement from the internal dictionary used to compute ratios.
           raises KeyError if id not in existing Measurements
        '''
        del self._measurements[id]
        self._set_modelfilesUsed()
   

    
    #TODO fix this after moving find_files to ModelSet.  models directory will be a function of ModelSet instance.
    def read_models(self,unit):
        """Given a list of measurement IDs, find and open the FITS files that have matching ratios
           and populate the _modelratios dictionary.  Use astropy's CCDdata as a storage mechanism. 
            Parameters: 
                 m - list of measurement IDS (string)
                 unit - units of the data (string)
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
    
    def run(self):
        '''Run the full computation'''
        self.read_models(unit='adu')
        self.computeValidRatios()
        self.computeDeltaSqMap()
        self.compute_chisq()
        self.write_chisq()
        self.compute_density_radiation_field_maps()
     
    def observedratios(self):
        '''Returns a list of the observed line ratios that have been input so far'''
        return list(self._observedratios.keys())

    def computeValidRatios(self):
        '''Compute the valid observed ratio maps for the available model data'''
        if not self._check_measurement_shapes():
            raise TypeError("Measurement maps have different dimensions")

        # Note find_ratio_elemets does not handle case of OI+CII/FIR so 
        # we have to deal with that separately below.
        z = self._modelset._find_ratio_elements(self.measurementIDs)
        self._observedratios = dict()
        for p in z:
            label = p["numerator"]+"/"+p["denominator"]
            # deepcopy workaround for bug: https://github.com/astropy/astropy/issues/9006
            self._observedratios[label] = deepcopy(self._measurements[p["numerator"]])/deepcopy(self._measurements[p["denominator"]])
            #@TODO create a meaningful header for the ratio map
            self._ratioHeader(p["numerator"],p["denominator"],label)
        self._add_oi_cii_fir()

    def _add_oi_cii_fir(self):
        '''add special case ([O I] 63 micron + [C II] 158 micron)/IFIR to observed ratios'''
        m = self.measurementIDs
        if "CII_158" in m and "FIR" in m:
            if "OI_63" in m:
                l="OI_63+CII_158/FIR"
                #print(l)
                a = deepcopy(self._measurements["OI_63"])+deepcopy(self._measurements["CII_158"])
                b = deepcopy(self._measurements["FIR"])
                self._observedratios[l] = a/b
                self._ratioHeader("OI_63+CII_158","FIR",l)
            if "OI_145" in m:
                ll="OI_145+CII_158/FIR"
                #print(ll)
                aa = deepcopy(self._measurements["OI_145"])+deepcopy(self._measurements["CII_158"])
                bb = deepcopy(self._measurements["FIR"])
                self._observedratios[ll] = aa/bb
                self._ratioHeader("OI_145+CII_158","FIR",ll)
                    
    ##deprecated
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

    def computeDeltaSqMap(self):
        '''Compute the difference-squared values from the observed ratios 
           and models - multi-pixel version and store in _deltasq member'''
        self._deltasq = self._computeDelta(f=0)

    def _computeDelta(self,f):
        '''Compute the difference-squared values from the observed ratios 
           and models - multi-pixel version
           Parameters:
                f - fractional amount by which the variance is underestimated. 
                    For traditional chi-squared calculation f is zero.  
                    For log-likelihood calculation f is positive and less than 1.
                    See, e.g. https://emcee.readthedocs.io/en/stable/tutorials/line/#maximum-likelihood-estimation
        '''
        if not self._modelratios: # empty list or None
            raise Exception("No model data ready.  You need to call read_fits")
            
        if self.ratiocount < 2 :
            raise Exception("Not enough ratios to compute deltasq.  Need 2, got %d"%self.ratiocount)

        if not self._check_ratio_shapes():
            raise TypeError("Observed ratio maps have different dimensions")
            
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
            newshape = np.hstack((self._modelratios[r].shape,self._observedratios[r].shape))
            # result order is y,x,g0,n
            #newshape = np.hstack((self._observedratios[r].shape,self._modelratios[r].shape))
            _qq = np.reshape(ff,newshape)
            _wcs = self._observedratios[r].wcs.deepcopy()
            _meta= deepcopy(self._observedratios[r].meta)
            returnval[r] = CCDData(_qq,unit="adu",wcs=_wcs,meta=_meta)
    
        return returnval

    def computeLogLikelihood(self,f):
        l = self._computeDelta(f)
        sumary = -0.5* sum((l[r]._data for r in l))
        k = utils.firstkey(self._deltasq)
        self._likelihood = CCDData(sumary,unit='adu',wcs=self._deltasq[k].wcs,meta=self._deltasq[k].meta)
        self._fixheader(self._likelihood)
        self._makehistory(self._likelihood)


#    def log_likelihood(self, theta):
#       ratio, log_f = theta
#       l = self._likelihood(f=np.exp(log_f))
#       sumary = -0.5* sum((self._likelihood[r]._data for r in self._likelihood))
#       return sumary

    def compute_chisq(self):
        '''Compute the chi-squared values from observed ratios and models'''
        if self.ratiocount < 2 :
            raise Exception("Not enough ratios to compute chisq.  Need 2, got %d"%self.ratiocount)
        sumary = sum((self._deltasq[r]._data for r in self._deltasq))
        self._dof = len(self._deltasq) - 1
        k = utils.firstkey(self._deltasq)
        self._chisq = CCDData(sumary,unit='adu',wcs=self._deltasq[k].wcs,meta=self._deltasq[k].meta)
        self._reducedChisq =  self._chisq.divide(self._dof)
        self._fixheader(self._chisq)
        self._fixheader(self._reducedChisq)
        self.setkey("BUNIT","Chi-squared",self._chisq)
        self.setkey("BUNIT",("Reduced Chi-squared (DOF=%d)"%self._dof),self._reducedChisq)
        self._makehistory(self._chisq)
        self._makehistory(self._reducedChisq)
        
    def write_chisq(self,file="chisq.fits",rfile="rchisq.fits"):
        '''Write the chisq and reduced-chisq data to a file'''
        self._chisq.write(file,overwrite=True,hdu_mask='MASK')
        self._reducedChisq.write(rfile,overwrite=True,hdu_mask='MASK')  

    def compute_likeliest(self):
        '''Compute the likeliest density n and radiation field spatial maps by searching for the minimum chisq at each spatial pixel.'''
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
        
    def compute_density_radiation_field_maps(self):
        '''Compute the best-fit density n and radiation field spatial maps 
           by searching for the minimum chisq at each spatial pixel.'''
        if self._chisq is None or self._reducedChisq is None: return
        
        # get the chisq minima of each pixel along the g,n axes
        z=np.amin(self._reducedChisq,(0,1))
        gi,ni,yi,xi=np.where(self._reducedChisq==z)
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

        # note this will reshape g0 in radiation_field for us!
        self._radiation_field=self._observedratios[fk2].copy()
        self._radiation_field.data[spatial_idx]=g0

        # We cannot mask nans because numpy does not support writing
        # MaskedArrays to a file. Will get a not implemented error.
        # Therefore just copy the nans over from the input observations.
        self._radiation_field.data[np.isnan(self._observedratios[fk2])] = np.nan
        self._radiation_field.unit = self.radiation_field_unit
        self._radiation_field.uncertainty.unit = self.radiation_field_unit

        self._density=self._observedratios[fk2].copy()      
        self._density.data[spatial_idx]=n
        self._density.data[np.isnan(self._observedratios[fk2])] = np.nan
        self._density.unit = self.density_unit
        self._density.uncertainty.unit = self.density_unit

        #fix the headers
        self._density_radiation_field_header() 
            
    def comment(self,value,image):
        '''Add a comment to the image header'''
        self.addkey("COMMENT",value,image)
        
    def history(self,value,image):
        '''Add a history to the image header'''
        self.addkey("HISTORY",value,image)
        
    def addkey(self,key,value,image):
        '''Add a keyword,value pair to the image header'''
        if key in image.header and type(value) == str:    
            image.header[key] = image.header[key]+" "+value
        else:
            image.header[key]=value

    def setkey(self,key,value,image):
        '''Set the value of an existing keyword in the image header'''
        image.header[key]=value
        
    def _dataminmax(self,image):
        '''Set the data maximum and minimum in image header'''
        self.setkey("DATAMIN",np.nanmin(image.data),image)
        self.setkey("DATAMAX",np.nanmax(image.data),image)
            
    def _signature(self,image):
        '''Add AUTHOR and DATE keywords to the image header'''
        self.setkey("AUTHOR","PDR Toolbox "+utils.version(),image)
        self.setkey("DATE",utils.now(),image)
                 
    def _makehistory(self,image):
        '''Add information to HISTORY keyword indicating how the n,G0 were computed (measurements give, ratios used)'''
        s = "Measurements provided: "
        for k in self._measurements.keys():
            s = s + k + ", "
        self.addkey("HISTORY",s,image)
        s = "Ratios used: "
        for k in self._deltasq.keys():
            s = s + k + ", "
        self.addkey("HISTORY",s,image)
        self._signature(image)
        self._dataminmax(image)

    def _ratioHeader(self,numerator,denominator,label):
        '''Add the RATIO identifier to the appropriate image
           Parameters:
                numerator - numerator string key of the line ratio
                denominator - denominator string key of the line ratio
                label - ratio label
        '''
        self.addkey("RATIO",label,self._observedratios[label])
        self._dataminmax(self._observedratios[label])
        self._signature(self._observedratios[label])
        
    def _fixheader(self,image):
        '''Put axis 3 and 4 header values into chisq maps
           Parameter:
                image - the image to which to add the header values
        '''
        # @TODO make these headers compliant with inputs (e.g. requested units)
        self.setkey("CTYPE3","Log(Volume Density)",image)
        self.setkey("CTYPE4","Log(Radiation Field)",image)
        self.setkey("CUNIT3",str(self.density_unit),image)
        self.setkey("CUNIT4",str(self.radiation_field_unit),image)
        #@TODO this cdelts will change with new models.  make this flexible
        self.setkey("CDELT3",0.25,image)
        self.setkey("CDELT4",0.25,image)
        self.setkey("CRVAL4",-0.5,image)
        self.setkey("CRVAL3",1.0,image)
        self.setkey("CRPIX3",1.0,image)
        self.setkey("CRPIX4",1.0,image)
        
         
    def _density_radiation_field_header(self):
        '''Common header items in the density and radiation field FITS files'''
        self._density.header.pop('RATIO')
        self._radiation_field.header.pop('RATIO')
        # note: must use to_string() here or astropy.io.fits.Card complains
        # about the value being a Unit.  Oddly it doesn't complain for the
        # data units.  Go figure.
        self.setkey("BUNIT",self.density_unit.to_string(),self._density)
        self.comment("Best-fit H2 volume density",self._density)
        self.setkey("BUNIT",self.radiation_field_unit.to_string(),self._radiation_field)
        self.comment("Best-fit interstellar radiation field",self._radiation_field)
        self._makehistory(self._density)
        self._makehistory(self._radiation_field)

       

if __name__ == "__main__":
    from measurement import Measurement 
    import pdrutils as utils
    m1 = Measurement(data=30,uncertainty = StdDevUncertainty([5.]),identifier="OI_145",unit="adu")
    m2 = Measurement(data=10.,uncertainty = StdDevUncertainty(2.),identifier="CI_609",unit="adu")
    m3 = Measurement(data=10.,uncertainty = StdDevUncertainty(1.5),identifier="CO_21",unit="adu")
    m4 = Measurement(data=100.,uncertainty = StdDevUncertainty(10.),identifier="CII_158",unit="adu")

    p = LineRatioFit(measurements = [m1,m2,m3,m4])
    print("num ratios:", p.ratiocount)
    print("modelfiles used: ", p._modelfilesUsed)
    p.run()
    p._modelset.get_ratio_elements(p.measurements)
    p.makeRatioOverlays(cmap='gray')
    p.plotRatiosOnModels()
    p._observedratios
