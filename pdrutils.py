#!/usr/bin/env python
# coding: utf-8

# In[192]:


#todo: Look into seaborn https://seaborn.pydata.org
# especially for coloring and style
import itertools
import collections
from copy import deepcopy

import numpy as np
import scipy.stats as stats

from astropy.io import fits
import astropy.wcs as wcs
import astropy.units as u
from astropy.table import Table
from astropy.nddata import NDDataArray, CCDData, NDUncertainty, StdDevUncertainty, VarianceUncertainty, InverseVariance
from astropy.visualization import simple_norm, ZScaleInterval , ImageNormalize
from astropy.visualization.stretch import SinhStretch,  LinearStretch

#from skimage import data, exposure

import matplotlib.pyplot as plt
import matplotlib.colors as mpcolors
from matplotlib import ticker
from matplotlib.lines import Line2D

class PDRutils:
    def __init__(self,models,measurements=None):
        if type(models) == str:
            self._initialize_modelTable(models)
        else:
            self._modelTable = models
        if type(measurements) == dict:
            self._measurements = measurements
        else:
            self._init_measurements(measurements)
        # todo: store model file data rather than re-read everytime
        self._modelratios = None
        self._set_modelfilesUsed()
        self._observedratios = None
        self._chisq = None
        self._deltasq = None
        self._reducedChisq = None
    
    def _init_measurements(self,m):
        self._measurements = dict()
        for mm in m:
            self._measurements[mm.id] = mm

    def _check_shapes(self,d):
       #ugly
       s1 = d[list(d.keys())[0]].shape
       return np.all([m.shape == s1 for m in d.values()])

    def check_measurement_shapes(self):
       if self._measurements == None: return False
       return self._check_shapes(self._measurements)

    def check_ratio_shapes(self):
       if self._observedratios == None: return False
       return self._check_shapes(self._observedratios)
            
    def addMeasurement(self,m):
        '''Add a Measurement to internal dictionary used to compute ratios'''
        if self._measurements:
            self._measurements[m.id] = m
        else:
            self._init_measurements(m)
        self._set_modelfilesUsed()
        
    def removeMeasurement(self,id):
        '''Delete a measurement from the internal dictionary used to compute ratios.
           raises KeyError if id not in existing Measurements
        '''
        del self._measurements[id]
        self._set_modelfilesUsed()
   
    @property
    def measurements(self):
        '''return stored measurements as dictionary with Measurement IDs as keys'''
        return self._measurements
    
    @property
    def measurementIDs(self):
        '''return stored measurement IDs as `dict_keys` iterator'''
        return self._measurements.keys()
    
    def _make_default_table(self):
        ratiodict = {
        "OI_145/OI_63"   : "oioi",
        "OI_145/CII_158" : "o145cii",
        "OI_63/CII_158"  : "oicp",
        "CII_158/CI_609" : "ciici609",
        "CI_370/CI_609"  : "cici",
        "CII_158/CO_10"  : "ciico",
        "CI_609/CO_10"   : "cico",
        "CI_609/CO_21"   : "cico21",
        "CI_609/CO_32"   : "cico32",
        "CI_609/CO_43"   : "cico43",
        "CI_609/CO_54"   : "cico54",
        "CI_609/CO_65"   : "cico65",
        "CO_21/CO_10"    : "co2110",
        "CO_32/CO_10"    : "co3210",
        "CO_32/CO_21"    : "co3221",
        "CO_43/CO_21"    : "co4321",
        "CO_65/CO_10"    : "co6510",
        "CO_65/CO_21"    : "co6521",
        "CO_65/CO_54"    : "co6554",
        "CO_76/CO_10"    : "co7610",
        "CO_76/CO_21"    : "co7621",
        "CO_76/CO_43"    : "co7643",
        "CO_76/CO_54"    : "co7654",
        "CO_76/CO_65"    : "co7665",
        "CO_87/CO_54"   : "co8754",
        "CO_87/CO_65"   : "co8765",
        "CO_98/CO_54"   : "co9854",
        "CO_98/CO_65"   : "co9865",
        "CO_109/CO_54"   : "co10954",
        "CO_109/CO_65"   : "co10965",
        "CO_1110/CO_54"   : "co111054",
        "CO_1110/CO_65"   : "co111065",
        "CO_1211/CO_54"   : "co121154",
        "CO_1211/CO_65"   : "co121165",
        "CO_1312/CO_54"   : "co131254",
        "CO_1312/CO_65"   : "co131265",
        "CO_1413/CO_54"   : "co141354",
        "CO_1413/CO_65"   : "co141365",
        "OI_63+CII_158/FIR"     : "fir",
        "OI_145+CII_158/FIR"  : "firoi145",
        "SIII_Z1/FEII_Z1"  : "siii35feii26z1",
        "SIII_Z3/FEII_Z3"  : "siii35feii26z3",
        "H200S1_Z1/H200S0_Z1" : "h200s1s0z1",
        "H200S1_Z3/H200S0_Z3" : "h200s1s0z3",
        "H200S2_Z1/H200S0_Z1" : "h200s2s0z1",
        "H200S2_Z3/H200S0_Z3" : "h200s2s0z3",
        "H200S2_Z1/H200S1_Z1" : "h200s2s1z1",
        "H200S2_Z3/H200S1_Z3" : "h200s2s1z3",
        "H200S3_Z1/H200S1_Z1" : "h200s3s1z1",
        "H200S3_Z3/H200S1_Z3" : "h200s3s1z3",
        "H200S1_Z1/SIII_Z1" : "h200s1siiiz1",
        "H200S1_Z3/SIII_Z3" : "h200s1siiiz3",
        "H200S2_Z1/SIII_Z1" : "h200s2siiiz1",
        "H200S2_Z3/SIII_Z3" : "h200s2siiiz3",
        "H264Q1_Z1/H210S1_Z1" : "h264q110s1z1",
        "H264Q1_Z3/H210S1_Z3" : "h264q110s1z3"
        }
        b = list()
        for r in ratiodict:
            nd = r.split("/")
            if ("Z3" in r):
                z=3
            else:
                z=1
            b.append((nd[0],nd[1],r,ratiodict[r]+"web",z))
            
        t = Table(rows=b,names=("numerator","denominator","label","filename","z"))
        t.add_index("label")
        t.write("current_models.tab",format="ascii.ipac",overwrite=True)
  
    def _initialize_modelTable(self,filename):
        """initialize models from an IPAC format ASCII file"""
        # Todo: add LaTeX labels as a column for more stylish plotting
        self._modelTable=Table.read(filename,format="ascii.ipac")
        self._modelTable.add_index("label")
        
    def supportedLines(self):
        '''Return a `set` of lines and continuum recognized by this class (i.e., that have been modeled by us)'''
        return set(np.append(self._modelTable["numerator"].data,self._modelTable["denominator"].data))

    def find_ratio_elements(self,m):
        """Return an iterator of valid numerator,denominator pairs in 
        dict format for the given list of measurement IDs
        """
        if not isinstance(m, collections.abc.Iterable) or isinstance(m, (str, bytes)) :
            raise Exception("m must be an array of strings")
            
        for q in itertools.product(m,m):
            s = q[0]+"/"+q[1]
            z = dict()
            if s in self._modelTable["label"]:
                z={"numerator":self._modelTable.loc[s]["numerator"],
                   "denominator":self._modelTable.loc[s]["denominator"]}
                yield(z)
                
    def get_ratio_elements(self,m):   
        """Return a list of valid numerator,denominator pairs in dict format for the 
        given list of measurement IDs
        """
        if not isinstance(m, collections.abc.Iterable) or isinstance(m, (str, bytes)) :
            raise Exception("m must be an array of strings")
        k = list()   
        for q in itertools.product(m,m):
            s = q[0]+"/"+q[1]
            if s in self._modelTable["label"]:
                z={"numerator":self._modelTable.loc[s]["numerator"],
                   "denominator":self._modelTable.loc[s]["denominator"]}
                k.append(z)        
        return k
                 
    def find_pairs(self,m):
        """Return an iterator of model ratios labels for the given list of measurement IDs"""
        if not isinstance(m, collections.abc.Iterable) or isinstance(m, (str, bytes)) :
            raise Exception("m must be an array of strings")
            
        for q in itertools.product(m,m):
            s = q[0]+"/"+q[1]
            if s in self._modelTable["label"]:
                yield(s)
    
    def find_files(self,m,ext="fits"):
        """Return an iterator of model ratio files for the given list of measurement IDs"""
        if not isinstance(m, collections.abc.Iterable) or isinstance(m, (str, bytes)):
            raise Exception("m must be an array of strings")
        for q in itertools.product(m,m):
            s = q[0]+"/"+q[1]
            if s in self._modelTable["label"]:
                tup = (s,self._modelTable.loc[s]["filename"]+"."+ext)
                #yield(self._modelTable.loc[s]["filename"]+"."+ext)
                yield(tup)
           
    def _set_modelfilesUsed(self):
        self._modelfilesUsed = dict()
        for x in self.find_files(self.measurementIDs):
            self._modelfilesUsed[x[0]]=x[1]
     
    @property
    def ratiocount(self):
        return self._ratiocount(self.measurementIDs)
    
    def _ratiocount(self,m):
        """Return the number of model ratios found for the given list of measurement IDs"""
        # since find_files is a generator, we can't use len(), so do this sum.
        # See https://stackoverflow.com/questions/393053/length-of-generator-output
        return(sum(1 for _ in self.find_files(m)))
                
    def read_fits(self,m):
        """Given a list of measurement IDs, find and open the FITS files that have matching ratios
           and populate the _modelratios dictionary
        """
        d = "models/"
        self._modelratios = dict()
        for (k,p) in self.find_files(m):
            self._modelratios[k] = fits.open(d+p)
            
    def read_ccd(self,unit):
        """Test method to use astropy's CCDdata as a storage mechanism for our model 
           ratio or observer data FITS files.
            Parameters: 
                 m - list of measurement IDS (string)
                 unit - units of the data (string)
        """
        d = "models/"
        self._modelratios = dict()
        for (k,p) in self.find_files(self.measurementIDs):
            self._modelratios[k] = CCDData.read(d+p,unit=unit)
    
    def doit(self):
        #self.read_fits(mids)
        self.read_ccd(unit='adu')
        self.computeValidRatios()
        self.computeDeltaSq()
        self.computeChisq()
        self.writeChisq()
      
    def computeValidRatios(self):
        if not self.check_measurement_shapes():
            raise TypeError("Measurement maps have different dimensions")

        z = self.find_ratio_elements(self.measurementIDs)
        self._observedratios = dict()
        for p in z:
            label = p["numerator"]+"/"+p["denominator"]
            # deepcopy workaround for bug: https://github.com/astropy/astropy/issues/9006
            self._observedratios[label] = deepcopy(self._measurements[p["numerator"]])/deepcopy(self._measurements[p["denominator"]])
        
    
    def computeDeltaSq(self):
        #@todo perhaps don't store _modelratios but have them be a return value of read_fits.
        # reasoning is that if we use the same PDRUtils object to make multiple computations, we
        # have to be careful to clear _modelratios each time.
        
        if not self._modelratios: # empty list or None
            raise Exception("No model data ready.  You need to call read_fits")
        
        self._deltasq = dict()
        for r in self._observedratios:
            _z = self._modelratios[r].multiply(-1.0)
            _z._data = _z._data + self._observedratios[r].flux
            _q = _z.divide(self._observedratios[r].error)
            self._deltasq[r] = _q.multiply(_q)

    def computeDeltaSqMap(self):
        #@todo perhaps don't store _modelratios but have them be a return value of read_fits.
        # reasoning is that if we use the same PDRUtils object to make multiple computations, we
        # have to be careful to clear _modelratios each time.
        
        if not self._modelratios: # empty list or None
            raise Exception("No model data ready.  You need to call read_fits")
        
        self._deltasq = dict()
        for r in self._observedratios:
            sz = self._modelratios[r].size
            _z = np.reshape(self._modelratios[r],sz)

            ff = list()
            for pix in _z:
                _q = (self._observedratios[r].flux - pix)/self._observedratios[r].error
                ff.append(_q*_q)

            newshape = np.hstack((self._modelratios[r].shape,self._observedratios[r].shape))
            #newshape = np.hstack((self._observedratios[r].shape,self._modelratios[r].shape))
            _qq = np.reshape(ff,newshape)
            self._deltasq[r] = CCDData(_qq,unit="adu",wcs=self._observedratios[r].wcs,meta=self._observedratios[r].meta)
        
    def computeChisq(self):
        sumary = sum((self._deltasq[r]._data for r in self._deltasq))
        self._dof = len(self._deltasq) - 1
        k = list(self._deltasq.keys())[0]
        self._chisq = CCDData(sumary,unit='adu',wcs=self._deltasq[k].wcs,meta=self._deltasq[k].meta)
        self._reducedChisq =  self._chisq.divide(self._dof)
    
    def _zscale(self,image):
        norm= ImageNormalize(image.data,ZscaleInterval(),stretch=LinearStretch())
        return norm
    
    def plotReducedChisq(self,cmap='plasma',image=True, contours=True,
                         levels=None,measurements=None):
        self._plot(self_reducedChisq,cmap,image, contours,levels,measurements)
        
    def plotChisq(self,cmap='plasma',image=True, contours=True, 
                  levels=None, measurements=None):
        self._plot(self._chisq,cmap,image, contours,levels,measurements)
        
    def writeChisq(self,file="chisq.fits",rfile="rchisq.fits"):
        self._chisq.write(file,overwrite=True)
        self._reducedChisq.write(rfile,overwrite=True)
       
    def _plot(self,datasource,cmap, image, contours, levels, measurements):
        if type(datasource) == str:
            k = fits.open(file)[0]
        else:
            k=datasource.to_hdu()[0]
        min_ = k.data.min()
        max_ = k.data.max()
        ax=plt.subplot(111)
        ax.set_aspect('equal')
        normalizer = simple_norm(k.data, min_cut=min_,max_cut=max_, stretch='log', clip=False)
        xstart=k.header['crval1']
        xstop=xstart+k.header['naxis1']*k.header['cdelt1']
        ystart=k.header['crval2']
        ystop=ystart+k.header['naxis2']*k.header['cdelt2']
        #print(xstart,xstop,ystart,ystop)
    
        y = 10**np.linspace(start=ystart, stop=ystop, num=k.header['naxis2'])
        x = 10**np.linspace(start=xstart, stop=xstop, num=k.header['naxis1'])
        ax.set_xscale('log')
        ax.set_yscale('log')
        #if True:
        #    chihist = exposure.equalize_hist(k.data)
        #    ax.pcolormesh(x,y,chihist,cmap=cmap)
        if image:
            ax.pcolormesh(x,y,k.data,cmap=cmap,norm=normalizer)
        
        locmaj = ticker.LogLocator(base=10.0, subs=(1.0, ),numticks=10)
        ax.xaxis.set_major_locator(locmaj)
        locmin = ticker.LogLocator(base=10.0, subs=np.arange(2, 10)*.1,numticks=10) 
        ax.xaxis.set_minor_locator(locmin)
        ax.xaxis.set_minor_formatter(ticker.NullFormatter())
        #todo: allow unit conversion to cgs or Draine,  1 Draine = 1.69 G0 (Habing)
        # 1 Habing = 5.29x10−14 erg cm−3
        ylab = 'Log($G_0$) [Habing]'
        ss = 'Log($P_{th}/k$) [cm$^{-2}$]'
        xlab = 'Log(n) [cm$^{-3}$]'
        ax.set_ylabel(ylab)
        ax.set_xlabel(xlab)

        if contours:
            if image==False: colors='black'
            else: colors='white'
            if levels is None:
                # Figure out some autolevels 
                # ------------------------------------------
                #  below resulted in poorly spaced levels
                # ------------------------------------------
                #if ( max_ - min_ ) < 101.0:
                #    steps = 'lin'
                #else:
                #    steps = 'log'
                # ------------------------------------------
                steps='log'
                contourset = ax.contour(x,y,k.data, levels=self._autolevels(k.data,steps),colors=colors)
            else:
                contourset = ax.contour(x,y,k.data, levels=levels, colors=colors)
                # todo: add contour level labelling
                # See https://matplotlib.org/3.1.1/gallery/images_contours_and_fields/contour_label_demo.html
        if measurements is not None:
            for m in measurements:
            #todo: use matplotlib contourf to shade the area between +/- error rather than use dashed lines
                lstyles = ['--','-','--']
                colors = [self._ratiocolor,self._ratiocolor,self._ratiocolor]
                for i in range(0,3):
                    #print(lstyles)
                    #print(len(x))
                    #print(len(y))
                    #print(len(k.data))
                    #print(colors)
                    cset = ax.contour(x,y,k.data,levels=m.levels,
                           linestyles=lstyles, colors=colors)
                
    def getBestnG0(self):
        if self._chisq is None: return [None,None]
        min_ary = np.flip(divmod(self._reducedChisq.data.argmin(),self._chisq.data.shape[1]))+1
        print("min indices (n,G0) = ",min_ary)
        #h = fits.open("chisq.fits")[0].header
        #logn = h['crval1']+(min_ary[0]-h['crpix1'])*h['cdelt1']
        #logg0 = h['crval2']+(min_ary[1]-h['crpix2'])*h['cdelt2']
        logn,logg0 = self._chisq.wcs.all_pix2world(min_ary,0)
        g0 = 10.0**logg0
        n  = 10.0**logn
        return [n,g0]
        
    def plotConfidenceIntervals(self,image=True, cmap='plasma',contours=True,levels=None,reduced=False):
        if reduced:  chi2_stat = stats.distributions.chi2.cdf(self._reducedChisq.data,self._dof)
        else:       chi2_stat = stats.distributions.chi2.cdf(self._chisq.data,self._dof)
        normalizer = simple_norm(chi2_stat, stretch='log', clip=False)
        if image: plt.imshow(chi2_stat*100,aspect='equal',cmap=cmap,norm=normalizer)
        print(100*chi2_stat.min(),100*chi2_stat.max())
        if contours:
            if levels==None:
                levels = [68.,80., 95., 99.]
            plt.contour(chi2_stat*100,colors='black',levels=sorted(levels))
    
    def plotRatiosOnModels(self):
        CB_color_cycle = ['#377eb8', '#ff7f00', '#4daf4a',
                  '#f781bf', '#a65628', '#984ea3',
                  '#999999', '#e41a1c', '#dede00']
        i =0 
        for x in self._modelfilesUsed:
            self._ratiocolor = CB_color_cycle[i]
            self._plot('models/'+self._modelfilesUsed[x],cmap='plasma',
                       measurements=[self._observedratios[x]],image=False,contours=False,levels=None)
            i = i+1
        # do some sort of legend
        lines = [Line2D([0], [0], color=c, linewidth=3, linestyle='-') for c in CB_color_cycle[0:i]]
        labels = list(self._modelfilesUsed.keys())
        plt.legend(lines, labels)
        plt.show()

    def makeRatioOverlays(self,cmap='plasma'):
        for x in self._modelfilesUsed:
            #self._ratiocolor='silver'
            self._ratiocolor='#4daf4a'
            self._plot('models/'+self._modelfilesUsed[x],cmap=cmap,
                       measurements=[self._observedratios[x]],image=True,contours=True,levels=None)
            plt.title(x)
            plt.show()
            plt.clf()
            
    def _autolevels(self,data,steps='log',numlevels=None):
        # tip of the hat to the WIP autolevels code lev.
        # http://admit.astro.umd.edu/wip ,  wip/src/plot/levels.c
        # CVS at http://www.astro.umd.edu/~teuben/miriad/install.html
        max_ =data.max()
        min_ = data.min()
        if numlevels is None:
            numlevels = int(0.5+3*(np.log(max_)-np.log(min_))/np.log(10))
        #print("levels start %d levels"%numlevels)
        # force number of levels to be between 5 and 15
        numlevels = max(numlevels,5)
        numlevels = min(numlevels,15)
    
        #print("Autolevels got %d levels"%numlevels)
        if steps[0:3] == 'lin':
            slope = (max_ - min_)/(numlevels-1)
            levels = np.array([min_+slope*j for j in range(0,numlevels)])
        elif steps[0:3] == 'log':
            # if data minimum is non-positive (shouldn't happen for models),
            # start log contours at lgo10(1) = 0
            if min_ <= 0: min_=1
            slope = np.log10(max_/min_)/(numlevels - 1)
            levels = np.array([min_ * np.power(10,slope*j) for j in range(0,numlevels)])
        else:
           raise Exception("steps must be 'lin' or 'log'")
        return levels
        
    
    #def plotChisq2(self,cmap='plasma',contours=True,levels=None):
    #    self._chisq.write("chisq.fits",overwrite=True)
    #    min_ = self._chisq.data.min()
    #    max_ = self._chisq.data.max()
     #   gc.show_colorscale(cmap=cmap,vmin=min_, vmax=max_,stretch='log')
    #    gc.axis_labels.set_ytext('Log($G_0$) [Habing]')
    #    gc.axis_labels.set_xtext('Log(n) [cm$^{-3}$] ')
     #   if contours:
    #        if levels is None:
    #            levels_ = np.array([2.,6.,12.,18.])*min_
    #            gc.show_contour(colors='white',levels=levels_)
     #       else:
     #           gc.show_contour(colors='white',levels=levels)
    
    def testme(self,min_,max_,numlevels):
        minsv = None
        if min_ < 0: 
            minsv = min_
            max_ = max_ + 1 - min_
            min_ = 1
        slope = np.log10(max_/min_)/(numlevels -1 )
        levels = np.array([min_ * np.power(10,slope*j) for j in range(0,numlevels)])
        print(levels)
        if minsv: 
            print("shift is ",(-1 - np.log10(np.abs(minsv))))
            levels = levels - 1 - np.log10(np.abs(minsv))
            print(levels)

                


if __name__ == "__main__":
    from measurement import Measurement 
    m1 = Measurement(data=[30],uncertainty = StdDevUncertainty([5.]),identifier="OI_145",unit="adu")
    m2 = Measurement(data=10.,uncertainty = StdDevUncertainty(2.),identifier="CI_609",unit="adu")
    m3 = Measurement(data=10.,uncertainty = StdDevUncertainty(1.5),identifier="CO_21",unit="adu")
    m4 = Measurement(data=100.,uncertainty = StdDevUncertainty(10.),identifier="CII_158",unit="adu")

    p = PDRutils("current_models.tab",measurements = [m1,m2,m3,m4])
    print("num ratios:", p.ratiocount)
    print("modelfiles used: ", p._modelfilesUsed)
    p.doit()
    p.get_ratio_elements(p.measurements)
    p.makeRatioOverlays(cmap='gray')
    p.plotRatiosOnModels()
    p._observedratios
