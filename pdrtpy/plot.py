#todo: Look into seaborn https://seaborn.pydata.org
# Also https://docs.bokeh.org/en
# especially for coloring and style

import numpy as np
import numpy.ma as ma
import scipy.stats as stats

import matplotlib.figure
import matplotlib.colors as mpcolors
import matplotlib.cm as mcm
from matplotlib import ticker
from matplotlib.lines import Line2D
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.axes as maxes

from astropy.nddata.utils import Cutout2D
from astropy.io import fits
import astropy.wcs as wcs
import astropy.units as u
from astropy.nddata import NDDataArray, CCDData, NDUncertainty, StdDevUncertainty, VarianceUncertainty, InverseVariance
from astropy.visualization import simple_norm, ZScaleInterval , ImageNormalize
from astropy.visualization.stretch import SinhStretch,  LinearStretch
from matplotlib.colors import LogNorm

import pdrutils as utils

rad_title = dict()
rad_title['Habing'] = '$G_0$'
rad_title['Draine'] = '$\chi$'
rad_title['Mathis'] = 'ISRF$_{Mathis}$'

class PlotBase:
    def __init__(self,tool):
        import matplotlib.pyplot as plt
        self._plt = matplotlib.pyplot
        self._figure = None
        self._axis = None
        self._tool = tool
        #print("Done PlotBase")

    def _autolevels(self,data,steps='log',numlevels=None):
        # tip of the hat to the WIP autolevels code lev.
        # http://admit.astro.umd.edu/wip ,  wip/src/plot/levels.c
        # CVS at http://www.astro.umd.edu/~teuben/miriad/install.html
        #print(type(data))
        max_ =data.max()
        min_ = data.min()
        #print("autolev min %f max %f"%(min_,max_))
        if numlevels is None:
            numlevels = int(0.5+3*(np.log(max_)-np.log(min_))/np.log(10))
        #print("levels start %d levels"%numlevels)
        # force number of levels to be between 5 and 15
        numlevels = max(numlevels,5)
        numlevels = min(numlevels,15)
    
        if steps[0:3] == 'lin':
            slope = (max_ - min_)/(numlevels-1)
            levels = np.array([min_+slope*j for j in range(0,numlevels)])
        elif steps[0:3] == 'log':
            # if data minimum is non-positive (shouldn't happen for models),
            #, min_cut=min_,max_cut=max_, stretch='log', clip=False) start log contours at lgo10(1) = 0
            if min_ <= 0: min_=1
            slope = np.log10(max_/min_)/(numlevels - 1)
            levels = np.array([min_ * np.power(10,slope*j) for j in range(0,numlevels)])
        else:
           raise Exception("steps must be 'lin' or 'log'")
        print("Computed %d contour autolevels: %s"%(numlevels,levels))
        return levels
        
    def _zscale(self,image):
        # clip=False required or NaNs get max color value, see https://github.com/astropy/astropy/issues/8165
        norm= ImageNormalize(data=image,interval=ZScaleInterval(contrast=0.5),stretch=LinearStretch(),clip=False)
        return norm

    def _wcs_colorbar(self,image, axis, pos="right", width="10%",pad=0.15,orientation="vertical"):
        """Create a colorbar for a subplot with WCSAxes 
           (as opposed to matplolib Axes).  There are some side-effects of
           using WCS procjection that need to be ameliorated.  Also for 
           subplots, we want the colorbars to have the same height as the 
           plot, which is not the default behavior.

           Parameters:
              image - the mappable object for the plot. Must not be masked.
              axis - which Axes object for the plot
              pos - colorbar position: ["left"|"right"|"bottom"|"top"]. Default: right
              width - string, width of the colorbar in terms of percent width of the plot.
              pad - padding between colorbar and plot, in inches.
              orientation - string, ["vertical" | "horizontal" ]
        """
        divider = make_axes_locatable(axis)
        # See https://stackoverflow.com/questions/47060939/matplotlib-colorbar-and-wcs-projection
        cax = divider.append_axes(pos, size=width, pad=pad, axes_class=maxes.Axes)
        cax.yaxis.set_ticks_position(pos)
        return self._figure.colorbar(image,ax=axis,cax=cax)

class LineRatioPlot(PlotBase):
    """ Class to plot various results from PDR Toolbox model fitting """
    def __init__(self,tool,**kwargs):
        super().__init__(tool)
        self._figure = None
        self._axis = None
        self._xlim = [None,None]
        self._ylim = [None,None]
        self._plotkwargs = kwargs

    def chisq(self,xaxis,xpix,ypix):
        """Make a line plot of chisq as a function of G0 or n for a given pixel"""
        axes = {"G0":0,"n":1}
        axis = axes[xaxis] #yep key error if you do it wrong
            
    def modelratio(self,identifier,**kwargs):
        self._plot(self._tool._modelratios[identifier],kwargs)

    def observedratio(self,identifier,image=True,contours=False,levels=None):
        self._plot(data=self._tool._observedratios[identifier],units=u.dimensionless_unscaled,cmap='viridis',image=image,contours=contours,levels=levels,norm=None,title=self._tool._modelset.table.loc[identifier]["title"])

    def density(self,units='cm^-3',cmap='plasma',image=True, contours=False,levels=None,norm=None,**kwargs):
        #fancyunits=self._tool._density.unit.to_string('latex')
        fancyunits=u.Unit(units).to_string('latex')
        _title = 'n ('+fancyunits+')'
        self._plot(self._tool._density,units,cmap,image,contours,levels,norm,title=_title,**kwargs)

    def radiationField(self,units='Habing',cmap='plasma',image=True, contours=False,levels=None,norm=None,**kwargs):
        #fancyunits=self._tool._radiation_field.unit.to_string('latex')
        if units not in rad_title:
            fancyunits=u.Unit(units).to_string('latex')
            _title='Radiation Field ('+fancyunits+')'
        else:
            _title=rad_title[units]
        self._plot(self._tool._radiation_field,units,cmap,image,contours,levels,norm,title=_title,**kwargs)

    def plot_both(self,units=['Habing','cm^-3'],cmap='plasma',image=True, contours=False,levels=None,norm=None):
       self.radiationField(units=units[0],cmap=cmap,image=image,contours=contours,levels=levels,norm=norm,nrows=1,ncols=2,index=1,reset=True)
       self.density(units=units[1],cmap=cmap,image=image,contours=contours,levels=levels,norm=norm,nrows=1,ncols=2,index=2,reset=False)

    def _plot(self,data,units,cmap,image,contours,levels,norm,title,**kwargs):
        #@Todo raise exceptions 
        #@Todo separate imshow kwargs from other kwargs. See e.g.  https://fermipy.readthedocs.io/en/latest/_modules/fermipy/plotting.html
        k=utils.to(units,data)
        km = ma.masked_invalid(k)
        # make sure nans don't affect the color map
        min_ = np.nanmin(km)
        max_ = np.nanmax(km)
        #print(kwargs)
        nrows = kwargs.get("nrows",1)
        ncols = kwargs.get("ncols",1)
        index = kwargs.get("index",1)
        reset = kwargs.get("reset",True)
        figsize=kwargs.get("figsize",(ncols*5,ncols*5))
        #print(nrows,ncols,index,reset,figsize)
        axidx = index-1
        if reset:
            self._figure,self._axis = self._plt.subplots(nrows,ncols,figsize=figsize,subplot_kw={'projection':k.wcs,'aspect':'auto'},constrained_layout=True)
        # Make sure self._axis is an array because we will index it below.
        if type(self._axis) is not np.ndarray:
            self._axis = np.array([self._axis])
        if norm == 'simple':
            normalizer = simple_norm(km, min_cut=min_,max_cut=max_, stretch='log', clip=False)
        elif norm == 'zscale':
            normalizer = self._zscale(km)
        elif norm == 'log':
            normalizer = LogNorm()
        else: 
            normalizer = norm
        
        if image:
            current_cmap = mcm.get_cmap(cmap)
            current_cmap.set_bad(color='white',alpha=1)
            im=self._axis[axidx].imshow(km,origin='lower',norm=normalizer,cmap=cmap)
            self._wcs_colorbar(im,self._axis[axidx])
        if contours:
            if image==False: colors='black'
            else: colors='white'
            if levels is None:
                # Figure out some autolevels 
                steps='log'
                contourset = self._axis[axidx].contour(km, levels=self._autolevels(km,steps),colors=colors)
            else:
                contourset = self._axis[axidx].contour(km, levels=levels, colors=colors)
                # todo: add contour level labelling
                # See https://matplotlib.org/3.1.1/gallery/images_contours_and_fields/contour_label_demo.html
        if title is not None: self._axis[axidx].set_title(title)
        self._axis[axidx].set_xlabel(k.wcs.wcs.lngtyp)
        self._axis[axidx].set_ylabel(k.wcs.wcs.lattyp)
        
######################################################3
## FROM HERE DOWN NEEDS WORK FOR MULTIPIXEL MAPS
######################################################3
    def reducedChisq(self,cmap='plasma',image=True, contours=True,
                         levels=None,measurements=None):
        self._plot(self._tool._reducedChisq,cmap,image, contours,levels,measurements)
        
    def _oldplotChisq(self,cmap='plasma',image=True, contours=True, 
                  levels=None, measurements=None):
        self._plot(self._chisq,cmap,image, contours,levels,measurements)
       
    # only works for single pixe mapsx
    def _oldplot(self,datasource,cmap, image, contours, levels, measurements):
        if type(datasource) == str:
            k = fits.open(datasource)[0]
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
                

    def confidenceIntervals(self,image=True, cmap='plasma',contours=True,levels=None,reduced=False):
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
            

##############################################################################
class H2ExcitationPlot(PlotBase):
    def __init__(self,tool,**kwargs):
        super().__init__(tool)
        self.figure = None
        self.axis = None
        self._xlim = [None,None]
        self._ylim = [None,None]
        self._plotkwargs = kwargs

    def _plotimage(self,data,units,cmap,image,contours,levels,norm):
        k=utils.to(units,data)
        km = ma.masked_invalid(k)
        min_ = km.min()
        max_ = km.max()
        ax=self._plt.subplot(111,projection=k.wcs,aspect='equal')
        if norm == 'simple':
            normalizer = simple_norm(km, min_cut=min_,max_cut=max_, stretch='log', clip=false)
        elif norm == 'zscale':
            normalizer = self._zscale(km)
        else: 
            normalizer = norm
        
        if image: 
            current_cmap = mcm.get_cmap(cmap)
            current_cmap.set_bad(color='white',alpha=1)
            self._plt.imshow(km,cmap=current_cmap,origin='lower',norm=normalizer)
            self._plt.colorbar()
        if contours:
            if image==false: colors='black'
            else: colors='white'
            if levels is none:
                # figure out some autolevels 
                steps='log'
                contourset = ax.contour(km, levels=self._autolevels(km,steps),colors=colors)
            else:
                contourset = ax.contour(km, levels=levels, colors=colors)
                # todo: add contour level labelling
                # See https://matplotlib.org/3.1.1/gallery/images_contours_and_fields/contour_label_demo.html
        if title is not None: self._plt.title(title)
        self._plt.xlabel(k.wcs.wcs.lngtyp)
        self._plt.ylabel(k.wcs.wcs.lattyp)

    def plot_diagram(self,x,y,xsize,ysize,norm=True):
        cdavg = self._tool.average_column_density(norm,x,y,xsize,ysize,line=False)
        cdval = list(cdavg.values())*self._tool._cd_units
        energies = list(self._tool.energies(line=False).values())*u.K
        ax=self._plt.subplot(111)
        ax.scatter(x=energies,y=cdavg)
        

