#todo: 
# document
#
# normalize the **kwargs arguments
# separate imshow kwargs from other kwargs. See e.g.  https://fermipy.readthedocs.io/en/latest/_modules/fermipy/plotting.html
#https://fermipy.readthedocs.io/en/latest/_modules/fermipy/plotting.html#ImagePlotter.plot
#
# raise appropriate exceptions 
#
# Look into seaborn https://seaborn.pydata.org
# Also https://docs.bokeh.org/en
# especially for coloring and style

from copy import deepcopy

import numpy as np
import numpy.ma as ma
import scipy.stats as stats

import matplotlib.figure
import matplotlib.colors as mpcolors
import matplotlib.cm as mcm
from matplotlib import ticker
from matplotlib.lines import Line2D

from astropy.nddata.utils import Cutout2D
from astropy.io import fits
import astropy.wcs as wcs
import astropy.units as u
from astropy.nddata import NDDataArray, CCDData, NDUncertainty, StdDevUncertainty, VarianceUncertainty, InverseVariance

from .plotbase import PlotBase
from ..pdrutils import to

rad_title = dict()
rad_title['Habing'] = '$G_0$'
rad_title['Draine'] = '$\chi$'
rad_title['Mathis'] = 'ISRF$_{Mathis}$'

class LineRatioPlot(PlotBase):
    """ Class to plot various results from PDR Toolbox model fitting """
    def __init__(self,tool,**kwargs):
        super().__init__(tool)
        self._figure = None
        self._axis = None
        self._xlim = [None,None]
        self._ylim = [None,None]
        self._plotkwargs = kwargs
        self._ratiocolor=[]
        self._CB_color_cycle = ['#377eb8', '#ff7f00', '#4daf4a',
                  '#f781bf', '#a65628', '#984ea3',
                  '#999999', '#e41a1c', '#dede00']

    def modelratio(self,identifier,**kwargs):
        self._plot(self._tool._modelratios[identifier],kwargs)

    def observedratio(self,identifier,image=True,contours=False,levels=None):
        self._plot(data=self._tool._observedratios[identifier],units=u.dimensionless_unscaled,cmap='viridis',image=image,contours=contours,levels=levels,norm=None,title=self._tool._modelset.table.loc[identifier]["title"])

    def density(self,units='cm^-3',cmap='plasma',image=True, contours=False,levels=None,norm=None,**kwargs):
        #fancyunits=self._tool._density.unit.to_string('latex')
        fancyunits=u.Unit(units).to_string('latex')
        _title = 'n ('+fancyunits+')'
        self._plot(self._tool._density,units,cmap,image,contours,levels,norm,title=_title,**kwargs)

    def radiation_fiield(self,units='Habing',cmap='plasma',image=True, contours=False,levels=None,norm=None,**kwargs):
        #fancyunits=self._tool._radiation_field.unit.to_string('latex')
        if units not in rad_title:
            fancyunits=u.Unit(units).to_string('latex')
            _title='Radiation Field ('+fancyunits+')'
        else:
            _title=rad_title[units]
        self._plot(self._tool._radiation_field,units,cmap,image,contours,levels,norm,title=_title,**kwargs)

    #def chisq(self,xaxis,xpix,ypix):
    #    """Make a line plot of chisq as a function of G0 or n for a given pixel"""
    #    axes = {"G0":0,"n":1}
    #    axis = axes[xaxis] #yep key error if you do it wrong
    #        
    def chisq(self,cmap='plasma',image=True, contours=True,levels=None,labelcont=True,norm='zscale',title=r'$\chi^2$'):           
        if len(self._tool._chisq.shape) != 2:
            raise NotImplementedError("Plotting of chisq is not yet implemented for maps")
        title=r'$\chi^2$'
        self._plot_no_wcs(data=self._tool._chisq,header=None,cmap=cmap,image=image,contours=contours,levels=levels,labelcont=labelcont,norm=norm,title=title,units=None)

    def reduced_chisq(self,cmap='plasma',image=True, contours=True,labelcont=True,levels=None,norm='zscale',title=r'$\chi_\nu^2$'):
        if len(self._tool._chisq.shape) != 2:
            raise NotImplementedError("Plotting of chisq is not yet implemented for maps")
        self._plot_no_wcs(self._tool._reduced_chisq,header=None,cmap=cmap,image=image,contours=contours,levels=levels,labelcont=labelcont,norm=norm,title=title,units=None)

    def plot_both(self,units=['Habing','cm^-3'],cmap='plasma',image=True, contours=False,levels=None,norm=None):
       self.radiationField(units=units[0],cmap=cmap,image=image,contours=contours,levels=levels,norm=norm,nrows=1,ncols=2,index=1,reset=True)
       self.density(units=units[1],cmap=cmap,image=image,contours=contours,levels=levels,norm=norm,nrows=1,ncols=2,index=2,reset=False)

    def _plot(self,data,units,cmap,image,contours,levels,labelcont,norm,title,**kwargs):
        if units is not None:
            k = to(units,data)
        else:
            k = data
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

        normalizer=self._get_norm(norm,km,min_,max_)
        
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
            if labelcont:
                self._axis[axidx].clabel(contourset,contourset.levels,inline=True,fmt='%1.1e')
        if title is not None: self._axis[axidx].set_title(title)
        if k.wcs is not None:
            self._axis[axidx].set_xlabel(k.wcs.wcs.lngtyp)
            self._axis[axidx].set_ylabel(k.wcs.wcs.lattyp)
        
       
    # only works for single pixe mapsx
    def _plot_no_wcs(self,data,header,units,cmap,image,contours,levels,labelcont,norm,title,**kwargs):
        measurements= kwargs.get("measurements",None)
        _dataheader = getattr(data,"header",None)
        if _dataheader is None  and header is None:
            raise Exception("Either your data must have a header dictionary or you must provide one via the header parameter")
        # input header supercedes data header, under assumption user knows what they are doing.
        if header is not None: 
            _header = deepcopy(header)
        else:
            _header = deepcopy(_dataheader)
            # CRxxx might be in wcs and not in header
            if data.wcs is not None:
                _header.update(data.wcs.to_header())
        
        if units is not None:
            k = to(units,data)
        else:
            k = data
        km = ma.masked_invalid(k)
        # make sure nans don't affect the color map
        min_ = np.nanmin(km)
        max_ = np.nanmax(km)
        
        nrows = kwargs.get("nrows",1)
        ncols = kwargs.get("ncols",1)
        index = kwargs.get("index",1)
        reset = kwargs.get("reset",True)
        figsize=kwargs.get("figsize",(ncols*5,ncols*5))
        #print(nrows,ncols,index,reset,figsize)
        axidx = index-1
        if reset:
            self._figure,self._axis = self._plt.subplots(nrows,ncols,figsize=figsize,subplot_kw={'aspect':'auto'},constrained_layout=True)

        # Make sure self._axis is an array because we will index it below.
        if type(self._axis) is not np.ndarray:
            self._axis = np.array([self._axis])

        normalizer = self._get_norm(norm,k,min_,max_)
        xstart=_header['CRVAL1']
        xstop=xstart+_header['naxis1']*_header['cdelt1']
        ystart=_header['crval2']
        ystop=ystart+_header['naxis2']*_header['cdelt2']
        #print(xstart,xstop,ystart,ystop)
    
        y = 10**np.linspace(start=ystart, stop=ystop, num=_header['naxis2'])
        x = 10**np.linspace(start=xstart, stop=xstop, num=_header['naxis1'])
        locmaj = ticker.LogLocator(base=10.0, subs=(1.0, ),numticks=10)
        locmin = ticker.LogLocator(base=10.0, subs=np.arange(2, 10)*.1,numticks=10) 
        xlab = _header['ctype1'] + ' ['+_header['cunit1']+']'
        ylab = _header['ctype2'] + ' ['+_header['cunit2']+']'
        self._axis[axidx].set_ylabel(ylab)
        self._axis[axidx].set_xlabel(xlab)


        self._axis[axidx].set_xscale('log')
        self._axis[axidx].set_yscale('log')
        self._axis[axidx].xaxis.set_major_locator(locmaj)
        self._axis[axidx].xaxis.set_minor_locator(locmin)
        self._axis[axidx].xaxis.set_minor_formatter(ticker.NullFormatter())
        if image:
            self._axis[axidx].pcolormesh(x,y,km,cmap=cmap,norm=normalizer)
    #todo: allow unit conversion to cgs or Draine
    
        if contours:
            colors = kwargs.pop("colors",'black')
            if levels is None:
                steps='log'
                contourset = self._axis[axidx].contour(x,y,k.data, levels=self._autolevels(k.data,steps),colors=colors)
            else:
                contourset = self._axis[axidx].contour(x,y,k.data, levels=levels, colors=colors)
            if labelcont:
                self._axis[axidx].clabel(contourset,contourset.levels,inline=True,fmt='%1.1e')

        if title is not None: self._axis[axidx].set_title(title)

            # todo: add contour level labelling
            # See https://matplotlib.org/3.1.1/gallery/images_contours_and_fields/contour_label_demo.html
        if measurements is not None:
            for m in measurements:
            #todo: use matplotlib contourf to shade the area between +/- error rather than use dashed lines
                lstyles = ['--','-','--']
                colors = [self._ratiocolor,self._ratiocolor,self._ratiocolor]
                for i in range(0,3):
                    cset = self._axis[axidx].contour(x,y,k.data,levels=m.levels, linestyles=lstyles, colors=colors)
                
######################################################
## FROM HERE DOWN NEEDS WORK FOR MULTIPIXEL MAPS
######################################################

    def confidence_intervals(self,cmap='plasma',image=True,contours=True,levels=None,labelcont=False,norm='simple',title="Confidence Intervals",**kwargs):
        chidata = self._tool._chisq.data
        chi2_stat = stats.distributions.chi2.cdf(chidata,self._tool._dof)
        if levels==None: levels = [50., 68., 80., 95., 99.]
        if image:
            self._plot_no_wcs(data=chi2_stat*100,header=self._tool._chisq.header,cmap=cmap,image=image,contours=contours,levels=sorted(levels), labelcont=labelcont,title=title,norm=norm,units=None,**kwargs)
        #print(100*chi2_stat.min(),100*chi2_stat.max())
    
    def overlay_all_ratios(self,**kwargs):
        i =0 
        ncols = 3
        nrows = int(round(self._tool.ratiocount/3+0.49,0))
        reset = True
        for key,val in self._tool._modelratios.items():
            self._ratiocolor = self._CB_color_cycle[i]
            o = self._tool._observedratios[key] 
            if i > 0: reset=False
            self._plot_no_wcs(val,cmap='plasma', measurements=[o],image=False,contours=False,levels=None, units=None,norm=None,title=None,labelcont=False,header=None,reset=reset,**kwargs)
            i = i+1
        # do some sort of legend
        lines = [Line2D([0], [0], color=c, linewidth=3, linestyle='-') for c in self._CB_color_cycle[0:i]]
        labels = list(self._tool._model_files_used.keys())
        self._plt.legend(lines, labels)

    def ratios_on_models(self,**kwargs):
        for key,val in self._tool._modelratios.items():
            m = self._tool._model_files_used[key]
            o = self._tool._observedratios[key]
            self._ratiocolor='#4daf4a'
            title = key + " model (Observed ratio indicated)"
            self._plot_no_wcs(val,cmap='plasma', measurements=[o],image=True,contours=True,levels=None, units=None,norm='zscale',title=title,labelcont=False,header=None,**kwargs)
            

