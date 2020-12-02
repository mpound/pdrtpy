from copy import deepcopy,copy
import warnings
  
import numpy as np
import numpy.ma as ma
import scipy.stats as stats

import matplotlib.figure
import matplotlib.colors as mpcolors
import matplotlib.cm as mcm
from matplotlib import ticker
from matplotlib.lines import Line2D

from astropy.io import fits
import astropy.wcs as wcs
import astropy.units as u
from astropy.units import UnitsWarning
from astropy.nddata import NDDataArray, CCDData, NDUncertainty, StdDevUncertainty, VarianceUncertainty, InverseVariance

from .plotbase import PlotBase
from ..measurement import Measurement
from .. import pdrutils as utils


class ModelPlot(PlotBase):
    """Class to plot models and optionally Measurements.  It does not LineRatioFit first.
    """
    def __init__(self,modelset,figure=None,axis=None):
        super().__init__(tool=None)
        self._modelset = modelset
        self._figure = figure
        self._axis = axis
        self._meascolor='#4daf4a'
        # color blind/friendly color cyle courtesy https://gist.github.com/thriveth/8560036
        self._CB_color_cycle = ['#377eb8', '#ff7f00', '#4daf4a',
                  '#f781bf', '#a65628', '#984ea3',
                  '#999999', '#e41a1c', '#dede00']
    """Init method

    """
    
    def plot(self,id,**kwargs):
        if '/' in id:
            self.ratio(id,**kwargs)
        else:
            self.intensity(id,**kwargs)

    def ratio(self,id,**kwargs):
        ms = self._modelset
        model = ms.get_model(id)
        kwargs_opts = {'title': ms.table.loc[id]["title"], 'colorbar':True}
        kwargs_opts.update(kwargs)
        self._plot_no_wcs(model,**kwargs_opts)

    def intensity(self,id,**kwargs):
        # shouldn't need separate model intensity as keyword would tell you.
        # Idea: Put a 'modeltyp' keyword in FITS header whether it is intensity ratio or intensity.
        ms = self._modelset
        meas = kwargs.get("measurements",None)
        model = ms.get_models([id],model_type="intensity")
        if meas is not None:
            if type(meas[0]) is not Measurement:
                raise TypeError("measurement keyword value must be a list of Measurements")
            if (model[id]._unit != meas[0].unit ):
                raise TypeError(f"Model and Measurement for {id} have different units: ({model._unit},{meas_unit})")
            if id != meas[0].id:
                msg = f"Identifiers of model {id} and supplied Measurement {meas[0].id} do not match. Plotting anyway."
                warnings.warn(msg)

        kwargs_opts = {'title': ms.table.loc[id]["title"], 'colorbar':True}
        kwargs_opts.update(kwargs)
        self._plot_no_wcs(model[id],**kwargs_opts)

    def _plot_no_wcs(self,data,header=None,**kwargs):
        '''generic plotting method for images with no WCS, used by other plot methods'''
        #print("KWARGS is ",kwargs)
        measurements= kwargs.pop("measurements",None)
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

        kwargs_opts = {'units' : None,
                       'image':True,
                       'colorbar': False,
                       'contours': True,
                       'label': False,
                       'title':None,
                       'xaxis_unit': None,
                       'yaxis_unit': None,
                       'legend': False
                       }

        kwargs_contour = {'levels': None, 
                          'colors': ['white'],
                          'linewidths': 1.0}


        # Merge in any keys the user provided, overriding defaults.
        kwargs_contour.update(kwargs)
        kwargs_opts.update(kwargs)

        #if self._tool is not None:
        #     if self._tool._modelnaxis is None and "NAXIS" not in _header:
        #         raise Exception("Image header/WCS has no NAXIS keyword")

        if "NAXIS" not in _header:
            raise Exception("Image header/WCS has no NAXIS keyword")
        else:
            _naxis = _header["NAXIS"]

        if _naxis == 2:
            if kwargs_opts['units'] is not None:
                k = utils.to(kwargs_opts['units'], data)
            else:
                k = data
        elif _naxis == 3:
            if kwargs_opts['units'] is not None:
                k = utils.to(kwargs_opts['units'], data[0,:,:])
            else:
                k = data[0,:,:]
        else:
            raise Exception("Unexpected NAXIS value: %d"%_naxis)

        km = ma.masked_invalid(k)
        #print("Masks equal? %s:"%np.all(k.mask==km.mask))
        if getattr(k,"mask",None) is not None:
            km.mask = np.logical_or(k.mask,km.mask)
        # make sure nans don't affect the color map
        min_ = np.nanmin(km)
        max_ = np.nanmax(km)

        kwargs_imshow = { 'origin': 'lower', 
                          'norm': 'simple',
                          'stretch': 'linear',
                          'vmin': min_, 
                          'vmax': max_,
                          'cmap': 'plasma',
                          'aspect': 'equal'}
 
        kwargs_subplot = {'nrows': 1,
                          'ncols': 1,
                          'index': 1,
                          'reset': True,
                          'constrained_layout': True
                         }

        # delay merge until min_ and max_ are known
        #print("plot kwargs 1: ",kwargs_imshow)
        kwargs_imshow.update(kwargs)
        #print("plot kwargs 2: ",kwargs_imshow)
        kwargs_imshow['norm']=self._get_norm(kwargs_imshow['norm'],km,
                                             kwargs_imshow['vmin'],kwargs_imshow['vmax'],
                                             kwargs_imshow['stretch'])

        kwargs_subplot.update(kwargs)
        # swap ncols and nrows in figsize to preserve aspect ratio
        kwargs_subplot['figsize'] = kwargs.get("figsize",(kwargs_subplot["ncols"]*5,kwargs_subplot["nrows"]*5))
        #print("subplot kwargs : ",kwargs_subplot)

        #print("Got non-default kwargs: ", kwargs)

        axidx = kwargs_subplot['index']-1
        if kwargs_subplot['reset']:
# @todo can probably consolidate this
            self._figure,self._axis = self._plt.subplots(kwargs_subplot['nrows'],kwargs_subplot['ncols'],
                                        figsize=kwargs_subplot['figsize'],
                                        subplot_kw={'aspect':kwargs_imshow['aspect']},
                                        constrained_layout=kwargs_subplot['constrained_layout'])

        #print(self._figure,self._axis)

        # Make sure self._axis is an array because we will index it below.
        if type(self._axis) is not np.ndarray:
            self._axis = np.array([self._axis])

        # When using ncols>1, either the index needs to be 2-d 
        # or the axis array needs to be 1-d.  This takes the second approach:
        if len(self._axis.shape) > 1:
            self._axis = self._axis.flatten()

        #if self._modelnaxis==2:
        #     ax1='1'
        #     ax2='2'
        #else:
        ax1='1'
        ax2='2'
            
        xstart=_header['crval'+ax1]
        xstop=xstart+_header['naxis'+ax1]*_header['cdelt'+ax1]
        ystart=_header['crval'+ax2]
        ystop=ystart+_header['naxis'+ax2]*_header['cdelt'+ax2]
        #print(xstart,xstop,ystart,ystop)
    
        # make the x and y axes.  Since the models are computed on a log grid, we
        # logarithmic ticks.
        y = 10**np.linspace(start=ystart, stop=ystop, num=_header['naxis'+ax2])
        x = 10**np.linspace(start=xstart, stop=xstop, num=_header['naxis'+ax1])
        locmaj = ticker.LogLocator(base=10.0, subs=(1.0, ),numticks=10)
        locmin = ticker.LogLocator(base=10.0, subs=np.arange(2, 10)*.1,numticks=10) 
        
        #allow unit conversion of density axis
        xax_unit = u.Unit(_header['cunit'+ax1])
        if kwargs_opts['xaxis_unit'] is not None:
            # Make density axis of the grid into a Quantity using the cunits from the grid header
            temp_x = x * xax_unit

            # Get desired unit from arguments
            xax_unit = u.Unit(kwargs_opts['xaxis_unit'])

            # Convert the unit-aware grid to the desired units and set X to the value (so it's no longer a Quantity)
            x = temp_x.to(xax_unit).value  

        # Set the x label appropriately, use LaTeX inline formatting
        xlab = r"{0} [{1:latex_inline}]".format(_header['ctype'+ax1],xax_unit)
        
        #allow unit conversion to cgs or Draine, for Y axis (FUV field):
        yax_unit = u.Unit(_header['cunit'+ax2])
        ytype = _header['ctype'+ax2]
        if kwargs_opts['yaxis_unit'] is not None:
            # Make FUV axis of the grid into a Quantity using the cunits from the grid header
            temp_y = y * yax_unit

            # Get desired unit from arguments; for special cases, use
            # the conventional symbol for the label (e.g. G_0 for Habing units)
            yunit = kwargs_opts['yaxis_unit']
            ytype = "log({0})".format(utils.get_rad(yunit))
            yax_unit = u.Unit(yunit)

            # Convert the unit-aware grid to the desired units and set Y to the value (so it's no longer a Quantity)
            y = temp_y.to(yax_unit).value  

        # Set the y label appropriately, use LaTeX inline formatting
        ylab = r"{0} [{1:latex_inline}]".format(ytype,yax_unit)
        #print("X axis min/max %.2e %.2e"%(x.min(),x.max()))
        #print("Y axis min/max %.2e %.2e"%(y.min(),y.max()))
        
        # Finish up axes details.
        self._axis[axidx].set_ylabel(ylab)
        self._axis[axidx].set_xlabel(xlab)
        self._axis[axidx].set_xscale('log')
        self._axis[axidx].set_yscale('log')
        self._axis[axidx].xaxis.set_major_locator(locmaj)
        self._axis[axidx].xaxis.set_minor_locator(locmin)
        self._axis[axidx].xaxis.set_minor_formatter(ticker.NullFormatter())

        if kwargs_opts['image']:
            # pass shading = auto to avoid deprecation warning
            # see https://matplotlib.org/3.3.0/gallery/images_contours_and_fields/pcolormesh_grids.html
            im = self._axis[axidx].pcolormesh(x,y,km,cmap=kwargs_imshow['cmap'],
                                              norm=kwargs_imshow['norm'],shading='auto')
            if kwargs_opts['colorbar']:
                self._figure.colorbar(im,ax=self._axis[axidx])
                #self._wcs_colorbar(im,self._axis[axidx])

        if kwargs_opts['contours']:
            if kwargs_contour['levels'] is None:
                # Figure out some autolevels 
                kwargs_contour['levels'] = self._autolevels(km,'log')

            # suppress warnings about unused keywords and potential error 
            # about cmap not being None. Also norm being a string will cause an error
            # in matplotlib==3.3.1+
            #@todo need a better solution for this, it is not scalable.
            for kx in ['units', 'image', 'contours', 'label', 'title',
                       'cmap','aspect','colorbar','reset', 'nrows', 'ncols',
                       'index','yaxis_unit','xaxis_unit','norm','legend','figsize',
                       'constrained_layout','figsize','stretch']:
                kwargs_contour.pop(kx,None)

            contourset = self._axis[axidx].contour(x,y,km.data, **kwargs_contour)
            #print(contourset.__dict__)

            if kwargs_opts['label']:
                drawn = self._axis[axidx].clabel(contourset,contourset.levels,inline=True,fmt='%1.2e')
                #print("drew %s"%drawn)

        if kwargs_opts['title'] is not None: 
            #if kwargs_opts['legend']:
            #    legend = self._axis[axidx].legend(handles=None,loc='upper center',title=kwargs_opts['title'])
            #else:
            self._axis[axidx].set_title(kwargs_opts['title'])

        if measurements is not None:
            lstyles = ['--','-','--']
            colors = [self._meascolor,self._meascolor,self._meascolor]
            for m in measurements:
                for i in range(0,3):
                    cset = self._axis[axidx].contour(x,y,k.data,levels=m.levels, 
                                                     linestyles=lstyles, colors=colors)

