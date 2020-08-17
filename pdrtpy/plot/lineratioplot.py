#todo: 
# FIGURE MANAGEMENT -- SAVE FIG! e.g. name a figure and save it by name.
# 
# Fix the rows/cols issue in ratios_on_models and overlay_all_ratios
#
# keywords for show_both need to be arrays. ugh.
# allo levels to be percent?
#
# Allow axis units to be something other than model units
#
#use matplotlib contourf to shade the area between +/- error rather than use dashed lines
#
# continue to raise appropriate exceptions 
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
    """Class to plot various results from PDR Toolbox model fitting.
    

    :Keyword Arguments:

    To manage the plots, the methods in this class take keywords (\*\*kwargs) that turn on or off various options, specify plot units, or map to matplotlib's :meth:`~matplotlib.axes.Axes.plot`, :meth:`~matplotlib.axes.Axes.imshow`, :meth:`~matplotlib.axes.Axes.contour` keywords.  The methods have reasonable defaults, so try them with no keywords to see what they do before modifying keywords.

     * *units* (``str`` or :class:`astropy.units.Unit`) data units to use in the plot. This can be either a string such as, 'cm^-3' or 'Habing', or it can be an :class:`astropy.units.Unit`.  Data will be converted to the desired unit.   Note these are **not** the axis units, but the image data units.  Modifying axis units is not yet implemented.

     * *image* (``bool``) whether or not to display the image map (imshow). 

     * *cmap* (``str``) colormap name, Default: 'plasma' 

     * *colorbar* (``str``) whether or not to display colorbar

     * *colors* (``str``) color of the contours. Default: 'whitecolor of the contours. Default: 'white'

     * *contours* (``bool``), whether or not to plot contours

     * *label* (``bool``), whether or not to label contours 

     * *linewidths* (``float or sequence of float``), the line width in points, Default: 1.0

     * *levels* (``int`` or array-like) Determines the number and positions of the contour lines / regions.  If an int n, use n data intervals; i.e. draw n+1 contour lines. The level heights are automatically chosen.  If array-like, draw contour lines at the specified levels. The values must be in increasing order.  

     * *norm* (``str`` or :mod:`astropy.visualization` normalization object) The normalization to use in the image. The string 'simple' will normalize with :func:`~astropy.visualization.simple_norm` with a log stretch and 'zscale' will normalize with IRAF's zscale algorithm.  See :class:`~astropy.visualization.ZScaleInterval`.

     * *aspect* (``str``) aspect ratio, 'equal' or 'auto' are typical defaults.

     * *origin* (``str``) Origin of the image. Default: 'lower'

     * *title* (``str``) A title for the plot.  LaTeX allowed.

     * *vmin*  (``float``) Minimum value for colormap normalization

     * *vmax*  (``float``) Maximum value for colormap normalization

     The following keywords are available, but you probably won't touch.

     * *nrows* (``int``) Number of rows in the subplot

     * *ncols* (``int``) Number of columns in the subplot

     * *index* (``int``) Index of the subplot

     * *reset* (``bool``) Whether or not to reset the figure.

     Providing keywords other than these has undefined results, but may just work!
       
    """

    def __init__(self,tool):
        """Init method

           :param tool: The line ratio fitting tool that is to be plotted.
           :type tool: `~pdrtpy.tool.LineRatioFit`
        """

        super().__init__(tool)
        self._figure = None
        self._axis = None
        self._ratiocolor=[]
        self._CB_color_cycle = ['#377eb8', '#ff7f00', '#4daf4a',
                  '#f781bf', '#a65628', '#984ea3',
                  '#999999', '#e41a1c', '#dede00']

    def modelratio(self,id,**kwargs):
        """Plot one of the model ratios
     
           :param id: the ratio identifier, such as ``CII_158/CO_32``.
           :type id: str
           :param \**kwargs: see class documentation above
           :raises KeyError: if is id not in existing model ratios

        """
        if len(self._tool._modelratios[id].shape) == 0:
            return self._tool._modelratios[id]

        kwargs_opts = {'title': self._tool._modelset.table.loc[id]["title"], 'units': u.dimensionless_unscaled , 'colorbar':True}
        kwargs_opts.update(kwargs)
        self._plot_no_wcs(self._tool._modelratios[id],**kwargs_opts)

    def observedratio(self,id,**kwargs):
        """Plot one of the observed ratios

           :param id: the ratio identifier, such as ``CII_158/CO_32``.
           :type id: - str
           :raises KeyError: if id is not in existing observed ratios
        """
        if len(self._tool._observedratios[id].shape) == 0:
            return self._tool._observedratios[id]

        kwargs_opts = {'title': self._tool._modelset.table.loc[id]["title"], 'units': u.dimensionless_unscaled , 'colorbar':False}
        kwargs_opts.update(kwargs)
        self._plot(data=self._tool._observedratios[id],**kwargs_opts)

    def density(self,**kwargs):
        '''Plot the hydrogen nucleus volume density map that was computed by :class:`~pdrtpy.tool.lineratiofit.LineRatioFit` tool. Default units: cm :math:`^{-3}`
        '''
        kwargs_opts = {'units': 'cm^-3',
                       'aspect': 'equal',
                       'image':True,
                       'contours': False,
                       'label': False,
                       'linewidths': 1.0,
                       'levels': None,
                       'norm': None,
                       'title': None}

        kwargs_opts.update(kwargs)

        # handle single pixel case
        if len( self._tool._density.shape) == 0 :
            return to(kwargs_opts['units'],self._tool._density)

        fancyunits=u.Unit(kwargs_opts['units']).to_string('latex')
        kwargs_opts['title'] = 'n ('+fancyunits+')'
        self._plot(self._tool._density,**kwargs_opts)

    def radiation_field(self,**kwargs):
        '''Plot the radiation field map that was computed by :class:`~pdrtpy.tool.lineratiofit.LineRatioFit` tool. Default units: Habing.
        '''

        #fancyunits=self._tool._radiation_field.unit.to_string('latex')

        kwargs_opts = {'units': 'Habing',
                       'aspect': 'equal',
                       'image':True,
                       'contours': False,
                       'label': False,
                       'linewidths': 1.0,
                       'levels': None,
                       'norm': None,
                       'title': None}
        kwargs_opts.update(kwargs)

        # handle single pixel case
        if len( self._tool._radiation_field.shape) == 0 :
            return to(kwargs_opts['units'],self._tool._radiation_field)

        if kwargs_opts['title'] is None:
            fancyunits=u.Unit(kwargs_opts['units']).to_string('latex')
            if kwargs_opts['units'] not in rad_title:
                kwargs_opts['title'] = 'Radiation Field ('+fancyunits+')'
            else:
                kwargs_opts['title'] = rad_title[kwargs_opts['units']]+' ('+fancyunits+')'

        self._plot(self._tool._radiation_field,**kwargs_opts)

    #def chisq(self,xaxis,xpix,ypix):
    #    """Make a line plot of chisq as a function of G0 or n for a given pixel"""
    #    axes = {"G0":0,"n":1}
    #    axis = axes[xaxis] #yep key error if you do it wrong
    #        
    def chisq(self, **kwargs):           
        '''Plot the :math:`\chi^2` map that was computed by the
        :class:`~pdrtpy.tool.lineratiofit.LineRatioFit` tool.
        
        '''

        kwargs_opts = {'units': None,
                       'aspect': 'equal',
                       'image':True,
                       'contours': True,
                       'label': False,
                       'colors': ['white'],
                       'linewidths': 1.0,
                       'norm': 'zscale',
                       'title': r'$\chi^2$' }
        kwargs_opts.update(kwargs)
        # make a sensible choice about contours if image is not shown
        if not kwargs_opts['image'] and kwargs_opts['colors'][0] == 'white':
           kwargs_opts['colors'][0] = 'black'

        if self._tool.has_maps:
            data = self._tool.chisq(min=True)
            self._plot(data,**kwargs_opts)
        else:
            data = self._tool.chisq(min=False)
            self._plot_no_wcs(data,header=None,**kwargs_opts)

    def reduced_chisq(self, **kwargs):
        '''Plot the reduced :math:`\chi^2` map that was computed by the
        :class:`~pdrtpy.tool.lineratiofit.LineRatioFit` tool.
        
        '''

        kwargs_opts = {'units': None,
                       'aspect': 'equal',
                       'image':True,
                       'contours': True,
                       'label': False,
                       'colors': ['white'],
                       'linewidths': 1.0,
                       'norm': 'zscale',
                       'title': r'$\chi_\nu^2$ (dof=%d)'%self._tool._dof}
        kwargs_opts.update(kwargs)
        if self._tool.has_maps:
            data = self._tool.reduced_chisq(min=True)
            #print("min=True, map inputs, calling plot")
            self._plot(data,**kwargs_opts)
        else:
            #print("min=False, calling plot_no_wcs")
            data = self._tool.reduced_chisq(min=False)
            self._plot_no_wcs(data,header=None,**kwargs_opts)

    def show_both(self,units = ['Habing','cm^-3'], **kwargs):
        '''Plot both radiation field and volume density maps computed by the
        :class:`~pdrtpy.tool.lineratiofit.LineRatioFit` tool in a 1x2 panel subplot. Default units: ['Habing','cm^-3']
        '''

        _index = [1,2]
        _reset = [True,False]

        kwargs_opts = {'image':True,
                       'aspect': 'equal',
                       'contours': False,
                       'label': False,
                       'levels': None,
                       'norm': None,
                       'title': None,
                       'nrows': 1,
                       'ncols': 2,
                       'index': _index[0],
                       'reset': _reset[0]
                       }

        kwargs_opts.update(kwargs)

        rf = self.radiation_field(units=units[0],**kwargs_opts)

        kwargs_opts['index'] = _index[1]
        kwargs_opts['reset'] = _reset[1]
 
        d = self.density(units=units[1],**kwargs_opts)
        return (rf,d)

    def confidence_intervals(self,**kwargs):
        '''Plot the confidence intervals from the :math:`\chi^2` map computed by the
        :class:`~pdrtpy.tool.lineratiofit.LineRatioFit` tool. Default levels:  [50., 68., 80., 95., 99.]
        
        **Currently only works for single-pixel Measurements**
        '''

        if self._tool.has_maps:
            raise NotImplementedError("Plotting of confidence intervals is not yet implemented for maps")

        kwargs_opts = {'units': None,
                       'aspect': 'equal',
                       'image':False,
                       'contours': True,
                       'label': True,
                       'levels': [50., 68., 80., 95., 99.],
                       'colors': ['black'],
                       'linewidths': 1.0,
                       'norm': 'simple',
                       'title': "Confidence Intervals"}

        kwargs_opts.update(kwargs)

        chidata = self._tool._chisq.data
        chi2_stat = 100*stats.distributions.chi2.cdf(chidata,self._tool._dof)
        self._plot_no_wcs(data=chi2_stat,header=self._tool._chisq.header,**kwargs_opts)
        #print("CF min max ",np.min(chi2_stat),np.max(chi2_stat))
    
    def overlay_all_ratios(self,**kwargs):
        '''Overlay all the measured ratios and their errors on the :math:`(n,G_0)` space. 

        **Currently only works for single-pixel Measurements**
        '''

        if self._tool.has_maps:
            raise NotImplementedError("Plotting of ratio overlays is not yet implemented for maps")

        kwargs_opts = {'units': None,
                       'image':False,
                       'contours': False,
                       'levels' : None,
                       'label': False,
                       'linewidths': 1.0,
                       'ncols': 1,
                       'norm': None,
                       'title': None,
                       'reset': True}

        kwargs_opts.update(kwargs)
        # force this as ncols !=1 makes no sense.
        kwargs_opts['ncols'] = 1

        i =0 
        for key,val in self._tool._modelratios.items():
            self._ratiocolor = self._CB_color_cycle[i]
            kwargs_opts['measurements'] = [self._tool._observedratios[key]]
            if i > 0: kwargs_opts['reset']=False
            self._plot_no_wcs(val,header=None,**kwargs_opts)
            i = i+1
        # do some sort of legend
        lines = [Line2D([0], [0], color=c, linewidth=3, linestyle='-') for c in self._CB_color_cycle[0:i]]
        labels = list(self._tool._model_files_used.keys())
        self._plt.legend(lines, labels)

    def ratios_on_models(self,**kwargs):
        '''Overlay all the measured ratios and their errors on the individual models for those ratios.  Plots are displayed in multi-column format, controlled the `ncols` keyword. Default: ncols=2

        **Currently only works for single-pixel Measurements**
        '''

        if self._tool.has_maps:
            raise NotImplementedError("Plotting of ratio overlays is not yet implemented for maps")

        kwargs_opts = {'units': None,
                       'image':True,
                       'colorbar': True,
                       'contours': True,
                       'levels' : None,
                       'label': False,
                       'linewidths': 1.0,
                       'ncols': 2,
                       'norm': 'zscale',
                       'title': None,
                       'index': 1,
                       'reset': True}

        kwargs_opts.update(kwargs)

        kwargs_opts["ncols"] = min(kwargs_opts["ncols"],self._tool.ratiocount)
        kwargs_opts["nrows"] = int(round(self._tool.ratiocount/kwargs_opts["ncols"]+0.49,0))
        for key,val in self._tool._modelratios.items():
            if kwargs_opts['index'] > 1: kwargs_opts['reset'] = False
            m = self._tool._model_files_used[key]
            kwargs_opts['measurements'] = [self._tool._observedratios[key]]
            self._ratiocolor='#4daf4a'
            kwargs_opts['title'] = key + " model (Observed ratio indicated)"
            self._plot_no_wcs(val,header=None,**kwargs_opts)
            kwargs_opts['index'] = kwargs_opts['index'] + 1

            # Turn off subplots greater than the number of
            # available ratios
            for i in range(self._tool.ratiocount,len(self._axis)):
                self._axis[i].axis('off')
            

    def _plot(self,data,**kwargs):
        '''generic plotting method used by other plot methods'''

        kwargs_opts = {'units' : None,
                       'image':True,
                       'colorbar': True,
                       'contours': True,
                       'label': False,
                       'title': None
                       }

        kwargs_contour = {'levels': None, 
                          'colors': ['white'],
                          'linewidths': 1.0}


        # Merge in any keys the user provided, overriding defaults.
        kwargs_contour.update(kwargs)
        kwargs_opts.update(kwargs)

        if self._tool._modelnaxis == 2 or len(data.shape)==2:
            if kwargs_opts['units'] is not None:
                k = to(kwargs_opts['units'], data)
            else:
                k = data
        elif self._tool._modelnaxis == 3:
            if kwargs_opts['units'] is not None:
                k = to(kwargs_opts['units'], data[0,:,:])
            else:
                k = data[0,:,:]
        else:
            raise Exception("Unexpected model naxis: %d"%self._tool._modelnaxis)
        km = ma.masked_invalid(k)
        # make sure nans don't affect the color map
        min_ = np.nanmin(km)
        max_ = np.nanmax(km)

        kwargs_imshow = { 'origin': 'lower', 
                          'norm': 'simple',
                          'vmin': min_, 
                          'vmax': max_,
                          'cmap': 'plasma',
                          'aspect': 'auto'}
 
        kwargs_subplot = {'nrows': 1,
                          'ncols': 1,
                          'index': 1,
                          'reset': True,
                         }

        # delay merge until min_ and max_ are known
        kwargs_imshow.update(kwargs)
        kwargs_imshow['norm']=self._get_norm(kwargs_imshow['norm'],km,min_,max_)

        kwargs_subplot.update(kwargs)
        # swap ncols and nrows in figsize to preserve aspect ratio
        kwargs_subplot['figsize'] = kwargs.get("figsize",(kwargs_subplot["ncols"]*5,kwargs_subplot["nrows"]*5))

        #print("Got non-default kwargs: ", kwargs)

        axidx = kwargs_subplot['index']-1
        if kwargs_subplot['reset']:
            self._figure,self._axis = self._plt.subplots(kwargs_subplot['nrows'],kwargs_subplot['ncols'],figsize=kwargs_subplot['figsize'],subplot_kw={'projection':k.wcs,'aspect':kwargs_imshow['aspect']},constrained_layout=True)

        # Make sure self._axis is an array because we will index it below.
        if type(self._axis) is not np.ndarray:
            self._axis = np.array([self._axis])
        
        if kwargs_opts['image']:
            current_cmap = mcm.get_cmap(kwargs_imshow['cmap'])
            current_cmap.set_bad(color='white',alpha=1)
            # suppress errors and warnings about unused keywords
            for kx in ['units', 'image', 'contours', 'label', 'title','linewidths','levels','nrows','ncols', 'index', 'reset','colors']:
                kwargs_imshow.pop(kx,None)
            im=self._axis[axidx].imshow(km,**kwargs_imshow)
            if kwargs_opts['colorbar']:
                self._wcs_colorbar(im,self._axis[axidx])

        if kwargs_opts['contours']:
            if kwargs_contour['levels'] is None:
                # Figure out some autolevels 
                kwargs_contour['levels'] = self._autolevels(km,'log')

            # suppress errors and warnings about unused keywords
            for kx in ['units', 'image', 'contours', 'label', 'title', 'cmap','aspect','colorbar','reset', 'nrows', 'ncols', 'index']:
                kwargs_contour.pop(kx,None)

            contourset = self._axis[axidx].contour(km, **kwargs_contour)
            if kwargs_opts['label']:
                self._axis[axidx].clabel(contourset,contourset.levels,inline=True,fmt='%1.1e')

        if kwargs_opts['title'] is not None: 
            self._axis[axidx].set_title(kwargs_opts['title'])

        if k.wcs is not None:
            self._axis[axidx].set_xlabel(k.wcs.wcs.lngtyp)
            self._axis[axidx].set_ylabel(k.wcs.wcs.lattyp)
        
       
    def _plot_no_wcs(self,data,header=None,**kwargs):
        '''generic plotting method for images with no WCS used by other plot methods'''
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
                       'yaxis_unit': None
                       }

        kwargs_contour = {'levels': None, 
                          'colors': ['white'],
                          'linewidths': 1.0}


        # Merge in any keys the user provided, overriding defaults.
        kwargs_contour.update(kwargs)
        #print("kwargs_opts 1: ",kwargs_opts)
        kwargs_opts.update(kwargs)
        #print("kwargs_opts 2: ",kwargs_opts)
        #print("kwargs 2: ",kwargs)

        if self._tool._modelnaxis == 2:
            if kwargs_opts['units'] is not None:
                k = to(kwargs_opts['units'], data)
            else:
                k = data
        elif self._tool._modelnaxis == 3:
            if kwargs_opts['units'] is not None:
                k = to(kwargs_opts['units'], data[0,:,:])
            else:
                k = data[0,:,:]
        else:
            raise Exception("Unexpected model naxis: %d"%self._tool._modelnaxis)

        km = ma.masked_invalid(k)
        # make sure nans don't affect the color map
        min_ = np.nanmin(km)
        max_ = np.nanmax(km)

        kwargs_imshow = { 'origin': 'lower', 
                          'norm': 'simple',
                          'vmin': min_, 
                          'vmax': max_,
                          'cmap': 'plasma',
                          'aspect': 'equal'}
 
        kwargs_subplot = {'nrows': 1,
                          'ncols': 1,
                          'index': 1,
                          'reset': True,
                         }

        # delay merge until min_ and max_ are known
        #print("plot kwargs 1: ",kwargs_imshow)
        kwargs_imshow.update(kwargs)
        #print("plot kwargs 2: ",kwargs_imshow)
        kwargs_imshow['norm']=self._get_norm(kwargs_imshow['norm'],km,min_,max_)

        kwargs_subplot.update(kwargs)
        # swap ncols and nrows in figsize to preserve aspect ratio
        kwargs_subplot['figsize'] = kwargs.get("figsize",(kwargs_subplot["ncols"]*5,kwargs_subplot["nrows"]*5))
        #print("subplot kwargs : ",kwargs_subplot)

        #print("Got non-default kwargs: ", kwargs)

        axidx = kwargs_subplot['index']-1
        if kwargs_subplot['reset']:
# @todo can probably consolodate this
            self._figure,self._axis = self._plt.subplots(kwargs_subplot['nrows'],kwargs_subplot['ncols'],figsize=kwargs_subplot['figsize'],subplot_kw={'aspect':kwargs_imshow['aspect']},constrained_layout=True)

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
    
        y = 10**np.linspace(start=ystart, stop=ystop, num=_header['naxis'+ax2])
        x = 10**np.linspace(start=xstart, stop=xstop, num=_header['naxis'+ax1])
        locmaj = ticker.LogLocator(base=10.0, subs=(1.0, ),numticks=10)
        locmin = ticker.LogLocator(base=10.0, subs=np.arange(2, 10)*.1,numticks=10) 
        
        xlab = _header['ctype'+ax1] + ' ['+_header['cunit'+ax1]+']'
        #todo: allow unit conversion to cgs or Draine
        if kwargs_opts['yaxis_unit'] is not None:
            yunit = kwargs_opts['yaxis_unit']
            temp_y= y * u.Unit(_header['cunit'+ax1])
            y = temp_y.to(yunit).value
            ylab = f"{_header['ctype'+ax2]} [{yunit}]"
        else:
            ylab = _header['ctype'+ax2] + ' ['+_header['cunit'+ax2]+']'
        
        self._axis[axidx].set_ylabel(ylab)
        self._axis[axidx].set_xlabel(xlab)


        self._axis[axidx].set_xscale('log')
        self._axis[axidx].set_yscale('log')
        self._axis[axidx].xaxis.set_major_locator(locmaj)
        self._axis[axidx].xaxis.set_minor_locator(locmin)
        self._axis[axidx].xaxis.set_minor_formatter(ticker.NullFormatter())

        if kwargs_opts['image']:
            im = self._axis[axidx].pcolormesh(x,y,km,cmap=kwargs_imshow['cmap'],norm=kwargs_imshow['norm'])
            if kwargs_opts['colorbar']:
                self._figure.colorbar(im,ax=self._axis[axidx])
                #self._wcs_colorbar(im,self._axis[axidx])

        if kwargs_opts['contours']:
            if kwargs_contour['levels'] is None:
                # Figure out some autolevels 
                kwargs_contour['levels'] = self._autolevels(km,'log')

            # suppress warnings about unused keywords and potential error 
            # about cmap not being None
            for kx in ['units', 'image', 'contours', 'label', 'title', 'cmap','aspect','colorbar','reset', 'nrows', 'ncols', 'index']:
                kwargs_contour.pop(kx,None)

            contourset = self._axis[axidx].contour(x,y,km.data, **kwargs_contour)
            #print("contourset :",contourset.levels)

            if kwargs_opts['label']:
                drawn = self._axis[axidx].clabel(contourset,contourset.levels,inline=True,fmt='%1.2e')
                #print("drew %s"%drawn)

        if kwargs_opts['title'] is not None: 
            self._axis[axidx].set_title(kwargs_opts['title'])

        if measurements is not None:
            for m in measurements:
                lstyles = ['--','-','--']
                colors = [self._ratiocolor,self._ratiocolor,self._ratiocolor]
                for i in range(0,3):
                    cset = self._axis[axidx].contour(x,y,k.data,levels=m.levels, linestyles=lstyles, colors=colors)
                
