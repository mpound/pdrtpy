#todo: 
# keywords for show_both need to be arrays. ugh.
#
# allow levels to be percent?
#
# Look into seaborn https://seaborn.pydata.org
# Also https://docs.bokeh.org/en
# especially for coloring and style

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

from astropy.nddata.utils import Cutout2D
from astropy.io import fits
import astropy.wcs as wcs
import astropy.units as u
from astropy.units import UnitsWarning
from astropy.nddata import NDDataArray, CCDData, NDUncertainty, StdDevUncertainty, VarianceUncertainty, InverseVariance

from .plotbase import PlotBase
from .modelplot import ModelPlot
from .. import pdrutils as utils


class LineRatioPlot(PlotBase):
    """Class to plot various results from PDR Toolbox model fitting.
    

    :Keyword Arguments:

    To manage the plots, the methods in this class take keywords (\*\*kwargs) that turn on or off various options, specify plot units, or map to matplotlib's :meth:`~matplotlib.axes.Axes.plot`, :meth:`~matplotlib.axes.Axes.imshow`, :meth:`~matplotlib.axes.Axes.contour` keywords.  The methods have reasonable defaults, so try them with no keywords to see what they do before modifying keywords.

     * *units* (``str`` or :class:`astropy.units.Unit`) image data units to use in the plot. This can be either a string such as, 'cm^-3' or 'Habing', or it can be an :class:`astropy.units.Unit`.  Data will be converted to the desired unit.   Note these are **not** the axis units, but the image data units.  Modifying axis units is implemented via the `xaxis_unit` and `yaxis_unit` keywords. 

     * *image* (``bool``) whether or not to display the image map (imshow). 

     * *show* (``str``) which quantity to display in the Measurement, one of 'data', 'error', 'mask'.  For example, this can be used to plot the errors in observed ratios. Default: 'data'

     * *cmap* (``str``) colormap name, Default: 'plasma' 

     * *colorbar* (``str``) whether or not to display colorbar

     * *colors* (``str``) color of the contours. Default: 'whitecolor of the contours. Default: 'white'

     * *contours* (``bool``), whether or not to plot contours

     * *label* (``bool``), whether or not to label contours 

     * *linewidths* (``float or sequence of float``), the line width in points, Default: 1.0

     * *legend* (``bool``) Draw a legend on the plot. If False, a title is drawn above the plot with the value of the *title* keyword

     * *levels* (``int`` or array-like) Determines the number and positions of the contour lines / regions.  If an int n, use n data intervals; i.e. draw n+1 contour lines. The level heights are automatically chosen.  If array-like, draw contour lines at the specified levels. The values must be in increasing order.  

     * *measurements* (array-like) A list of single pixel Measurements that can be contoured over a model ratio or intensity map.

     * *meas_color* (array of str) A list of colors to use when overlaying Measurement contours. There should be one color for each Measurement in the *measurement* keyword.  The Default of None will use a color-blind friendly color cycle.

     * *norm* (``str`` or :mod:`astropy.visualization` normalization object) The normalization to use in the image. The string 'simple' will normalize with :func:`~astropy.visualization.simple_norm` and 'zscale' will normalize with IRAF's zscale algorithm.  See :class:`~astropy.visualization.ZScaleInterval`.

     * *stretch* (``str``)  {'linear', 'sqrt', 'power', log', 'asinh'}. The stretch function to apply to the image for simple_norm.  The Default is 'linear'.

     * *aspect* (``str``) aspect ratio, 'equal' or 'auto' are typical defaults.

     * *origin* (``str``) Origin of the image. Default: 'lower'

     * *title* (``str``) A title for the plot.  LaTeX allowed.

     * *vmin*  (``float``) Minimum value for colormap normalization

     * *vmax*  (``float``) Maximum value for colormap normalization
    
     * *xaxis_unit* (``str`` or :class:`astropy.units.Unit`) X axis (density) units to use when plotting models, such as in :meth:`overlay_all_ratios` or :meth:`modelratio`.  If None, the native model axis units are used.

     * *yaxis_unit* (``str`` or :class:`astropy.units.Unit`) Y axis (FUV radiation field flux) units to use when plotting models, such as in :meth:`overlay_all_ratios` or :meth:`modelratio`.  If None, the native model axis units are used.

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
        self._modelplot = ModelPlot(self._tool._modelset,self._figure,self._axis)
        self._ratiocolor=[]
        self._CB_color_cycle = [ '#377eb8', '#ff7f00', '#4daf4a',
                  '#f781bf', '#a65628', '#984ea3',
                  '#999999', '#e41a1c', '#dede00']

    def modelintensity(self,id,**kwargs):
        """Plot one of the model intensities

           :param id: the intensity identifier, such as `CO_32``.
           :type id: str
           :param \**kwargs: see class documentation above
           :raises KeyError: if is id not in existing model intensities
        """
        ms = self._tool.modelset
        if id not in ms.supported_lines["intensity label"]:
            raise KeyError(f"{id} is not in the ModelSet of your LineRatioFit")
        
        model = ms.get_models([id],model_type="intensity")
        kwargs_opts = {'title': self._tool._modelset.table.loc[id]["title"], 'colorbar':True}
        kwargs_opts.update(kwargs)
        self._modelplot._plot_no_wcs(model[id],**kwargs_opts)

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
        self._modelplot._plot_no_wcs(self._tool._modelratios[id],**kwargs_opts)

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
            return utils.to(kwargs_opts['units'],self._tool._density)

        tunit=u.Unit(kwargs_opts['units'])
        kwargs_opts['title'] = r"n [{0:latex_inline}]".format(tunit)
        self._plot(self._tool._density,**kwargs_opts)

    def radiation_field(self,**kwargs):
        '''Plot the radiation field map that was computed by :class:`~pdrtpy.tool.lineratiofit.LineRatioFit` tool. Default units: Habing.
        '''

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
        if len( self._tool.radiation_field.shape) == 0 :
            return utils.to(kwargs_opts['units'],self._tool.radiation_field)

        if kwargs_opts['title'] is None:
            rad_title = utils.get_rad(kwargs_opts['units']) 
            tunit=u.Unit(kwargs_opts['units'])
            kwargs_opts['title'] = r"{0} [{1:latex_inline}]".format(rad_title,tunit)

        self._plot(self._tool.radiation_field,**kwargs_opts)

    #def chisq(self,xaxis,xpix,ypix):
    #    """Make a line plot of chisq as a function of G0 or n for a given pixel"""
    #    axes = {"G0":0,"n":1}
    #    axis = axes[xaxis] #yep key error if you do it wrong
    #        

    #@TODO refactor this method with reduced_chisq()
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
                       'norm': 'simple',
                       'stretch': 'linear',
                       'xaxis_unit': None,
                       'yaxis_unit': None,
                       'legend': None,
                       'title': None}

        kwargs_opts.update(kwargs)
        # make a sensible choice about contours if image is not shown
        if not kwargs_opts['image'] and kwargs_opts['colors'][0] == 'white':
           kwargs_opts['colors'][0] = 'black'

        if self._tool.has_maps:
            data = self._tool.chisq(min=True)
            if 'title' not in kwargs:
                kwargs_opts['title'] = r'$\chi^2$ (dof=%d)'%self._tool._dof
            self._plot(data,**kwargs_opts)
        else:
            data = self._tool.chisq(min=False)
            self._modelplot._plot_no_wcs(data,header=None,**kwargs_opts)
            # Put a crosshair where the chisq minimum is.
            # To do this we first get the array index of the minimum
            # then use WCS to translate to world coordinates.
            [row,col] = np.where(self._tool._chisq==self._tool._chisq_min.flux)
            #print("ROW,COL (Pixel units) = ",row,col)
            mywcs = wcs.WCS(data.header)
            # Suppress WCS warning about 1/cm3 not being FITS
            warnings.simplefilter('ignore',category=UnitsWarning)
            logn,logrf = mywcs.array_index_to_world(row,col)
            logn,logrf = mywcs.pixel_to_world(col,row)
            #print("logrf.unit",logrf.unit)
            warnings.resetwarnings()
            n = (10**logn.value[0])*u.Unit(logn.unit.to_string())
            rf = (10**logrf.value[0])*u.Unit(logrf.unit.to_string())
            #print("n dn",n,mywcs.wcs.cdelt[0])
            #print("rf drf",rf,mywcs.wcs.cdelt[0])
            # x,y are at the bottom corner of the pixel. So add half a pixel in each
            # dimension?
            #logn.value[0] += 0.5*mywcs.wcs.cdelt[0]
            #logrf.value[0] += 0.5*mywcs.wcs.cdelt[1]
            if kwargs_opts['xaxis_unit'] is not None:
                x = n.to(kwargs_opts['xaxis_unit']).value
            else:
                x = n.value
            if kwargs_opts['yaxis_unit'] is not None:
                y = rf.to(kwargs_opts['yaxis_unit']).value
            else:
                y = rf.value

            if kwargs_opts['title'] is None:
                kwargs_opts['title'] = r'$\chi^2$ (dof=%d)'%self._tool._dof
            label = r'$\chi_{min}^2$ = %.2g @ (n,FUV) = (%.2g,%.2g)'%(self._tool._chisq_min.flux,x,y)
            self._axis[0].scatter(x,y,c='r',marker='+',s=200,linewidth=2,label=label)
            # handle legend locally
            if kwargs_opts['legend']:
                legend = self._axis[0].legend(loc='upper center',title=kwargs_opts['title'])

    def cost(self, **kwargs):           
        '''Plot the cost funtion map that was computed by the
        :class:`~pdrtpy.tool.lineratiofit.LineRatioFit` tool.
        
        '''

        kwargs_opts = {'units': None,
                       'aspect': 'equal',
                       'image':True,
                       'contours': True,
                       'label': False,
                       'colors': ['white'],
                       'linewidths': 1.0,
                       'norm': 'simple',
                       'stretch': 'linear',
                       'xaxis_unit': None,
                       'yaxis_unit': None,
                       'legend': None,
                       'title': None}

        kwargs_opts.update(kwargs)
        # make a sensible choice about contours if image is not shown
        if not kwargs_opts['image'] and kwargs_opts['colors'][0] == 'white':
           kwargs_opts['colors'][0] = 'black'

        if self._tool.has_maps:
            data = self._tool._cost_min
            if 'title' not in kwargs:
                kwargs_opts['title'] = "Cost"
            self._plot(data,**kwargs_opts)

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
                       'norm': 'simple',
                       'stretch': 'linear',
                       'xaxis_unit': None,
                       'yaxis_unit': None,
                       'legend': None,
                       'title': None
                      }
        kwargs_opts.update(kwargs)
        if self._tool.has_maps:
            if 'title' not in kwargs:
                kwargs_opts['title'] = r'$\chi_\nu^2$ (dof=%d)'%self._tool._dof
            data = self._tool.reduced_chisq(min=True)
            self._plot(data,**kwargs_opts)
            # doesn't make sense to point out minimum chisq on a spatial-spatial map,
            # so no legend
        else:
            data = self._tool.reduced_chisq(min=False)
            self._modelplot._plot_no_wcs(data,header=None,**kwargs_opts)
            # Put a crosshair where the chisq minimum is.
            # To do this we first get the array index of the minimum
            # then use WCS to translate to world coordinates.
            [row,col] = np.where(self._tool._reduced_chisq==self._tool._reduced_chisq_min.flux)
            #print("ROW,COL (Pixel units) = ",row,col)
            mywcs = wcs.WCS(data.header)
            # Suppress WCS warning about 1/cm3 not being FITS
            warnings.simplefilter('ignore',category=UnitsWarning)
            logn,logrf = mywcs.array_index_to_world(row,col)
            warnings.resetwarnings()
            # logn, logrf are Quantities of the log(density) and log(radiation field),
            # respectively.  The model default units are cm^-2 and erg/s/cm^-2. 
            # These must be converted to plot units based on user input 
            # xaxis_unit and yaxis_unit. 
            # Note: multiplying by n.unit causes the ValueError:
            # "The unit '1/cm3' is unrecognized, so all arithmetic operations with it are invalid."
            # Yet by all other measures this appears to be a valid unit. 
            # The workaround is to us to_string() method.
            #print("logrf.unit",logrf.unit)
            n = (10**logn.value[0])*u.Unit(logn.unit.to_string())
            rf = (10**logrf.value[0])*u.Unit(logrf.unit.to_string())
            if kwargs_opts['xaxis_unit'] is not None:
                x = n.to(kwargs_opts['xaxis_unit']).value
            else:
                x = n.value
            if kwargs_opts['yaxis_unit'] is not None:
                y = rf.to(kwargs_opts['yaxis_unit']).value
            else:
                y = rf.value

            if kwargs_opts['title'] is None:
                kwargs_opts['title'] = r'$\chi_\nu^2$ (dof=%d)'%self._tool._dof
            label = r'$\chi_{\nu,min}^2$ = %.2g @ (n,FUV) = (%.2g,%.2g)'%(self._tool._reduced_chisq_min.flux,x,y)
            self._axis[0].scatter(x,y,c='r',marker='+',s=200,linewidth=2,label=label)
            # handle legend locally
            if kwargs_opts['legend']:
                legend = self._axis[0].legend(loc='upper center',title=kwargs_opts['title'])

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
                       'stretch': 'linear',
                       'xaxis_unit': None,
                       'yaxis_unit': None,
                       'title':  "Confidence Intervals"}

        kwargs_opts.update(kwargs)

        chidata = self._tool._chisq.data
        chi2_stat = 100*stats.distributions.chi2.cdf(chidata,self._tool._dof)
        self._modelplot._plot_no_wcs(data=chi2_stat,header=self._tool._chisq.header,**kwargs_opts)
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
                       'meas_color': self._CB_color_cycle,
                       'levels' : None,
                       'label': False,
                       'linewidths': 1.0,
                       'ncols': 1,
                       'norm': None,
                       'title': None,
                       'reset': True,
                       'legend': True}

        kwargs_opts.update(kwargs)
        # force this as ncols !=1 makes no sense.
        kwargs_opts['ncols'] = 1

        i =0 
        for key,val in self._tool._modelratios.items():
            kwargs_opts['measurements'] = [self._tool._observedratios[key]]
            if i > 0: kwargs_opts['reset']=False
            # pass the index of the contour color to use via the "secret" colorcounter keyword.
            self._modelplot._plot_no_wcs(val,header=None,colorcounter=i,**kwargs_opts)
            i = i+1

        if kwargs_opts['legend']:
            lines = [Line2D([0], [0], color=c, linewidth=3, linestyle='-') for c in kwargs_opts['meas_color'][0:i]]
            labels = [self._tool._modelratios[k].title for k in self._tool._modelratios]
            self._plt.legend(lines, labels,loc='upper center',title='Observed Ratios')

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
                       'colors': ['white'],
                       'levels' : None,
                       'label': False,
                       'linewidths': 1.0,
                       'meas_color':  ['#4daf4a'],
                       'ncols': 2,
                       'norm': 'simple',
                       'stretch': 'linear',
                       'index': 1,
                       'reset': True,
                       'legend': True}

        kwargs_opts.update(kwargs)

        kwargs_opts["ncols"] = min(kwargs_opts["ncols"],self._tool.ratiocount)
        kwargs_opts["nrows"] = int(round(self._tool.ratiocount/kwargs_opts["ncols"]+0.49,0))
        # defend against meas_color not being a list
        if type(kwargs_opts['meas_color']) == str:
            #warnings.warn("meas_color should be a list")
            kwargs_opts['meas_color']=[kwargs_opts['meas_color']]

        for key,val in self._tool._modelratios.items():
            axidx = kwargs_opts['index']-1
            if kwargs_opts['index'] > 1: kwargs_opts['reset'] = False
            m = self._tool._model_files_used[key]
            kwargs_opts['measurements'] = [self._tool._observedratios[key]]
            self._modelplot._plot_no_wcs(val,header=None,**kwargs_opts)
            self._axis = self._modelplot._axis
            self._figure = self._modelplot._figure
            kwargs_opts['index'] = kwargs_opts['index'] + 1
            if kwargs_opts['legend']:
                if 'title' not in kwargs: # then it was None, and we customize it
                    _title = self._tool._modelratios[key].title
                else:
                    _title = kwargs['title']
                lines = list()
                labels = list()
                if kwargs_opts['contours']:
                    lines.append(Line2D([0], [0], color=kwargs_opts['colors'][0], linewidth=3, linestyle='-'))
                    labels.append("model")
                lines.append(Line2D([0], [0], color=kwargs_opts['meas_color'][0], linewidth=3, linestyle='-'))
                labels.append("observed")
                #maybe loc should be 'best' but then it bounces around
                self._axis[axidx].legend(lines, labels,loc='upper center',title=_title)

            # Turn off subplots greater than the number of
            # available ratios
            for i in range(self._tool.ratiocount,len(self._axis)):
                self._axis[i].axis('off')


    def _plot(self,data,**kwargs):
        '''generic plotting method used by other plot methods'''

        kwargs_plot = {'show' : 'data' # or 'mask' or 'error'
                      } 

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
        kwargs_plot.update(kwargs)

        _data = data  # default is show the data

        if kwargs_plot['show'] == 'error':
            _data = deepcopy(data)
            _data.data = _data.error
        if kwargs_plot['show'] == 'mask':
            _data = deepcopy(data)
            _data.data = _data.mask
            # can't contour a boolean
            kwargs_opts['contours'] = False

        if self._tool._modelnaxis == 2 or len(_data.shape)==2:
            if kwargs_opts['units'] is not None:
                k = utils.to(kwargs_opts['units'], _data)
            else:
                k = _data
        elif self._tool._modelnaxis == 3:
            if kwargs_opts['units'] is not None:
                k = utils.to(kwargs_opts['units'], _data[0,:,:])
            else:
                k = _data[0,:,:]
        else:
            raise Exception("Unexpected model naxis: %d"%self._tool._modelnaxis)

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
                          'aspect': 'auto'}
 
        kwargs_subplot = {'nrows': 1,
                          'ncols': 1,
                          'index': 1,
                          'reset': True,
                          'constrained_layout': True
                         }

        # delay merge until min_ and max_ are known
        kwargs_imshow.update(kwargs)
        kwargs_imshow['norm']=self._get_norm(kwargs_imshow['norm'],km,
                                             kwargs_imshow['vmin'],kwargs_imshow['vmax'],
                                             kwargs_imshow['stretch'])

        kwargs_subplot.update(kwargs)
        # swap ncols and nrows in figsize to preserve aspect ratio
        kwargs_subplot['figsize'] = kwargs.get("figsize",(kwargs_subplot["ncols"]*5,kwargs_subplot["nrows"]*5))

        #print("Got non-default kwargs: ", kwargs)

        axidx = kwargs_subplot['index']-1
        if kwargs_subplot['reset']:
            self._figure,self._axis = self._plt.subplots(kwargs_subplot['nrows'],kwargs_subplot['ncols'],
                                                    figsize=kwargs_subplot['figsize'],
                                                    subplot_kw={'projection':k.wcs,
                                                                'aspect':kwargs_imshow['aspect']},
                                                    constrained_layout=kwargs_subplot['constrained_layout'])
            #print("CL=",self._plt.rcParams['figure.constrained_layout.use']) 
        #self._figure.subplots_adjust(top=0.7)
        #self._figure.tight_layout()

        # Make sure self._axis is an array because we will index it below.
        if type(self._axis) is not np.ndarray:
            self._axis = np.array([self._axis])
        
        if kwargs_opts['image']:
            current_cmap = copy(mcm.get_cmap(kwargs_imshow['cmap']))
            current_cmap.set_bad(color='white',alpha=1)
            # suppress errors and warnings about unused keywords
            #@todo need a better solution for this, it is not scalable.
            #push onto a stack?
            for kx in ['units', 'image', 'contours', 'label', 'title','linewidths','levels','nrows','ncols', 
                       'index', 'reset','colors','colorbar','show','yaxis_unit','xaxis_unit',
                       'constrained_layout','figsize','stretch','legend']:
                kwargs_imshow.pop(kx,None)
            # eliminate deprecation warning.  vmin,vmax are passed to Normalization object.
            if kwargs['norm'] is not None:
                kwargs_imshow.pop('vmin',None)
                kwargs_imshow.pop('vmax',None)
            im=self._axis[axidx].imshow(km,**kwargs_imshow)
            if kwargs_opts['colorbar']:
                self._wcs_colorbar(im,self._axis[axidx])

        if kwargs_opts['contours']:
            if kwargs_contour['levels'] is None:
                # Figure out some autolevels 
                kwargs_contour['levels'] = self._autolevels(km,'log')

            # suppress errors and warnings about unused keywords
            for kx in ['units', 'image', 'contours', 'label', 'title', 'cmap','aspect',
                       'colorbar','reset', 'nrows', 'ncols', 'index','show','yaxis_unit',
                       'xaxis_unit','norm','constrained_layout','figsize','stretch','legend']:
                kwargs_contour.pop(kx,None)

            contourset = self._axis[axidx].contour(km, **kwargs_contour)
            if kwargs_opts['label']:
                self._axis[axidx].clabel(contourset,contourset.levels,inline=True,fmt='%1.1e')

        if kwargs_opts['title'] is not None: 
            self._axis[axidx].set_title(kwargs_opts['title'])

        if k.wcs is not None:
            self._axis[axidx].set_xlabel(k.wcs.wcs.lngtyp)
            self._axis[axidx].set_ylabel(k.wcs.wcs.lattyp)
