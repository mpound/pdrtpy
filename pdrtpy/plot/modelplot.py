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
    """Class to plot models and optionally Measurements.  It does not require :class:`~pdrtpy.tool.lineratiofit.LineRatioFit` first.
        The methods of this class can take a variety of optional keywords. See the doc for LineRatioPlot for a description of these keywords. @todo move doc someplace more useful
    """
    def __init__(self,modelset,figure=None,axis=None):
        """Init method

           :param modelset: The set of models to use in these plots.
           :type modelset: `~pdrtpy.modelset.ModelSet`
        """
        super().__init__(tool=None)
        self._modelset = modelset
        self._figure = figure
        self._axis = axis
        # color blind/friendly color cyle courtesy https://gist.github.com/thriveth/8560036
        self._CB_color_cycle = ['#377eb8', '#ff7f00','#4daf4a',
                  '#f781bf', '#a65628', '#984ea3',
                  '#999999', '#e41a1c', '#dede00']
    """Init method

    """
    
    def plot(self,identifier,**kwargs):
        """Plot a model intensity or ratio

        :param identifier: Identifier tag for the model to plot, e.g., "CII_158","OI_145","CO_43/CO_21']
        :type identifier: str

        :SeeAlso: `meth:~pdrtpy.modelset.ModelSet.supported_lines` for a list of available identifer tags
        """
        kwargs_opts = { 'measurements': None}
        kwargs_opts.update(kwargs)
        if '/' in identifier:
            self.ratio(identifier,**kwargs_opts)
        else:
            self.intensity(identifier,**kwargs_opts)

    def ratio(self,identifier,**kwargs):
        """Plot a model ratio

        :param identifier: Identifier tag for the model to plot, e.g., "OI_63+CII_158/FIR", "CO_43/CO_21']
        :type identifier: str

        :SeeAlso: `meth:~pdrtpy.modelset.ModelSet.supported_ratios` for a list of available identifer tags
        """
        ms = self._modelset
        model = ms.get_model(identifier)
        kwargs_opts = {'title': ms.table.loc[identifier]["title"], 
                       'colorbar':True, 
                       'contours':True,
                       'colors':['white'],
                       'meas_color': [self._CB_color_cycle[0]],
                       'legend':True,
                       'image':True,
                      }
        kwargs_opts.update(kwargs)

        # make a sensible choice about contours if image is not shown
        if not kwargs_opts['image'] and kwargs_opts['colors'][0] == 'white':
           kwargs_opts['colors'][0] = 'black'

        self._plot_no_wcs(model,**kwargs_opts)
        if kwargs_opts['legend']:
            lines = list()
            labels = list()
            if kwargs_opts['contours']:
                lines.append(Line2D([0], [0], color=kwargs_opts['colors'][0], linewidth=3, linestyle='-'))
                labels.append("model")
            if kwargs_opts['measurements'] is not None:
                lines.append(Line2D([0], [0], color=kwargs_opts['meas_color'][0], linewidth=3, linestyle='-'))
                labels.append("observed")
            #maybe loc should be 'best' but then it bounces around
            self._axis[0].legend(lines, labels,loc='upper center',title=kwargs_opts['title'])

    def intensity(self,identifier,**kwargs):
        """Plot a model ratio

        :param identifier: Identifier tag for the model to plot, e.g., "OI_63", "CII_158", "CO_10"]
        :type identifier: str

        :SeeAlso: `meth:~pdrtpy.modelset.ModelSet.supported_intensities` for a list of available identifer tags
        """
        # shouldn't need separate model intensity as keyword would tell you.
        # Idea: Put a 'modeltyp' keyword in FITS header whether it is intensity ratio or intensity.
        ms = self._modelset
        meas = kwargs.get("measurements",None)
        model = ms.get_models([identifier],model_type="intensity")
        if meas is not None:
            if type(meas[0]) is not Measurement:
                raise TypeError("measurement keyword value must be a list of Measurements")
            if (model[identifier]._unit != meas[0].unit ):
                raise TypeError(f"Model and Measurement for {identifier} have different units: ({model._unit},{meas_unit})")
            if identifier != meas[0].id:
                msg = f"Identifiers of model {identifier} and supplied Measurement {meas[0].id} do not match. Plotting anyway."
                warnings.warn(msg)

        kwargs_opts = {'title': ms.table.loc[identifier]["title"], 
                       'colorbar':True, 
                       'contours':True,
                       'colors':['white'],
                       'meas_color': [self._CB_color_cycle[0]],
                       'legend':True,
                       'image':True
                      }

        kwargs_opts.update(kwargs)
        if not kwargs_opts['image'] and kwargs_opts['colors'][0] == 'white':
           kwargs_opts['colors'][0] = 'black'
        self._plot_no_wcs(model[identifier],**kwargs_opts)
        if kwargs_opts['legend']:
            lines = list()
            labels = list()
            if kwargs_opts['contours']:
                lines.append(Line2D([0], [0], color=kwargs_opts['colors'][0], linewidth=3, linestyle='-'))
                labels.append("model")
            if kwargs_opts['measurements'] is not None:
                lines.append(Line2D([0], [0], color=kwargs_opts['meas_color'][0], linewidth=3, linestyle='-'))
                labels.append("observed")
            #maybe loc should be 'best' but then it bounces around
            self._axis[0].legend(lines, labels,loc='upper center',title=kwargs_opts['title'])


    def overlay(self,measurements,**kwargs):
        '''Overlay one or more single-pixel measurements in the model space ($n,G_0). 

        :param measurements: a list of one or more :class:`pdrtpy.measurement.Measurement` to overlay.
        :type measurements: list
        :param shading: Controls how measurements and errors are drawn.
        If shading is zero, Measurements will be drawn in solid contour for
        the value and dashed for the +/- errors. If shading is between 0
        and 1, Measurements are drawn with as filled contours representing
        the size of the errors (see :meth:`~matplotlib.pyplot.contourf`)
        with alpha set to the shading value.  Default value: 0.4
        :type shading: float
        
        '''

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
                       'legend': True,
                       'shading': 0.4}

        kwargs_opts.update(kwargs)
        if kwargs_opts['shading'] <0 or kwargs_opts['shading']>1:
            raise ValueError("Shading must be between 0 and 1 inclusive")
        ids = [m.id for m in measurements]
        meas = dict(zip(ids,measurements))
        models = [self._modelset.get_model(i) for i in ids]
        i =0 
        nratio = 0
        nintensity = 0
        for val in models:
            if len(meas[val.id].data) != 1:
                raise ValueError(f"Can't plot {val.id}. This method only works with single pixel Measurements [len(measurement.data) must be 1]")
            if i > 0: kwargs_opts['reset']=False
            # pass the index of the contour color to use via the "secret" colorcounter keyword.
            self._plot_no_wcs(val,header=None,
                              measurements=[utils.convert_if_necessary(meas[val.id])],
                              colorcounter=i,**kwargs_opts)
            if val.modeltype == "ratio": nratio=nratio+1
            if val.modeltype == "intensity": nintensity=nintensity+1
            i = i+1
        if kwargs_opts['legend']:
            if nratio == 0 and nintensity >0:
                word = "Intensities"
            elif nratio >0 and nintensity == 0:
                word = "Ratios"
            else:
                word = "Values"
            lines = [Line2D([0], [0], color=c, linewidth=3, linestyle='-') for c in kwargs_opts['meas_color'][0:i]]
            labels = [k.title for k in models]
            self._plt.legend(lines, labels,loc='upper center',title='Observed '+word)


    def rvr(self,identifiers,
                 dens_clip=[10,1E7]*u.Unit("cm-3"),
                 rad_clip=[10,1E6]*utils.habing_unit,
                 reciprocal=[False,False]):
        '''Plot lines of constant density and radiation field on a ratio-ratio, ratio-intensity, or intensity-intensity map

        :param identifiers: list of two identifier tags for the model to plot, e.g., ["OI_63/CO_21", "CII_158"]
        :type identifier: list of str

        '''
        if len(list(identifiers)) != 2:
            raise ValueError("Length of identifiers list must be exactly 2")
        models = [self._modelset.get_model(i) for i in identifiers]

        xlog,ylog=self._get_xy_from_wcs(models[0],quantity=True,linear=False)
        xlin,ylin=self._get_xy_from_wcs(models[0],quantity=True,linear=True)

        dcc=dens_clip.to(xlog.unit)
        rcc=rad_clip.to(ylog.unit)
        
        xi=np.where((xlin>=dcc[0]) & (xlin<=dcc[1]))[0]
        yi=np.where((ylin>=rcc[0]) & (ylin<=rcc[1]))[0]
        x2= np.hstack([np.where((np.round(xlog.value,1))==i)[0] for i in np.arange(-5,12)])
        # for 2020 models Y is not an integral value in erg s-1 cm-2
        # so rounding is necessary.
        y2 = np.hstack([np.where((np.round(ylog.value,1))==i)[0] for i in np.arange(-5,12)])
        xi2=np.intersect1d(xi,x2)
        yi2=np.intersect1d(yi,y2)
        
        self._figure,self._axis = self._plt.subplots(nrows=1,ncols=1)
        linesN=[]
        linesG=[]
        for j in xi2:
            label=np.round(np.log10(xlin[j].to(dens_clip.unit).value),1)
            if models[0].unit == '':
                m0label = models[0].title
            else:
                m0label = models[0].title + ' ['+u.Unit(models[0].unit).to_string('latex_inline')+']'
            if models[1].unit == '':
                m1label = models[1].title
            else:
                m1label = models[1].title + ' ['+u.Unit(models[1].unit).to_string('latex_inline')+']'
            if reciprocal[0]:
                xx=1/models[0][yi2[0]:yi2[-1]+1,j]
                self._axis.set_xlabel(utils.fliplabel(m0label))
            else:
                xx=models[0][yi2[0]:yi2[-1]+1,j]
                self._axis.set_xlabel(m0label)
            if reciprocal[1]:
                yy=1/models[1][yi2[0]:yi2[-1]+1,j]
                self._axis.set_ylabel(utils.fliplabel(m1label))
            else:
                yy=models[1][yi2[0]:yi2[-1]+1,j]
                self._axis.set_ylabel(m1label)
            linesN.extend(self._axis.loglog(xx,yy,label=label,lw=2))

        for j in yi2:
            label=np.round(np.log10(ylin[j].to(rad_clip.unit).value),1)
            if reciprocal[0]:
                xx=1/models[0][j,xi2[0]:xi2[-1]+1]
            else:
                xx=models[0][j,xi2[0]:xi2[-1]+1]
            if reciprocal[1]:
                yy=1/models[1][j,xi2[0]:xi2[-1]+1]
            else:
                yy=models[1][j,xi2[0]:xi2[-1]+1]
            linesG.extend(self._axis.loglog(xx,yy,label=label,lw=2,ls='--'))
            
        # create the column headers for the legend
        # and blank handles and labels to take up space for the headers and
        # when the number of density traces and radiation field traces
        # are not equal. 
        title1 = "log(n)"
        unit1="["+dens_clip.unit.to_string("latex_inline")+"]" 
        rs = rad_clip.unit.to_string()
        rsl = rad_clip.unit.to_string("latex_inline")
        title2 = "log("+utils.get_rad(rs)+")"
        unit2="["+rsl+"]"
        handles,labels=self._axis.get_legend_handles_labels()
        phantom = [self._axis.plot([],marker="", markersize=0,ls="",lw=0)[0]]*2
        lN = len(linesN)
        lG = len(linesG)
        diff = lN-lG
        adiff=abs(diff)
        phantom2 = [self._axis.plot([],marker="", markersize=0,ls="",lw=0)[0]]*adiff
        blank = ['']*adiff

        if diff == 0:
            labels.insert(lN,unit2)
            labels.insert(lN,title2)
            labels = [title1,unit1]+labels
            linesN = phantom + linesN
            linesG = phantom + linesG
        elif diff > 0: # more densities than radiation fields
            labels.insert(lN,unit2)
            labels.insert(lN,title2)
            labels = [title1,unit1]+labels + blank
            linesN = phantom + linesN
            linesG = phantom + linesG + phantom2  
        elif diff < 0: # more radiation fields than densities
            labels = labels[0:lN]+blank+labels[lN:]
            labels.insert(lN+adiff,unit2)
            labels.insert(lN+adiff,title2)
            labels = [title1,unit1]+labels
            linesN = phantom + linesN + phantom2
            linesG = phantom + linesG
        handles = linesN+linesG
        #for kk in range(len(handles)):
        #    print(handles[kk],labels[kk])

        leg=self._axis.legend(handles,labels,ncol=2,markerfirst=True,bbox_to_anchor=(1.024,1),loc="upper left")
        #leg._legend_box.align = "left"
        # trick to remove extra left side space in legend column headers.
        # doesn't completely center the headers, but gets as close as possible
        # See https://stackoverflow.com/questions/44071525/matplotlib-add-titles-to-the-legend-rows/44072076
        for vpack in leg._legend_handle_box.get_children():
            for hpack in vpack.get_children()[:2]:
                hpack.get_children()[0].set_width(0)
        #self._figure = self._plt.gcf()
        #self._axis = self._plt.gca()
        
    def _get_xy_from_wcs(self,data,quantity=False,linear=False):
        w = data.wcs
        xind=np.arange(w._naxis[0])
        yind=np.arange(w._naxis[1])
        # wcs methods want broadcastable arrays, but in our
        # case naxis1 != naxis2, so make two 
        # calls and take x from the one and y from the other.
        if quantity:
            x=w.array_index_to_world(xind,xind)[0]
            y=w.array_index_to_world(yind,yind)[1]
            # Need to handle Habing units which are non-standard FITS.
            # Can't apply them to a WCS because it will raise an Exception.
            # See ModelSet.get_model
            cunit=data.header.get("CUNIT2",None) 
            if cunit == "Habing":
               y._unit=utils.habing_unit     
            if linear:
               j = 10*np.ones(len(x.value))
               k = 10*np.ones(len(y.value))
               x = np.power(j,x.value)*x.unit
               y = np.power(k,y.value)*y.unit
        else:
            x=w.array_index_to_world_values(xind,xind)[0]
            y=w.array_index_to_world_values(yind,yind)[1]
            if linear:
               j = 10*np.ones(len(x))
               k = 10*np.ones(len(y))
               x = np.power(j,x)
               y = np.power(k,y)
        return (x,y)
    
        
    #@todo allow data to be an array? see overlay()
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
                       'xlim':None,
                       'ylim':None,
                       'legend': False,
                       'meas_color': ['#4daf4a'],
                       'shading': 0.4
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
            
        # make the x and y axes.  Since the models are computed on a log grid, we
        # use logarithmic ticks.
        x,y = self._get_xy_from_wcs(data,quantity=True,linear=True)
        locmaj = ticker.LogLocator(base=10.0, subs=(1.0, ),numticks=10)
        locmin = ticker.LogLocator(base=10.0, subs=np.arange(2, 10)*.1,numticks=10) 
        
        #allow unit conversion of density axis
        xax_unit = u.Unit(_header['cunit'+ax1])
        if kwargs_opts['xaxis_unit'] is not None:
            # Make density axis of the grid into a Quantity using the cunits from the grid header
            #temp_x = x * xax_unit

            # Get desired unit from arguments
            xax_unit = u.Unit(kwargs_opts['xaxis_unit'])

            # Convert the unit-aware grid to the desired units and set X to the value (so it's no longer a Quantity)
            #x = temp_x.to(xax_unit).value  
            x = x.to(xax_unit)

        # Set the x label appropriately, use LaTeX inline formatting
        xlab = r"{0} [{1:latex_inline}]".format(_header['ctype'+ax1],xax_unit)
        
        #allow unit conversion to cgs or Draine, for Y axis (FUV field):
        yax_unit = u.Unit(_header['cunit'+ax2])
        ytype = _header['ctype'+ax2]
        if kwargs_opts['yaxis_unit'] is not None:
            # Make FUV axis of the grid into a Quantity using the cunits from the grid header
            #temp_y = y * yax_unit

            # Get desired unit from arguments; for special cases, use
            # the conventional symbol for the label (e.g. G_0 for Habing units)
            yunit = kwargs_opts['yaxis_unit']
            ytype = "log({0})".format(utils.get_rad(yunit))
            yax_unit = u.Unit(yunit)

            # Convert the unit-aware grid to the desired units and set Y to the value (so it's no longer a Quantity)
            #y = temp_y.to(yax_unit).value  
            y = y.to(yunit)

        # Set the y label appropriately, use LaTeX inline formatting
        ylab = r"{0} [{1:latex_inline}]".format(ytype,yax_unit)
        print("X axis min/max %.2e %.2e"%(x.value.min(),x.value.max()))
        print("Y axis min/max %.2e %.2e"%(y.value.min(),y.value.max()))
        print("AXIndex=%d"%axidx)
        
        # Finish up axes details.
        self._axis[axidx].set_ylabel(ylab)
        self._axis[axidx].set_xlabel(xlab)
        if kwargs_opts['xlim'] is not None:
            xlim = kwargs_opts['xlim']
            self._axis[axidx].set_xlim(left=xlim[0],right=xlim[1])
        if kwargs_opts['ylim'] is not None:
            ylim = kwargs_opts['ylim']
            self._axis[axidx].set_ylim(bottom=ylim[0],top=ylim[1])
        self._axis[axidx].set_xscale('log')
        self._axis[axidx].set_yscale('log')
        self._axis[axidx].xaxis.set_major_locator(locmaj)
        self._axis[axidx].xaxis.set_minor_locator(locmin)
        self._axis[axidx].xaxis.set_minor_formatter(ticker.NullFormatter())

        if kwargs_opts['image']:
            # pass shading = auto to avoid deprecation warning
            # see https://matplotlib.org/3.3.0/gallery/images_contours_and_fields/pcolormesh_grids.html
            im = self._axis[axidx].pcolormesh(x.value,y.value,km,cmap=kwargs_imshow['cmap'],
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

            warnings.simplefilter('ignore',category=UserWarning)
            contourset = self._axis[axidx].contour(x.value,y.value,km.data, **kwargs_contour)
            warnings.resetwarnings()
            #print(contourset.__dict__)

            if kwargs_opts['label']:
                drawn = self._axis[axidx].clabel(contourset,contourset.levels,inline=True,fmt='%1.2e')

        if kwargs_opts['title'] is not None and not kwargs_opts['legend']:
            self._axis[axidx].set_title(kwargs_opts['title'])

        if measurements is None:
            mlen = 0
        else:
            mlen =  len(measurements)
        if len(kwargs_opts['meas_color']) < mlen:
            raise ValueError(f"Number of measurement colors (meas_color keyword) must match number of measurements ({mlen})")

        if measurements is not None:
            lstyles = ['--','-','--']
            # for serial calls to plot_no_wcs in an outside method (i.e. lineratioplot.overlay_all_ratios),
            # we need to keep track of the index for the measurement color, otherwise we always 
            # select color index 0, resulting in all measurements contours having the same color.
            # this is a kluge but it's all I can think of right now.
            if 'colorcounter' in kwargs:
                jj = kwargs['colorcounter']
            else:
                jj = 0
            for m in measurements:
                # for the case of colorcounter kluge len(m) will always be 1, so we don't
                # run into issues with incrementing of jj interfering with colorcounter.
                colors = kwargs_opts['meas_color'][jj]*mlen
                if kwargs_opts['shading'] != 0:
                    cset = self._axis[axidx].contourf(x.value,y.value,k.data,levels=m.levels, colors=colors,alpha=kwargs_opts['shading'])
                else:
                    cset = self._axis[axidx].contour(x.value,y.value,k.data,levels=m.levels, 
                                                     linestyles=lstyles, colors=colors)
                jj=jj+1

#def legend(self,labels,colors,loc='upper center',title=None,axindex=0):
#    lw = 3
#    ls = '-'
#    self._axis[axindex].legend(lin
