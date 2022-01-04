import numpy as np
import numpy.ma as ma
import scipy.stats as stats
import math
from copy import deepcopy,copy

import matplotlib.figure
import matplotlib.colors as mpcolors
import matplotlib.cm as mcm
from matplotlib import ticker
from matplotlib.lines import Line2D
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.axes as maxes
from matplotlib.ticker import MultipleLocator, FormatStrFormatter,AutoMinorLocator

from astropy.nddata.utils import Cutout2D
from astropy.io import fits
import astropy.wcs as wcs
import astropy.units as u
from astropy.nddata import NDDataArray, CCDData, NDUncertainty, StdDevUncertainty, VarianceUncertainty, InverseVariance
from astropy.visualization import simple_norm, ZScaleInterval , ImageNormalize
from astropy.visualization.stretch import SinhStretch,  LinearStretch
from matplotlib.colors import LogNorm


from .plotbase import PlotBase
from ..pdrutils import to,float_formatter,LOGE,isOdd

class ExcitationPlot(PlotBase):
    """
ExcitationPlot creates excitation diagrams  using the results of :class:`~pdrtpy.tool.h2excitationfit.H2ExcitationFit`. It can plot the observed excitation diagram with or without fit results, and allows averaging over user-given spatial areas.  
    """
    def __init__(self,tool,label):
        super().__init__(tool)
        self._xlim = []
        self._ylim = []
        self._label = label

    def ex_diagram(self,position=None,size=None,norm=True,show_fit=False,**kwargs):
        #@todo position and size might not necessarily match how the fit was done.
        #:type position: tuple or :class:`astropy.coordinates.SkyCoord`
        #or a :class:`~astropy.coordinates.SkyCoord`, which will use the :class:`~astropy.wcs.WCS` of the ::class:`~pdrtpy.measurement.Measurement`s added to this tool.
        r"""Plot the excitation diagram
        
        :param position: The position of the cutout array's center with respect to the data array. The position is specified as a `(x, y)` tuple of pixel coordinates. 
        :type position: tuple
        :param size: The size of the cutout array along each axis. If size is a scalar number or a scalar :class:`~astropy.units.Quantity`, then a square cutout of size will be created. If `size` has two elements, they should be in `(ny, nx)` order. Scalar numbers in size are assumed to be in units of pixels. `size` can also be a :class:`~astropy.units.Quantity` object or contain :class:`~astropy.units.Quantity` objects. Such :class:`~astropy.units.Quantity` objects must be in pixel or angular units. For all cases, size will be converted to an integer number of pixels, rounding the the nearest integer.  See :class:`~astropy.nddata.utils.Cutout2D`
        :type size: int, array_like, or :class:`astropy.units.Quantity`
        :param norm: if True, normalize the column densities by the 
                       statistical weight of the upper state, :math:`g_u`.  
        :type norm: bool
        :param show_fit: Show the most recent fit done the the associated H2ExcitationFit tool. 
        :type show_fit: bool
        """
        kwargs_opts = {'xmin':0.0,
                      'xmax':5000.0, #@TODO this should scale with max(energy)
                      'ymax':22,
                      'ymin': 15,
                      'grid' :False,
                      'figsize':(10,7),
                      'capsize':3,
                      'linewidth': 2.0,
                      'markersize': 8,
                      'color':None,
                      'axis':None}
        kwargs_opts.update(kwargs)

        cdavg = self._tool.average_column_density(norm=norm, position=position, size=size, line=False)
        energies = self._tool.energies(line=False)
        energy = np.array([c for c in energies.values()])
        colden = np.squeeze(np.array([c.data for c in cdavg.values()]))
        error = np.squeeze(np.array([c.error for c in cdavg.values()]))
        sigma = LOGE*error/colden
        if kwargs_opts['axis'] is None:
            self._figure, self._axis = self._plt.subplots(figsize=kwargs_opts['figsize'])
            kwargs_opts['axis'] = self._axis
        if self._tool.opr_fitted and show_fit:
            _label = "LTE"
        else:
            _label = '$'+self._label+'$ data'
        ec = kwargs_opts['axis'].errorbar(energy,np.log10(colden),yerr=sigma,
                            fmt="o", capsize=kwargs_opts['capsize'],
                            label=_label, lw=kwargs_opts['linewidth'],
                            ms=kwargs_opts['markersize'],color=kwargs_opts['color'])
        tt = self._tool
        if self._tool.opr_fitted and show_fit:
            # Plot only the odd-J ones!
            if position is not None:
                opr_p = tt._fitresult[position].params['opr'].value
            else:
                opr_p = tt.opr.value[0]
            cddn = colden*self._tool._canonical_opr/opr_p
            odd_index = np.where([isOdd(c) for c in cdavg.keys()])
            #color = ec.lines[0].get_color() # want these to be same color as data
            kwargs_opts['axis'].errorbar(x=energy[odd_index], 
                                y=np.log10(cddn[odd_index]),marker="^",
                                label=f"OPR = {opr_p:.2f}",
                                yerr=sigma[odd_index], 
                                capsize=2*kwargs_opts['capsize'],
                                linestyle='none',color='k',
                                lw=kwargs_opts['linewidth'],
                                ms=kwargs_opts['markersize'])
        kwargs_opts['axis'].set_xlabel("$E_u/k$ (K)")
        if norm:
            kwargs_opts['axis'].set_ylabel("log $(N_u/g_u) ~({\\rm cm}^{-2})$")
        else:
            kwargs_opts['axis'].set_ylabel("log $(N_u) ~({\\rm cm}^{-2})$")
        # label the points with e.g. J=2,3,4...
        first=True
        for lab in sorted(cdavg):
            if first: 
                ss="J="+str(lab)
                first=False
            else: 
                ss=str(lab)
            kwargs_opts['axis'].text(x=energies[lab]+100,y=np.log10(cdavg[lab]),s=ss)
        handles,labels=kwargs_opts['axis'].get_legend_handles_labels()
        if show_fit:
            if tt.fit_result is None:
                raise ValueError("No fit to show. Have you run the fit in your H2ExcitationFit?")
            if np.shape(tt.fit_result.data) == (1,):
                position = 0
            elif position is None:
                raise ValueError("position must be provided for map fit results")
            if tt.fit_result[position] is None:
                raise ValueError(f"The Excitation Tool was unable to fit pixel {position}. Try show_fit=False")
            x_fit = np.linspace(0,max(energy), 30)  
            outpar = tt.fit_result[position].params.valuesdict()
            labcold = r"$T_{cold}=$" + f"{tt.tcold[position]:3.0f}" +r"$\pm$" + f"{tt.tcold.error[position]:.1f} {tt.tcold.unit}"
            #labcold = r"$T_{cold}=$" + f"{tt.tcold[position]:3.1f}"
            #labhot= r"$T_{hot}=$" + f"{tt.thot.value:3.0f}"+ r"$\pm$" + f"{tt.thot.error:.1f} {tt.thot.unit}"
            #labhot= r"$T_{hot}=$" + f"{tt.thot[position]:3.1f}"
            labhot= r"$T_{hot}=$" + f"{tt.thot[position]:3.0f}"+ r"$\pm$" + f"{tt.thot.error[position]:.1f} {tt.thot.unit}"
            if position == 0:
                labnh = r"$N("+self._label+")=" + float_formatter(tt.total_colden,2)+"$" 
            else:
                labnh = r"$N("+self._label+")=" + float_formatter(u.Quantity(tt.total_colden[position],tt.total_colden.unit),2)+"$"
            kwargs_opts['axis'].plot(x_fit,tt._one_line(x_fit, outpar['m1'], 
                            outpar['n1']), '.' ,label=labcold,
                            lw=kwargs_opts['linewidth'])
            kwargs_opts['axis'].plot(x_fit,tt._one_line(x_fit, outpar['m2'], 
                            outpar['n2']), '.', label=labhot,
                            lw=kwargs_opts['linewidth'])

            kwargs_opts['axis'].plot(x_fit, tt.fit_result[position].eval(x=x_fit,fit_opr=False), label="fit")
            handles,labels=kwargs_opts['axis'].get_legend_handles_labels()
            #kluge to ensure N(H2) label is last
            phantom = kwargs_opts['axis'].plot([],marker="", markersize=0,ls="",lw=0)
            handles.append(phantom[0])
            labels.append(labnh)

        kwargs_opts['axis'].set_xlim(kwargs_opts['xmin'],kwargs_opts['xmax'])
        kwargs_opts['axis'].set_ylim(kwargs_opts['ymin'],kwargs_opts['ymax'])
        kwargs_opts['axis'].xaxis.set_major_locator(MultipleLocator(1000))
        kwargs_opts['axis'].yaxis.set_major_locator(MultipleLocator(1))
        kwargs_opts['axis'].xaxis.set_minor_locator(MultipleLocator(200))
        kwargs_opts['axis'].yaxis.set_minor_locator(MultipleLocator(0.2))
        if kwargs_opts['grid']:
            kwargs_opts['axis'].grid(b=True,which='both',axis='both',lw=kwargs_opts['linewidth'],
                            color='k',alpha=0.33)
            
        kwargs_opts['axis'].legend(handles,labels)
    
    def temperature(self,component,**kwargs):
        """Plot the temperature of hot or cold gas component.
        
        :param component: 'hot' or 'cold'
        :type component: str
        """
        if component not in self._tool.temperature:
            raise KeyError(f"{component} not a valid component. Must be one of {list(self._tool.temperature.keys())}")
        self._plot(self._tool.temperature[component],**kwargs)
        
    def column_density(self,component,log=True,**kwargs):
        """Plot the column density of hot or cold gas component, or total column density.
        
        :param component: 'hot', 'cold', or 'total
        :type component: str
        :param log: take the log10 of the column density before plotting
        """
        self._plot(self._tool.colden(component),log=log,**kwargs)
           
    def opr(self,**kwargs):
        if type(self._tool.opr) == float:
            return self._tool.opr
        self._plot(self._tool.opr,**kwargs)
        
    def _plot(self,data,**kwargs):
        '''generic plotting method used by other plot methods'''

        kwargs_plot = {'show' : 'data' # or 'mask' or 'error'
                      } 

        kwargs_opts = {'units' : None,
                       'image':True,
                       'colorbar': True,
                       'contours': True,
                       'label': False,
                       'title': None,
                       'norm': 'simple',
                       'log': False,
                       'axis': None
                       }

        kwargs_contour = {'levels': None, 
                          'colors': ['white'],
                          'linewidths': 1.0}

        # Merge in any keys the user provided, overriding defaults.
        kwargs_contour.update(kwargs)
        kwargs_opts.update(kwargs)
        kwargs_plot.update(kwargs)

        _data = deepcopy(data)  # default is show the data

        if kwargs_plot['show'] == 'error':
            _data.data = _data.error
        # do the log here, because we won't take log of a mask.
        if kwargs_opts['log']:
            _data.data = np.log10(_data.data)
        kwargs_opts.pop('log',None)
        kwargs.pop('log',None)
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
                          'constrained_layout': False # this appears to have no effect
                         }

        # delay merge until min_ and max_ are known
        kwargs_imshow.update(kwargs)
        kwargs_imshow['norm']=self._get_norm(kwargs_imshow['norm'],km, 
                                             kwargs_imshow['vmin'],kwargs_imshow['vmax'],
                                             kwargs_imshow['stretch'])

        kwargs_subplot.update(kwargs)
        # swap ncols and nrows in figsize to preserve aspect ratio
        kwargs_subplot['figsize'] = kwargs.get("figsize",(kwargs_subplot["ncols"]*5,kwargs_subplot["nrows"]*5))

        axidx = kwargs_subplot['index']-1
        if kwargs_opts['axis'] is None:
            self._figure,self._axis = self._plt.subplots(kwargs_subplot['nrows'],kwargs_subplot['ncols'],
                                                    figsize=kwargs_subplot['figsize'],
                                                    subplot_kw={'projection':k.wcs,
                                                                'aspect':kwargs_imshow['aspect']},
                                                    constrained_layout=kwargs_subplot['constrained_layout'])
            _axis = self._axis
        else:
            _axis = kwargs_opts['axis']

        # Make sure self._axis is an array because we will index it below.
        if type(_axis) is not np.ndarray:
            _axis= np.array([_axis])
        for a in _axis:
            a.tick_params(axis='both',direction='in') # axes vs axis???
            if hasattr(a,'coords'):
                for c in a.coords:
                    c.display_minor_ticks(True)
        if kwargs_opts['image']:
            current_cmap = copy(mcm.get_cmap(kwargs_imshow['cmap']))
            current_cmap.set_bad(color='white',alpha=1)
            # suppress errors and warnings about unused keywords
            #@todo need a better solution for this, it is not scalable.
            #push onto a stack?
            for kx in ['units', 'image', 'contours', 'label', 'title','linewidths','levels','nrows','ncols', 
                       'index', 'reset','colors','colorbar','show','yaxis_unit','xaxis_unit','axis',
                       'constrained_layout','figsize','stretch','legend','markersize','show_fit']:
                kwargs_imshow.pop(kx,None)
            # eliminate deprecation warning.  vmin,vmax are passed to Normalization object.
            if kwargs_opts['norm'] is not None:
                kwargs_imshow.pop('vmin',None)
                kwargs_imshow.pop('vmax',None)
            im=_axis[axidx].imshow(km,**kwargs_imshow)
            if kwargs_opts['colorbar']:
                self._wcs_colorbar(im,_axis[axidx])

        if kwargs_opts['contours']:
            if kwargs_contour['levels'] is None:
                # Figure out some autolevels 
                kwargs_contour['levels'] = self._autolevels(km,'log')

            # suppress errors and warnings about unused keywords
            for kx in ['units', 'image', 'contours', 'label', 'title', 'cmap','aspect',
                       'colorbar','reset', 'nrows', 'ncols', 'index','show','yaxis_unit','axis',
                       'xaxis_unit','norm','constrained_layout','figsize','stretch','legend','markersize','show_fit']:
                kwargs_contour.pop(kx,None)

            contourset = _axis[axidx].contour(km, **kwargs_contour)
            if kwargs_opts['label']:
                _axis[axidx].clabel(contourset,contourset.levels,inline=True,fmt='%1.1e')

        if kwargs_opts['title'] is not None: 
            #self.figure.subplots_adjust(top=0.95)
            #self._axis[axidx].set_title(kwargs_opts['title'])
            # Using ax.set_title causes the title to be cut off.  No amount of
            # diddling with tight_layout, constrained_layout, subplot adjusting, etc
            # would affect this.  However using Figure.suptitle seems to work.
            self.figure.suptitle(kwargs_opts['title'],y=0.95)

        if k.wcs is not None:
            _axis[axidx].set_xlabel(k.wcs.wcs.lngtyp)
            _axis[axidx].set_ylabel(k.wcs.wcs.lattyp)


    def explore(self,data=None,interaction_type="click",**kwargs): 
        kwargs_opts = {'units' : None,
                       'image':True,
                       'colorbar': True,
                       'contours': False,
                       'label': False,
                       'title': None,
                       'norm': 'simple',
                       'log': False,
                       'show_fit': True,
                       'figsize': (5,3),
                       'markersize': 20,
                       'fmt': 'r+'
                       }
        # starting position is middle pixel of image. note // for integer arithmetic
        position = tuple(np.array(np.shape(data))//2)
        kwargs_opts.update(kwargs)
        self._figure = self._plt.figure(figsize=kwargs_opts['figsize'],clear=True)
        self._axis = np.empty([2],dtype=object)
        self._axis[0] = self._figure.add_subplot(121,projection=data.wcs,aspect='auto')
        self._axis[1] = self._figure.add_subplot(122,projection=None,aspect='auto')
        self._axis[1].tick_params('y',labelright=True,labelleft=False) # avoid overlap with colorbar
        self._axis[1].get_yaxis().set_label_position("right")
        fmt      = kwargs_opts.pop('fmt','r+')
        show_fit = kwargs_opts.pop('show_fit')
        self._plot(data,axis=self._axis,index=1,**kwargs_opts)
        self.ex_diagram(axis=self._axis[1], reset=False,position=position,size=1,
                        norm=True,show_fit=show_fit)

        self._marker = self.axis[0].plot(position[0],position[1],fmt,markersize=kwargs_opts['markersize'])

        
        def update_lines(event):
            try:
                #self._logfile = open(f"/tmp/test.log","a")
               # self._logfile.write(f"event.inaxes = {event.inaxes} x,y={event.xdata,event.ydata}\n")
                if event.inaxes == self._axis[0]:  # the click must be on the left panel (map)
                    position = (int(round(event.xdata)),int(round(event.ydata)))
                    self._marker[0].set_marker(None)
                    self._marker = self.axis[0].plot(position[0],position[1],fmt,markersize=kwargs_opts['markersize'])
                    self._axis[1].clear()
                    self._axis[1].remove()
                    self._axis[1] = self._figure.add_subplot(122,projection=None,aspect='auto')
                    self._axis[1].tick_params('y',labelright=True,labelleft=False) 
                    self._axis[1].get_yaxis().set_label_position("right")
                    self.ex_diagram(axis=self._axis[1], reset=False,position=position,size=1,figsize=(5,3),
                                norm=True,show_fit=show_fit)
                    #self._axis[0].set_title(f'{position}')
            except Exception as err:
                pass
                #self._logfile.write("Exception {0}".format(err))
                
            #self._logfile.close()
            self._figure.canvas.draw_idle()
            
        if interaction_type == "move":
            self._figure.canvas.mpl_connect("motion_notify_event", update_lines)
        elif interaction_type == "click":
            print("connect button press to update_lines")
            self._figure.canvas.mpl_connect("button_press_event", update_lines)
        else:
            close(self._figure)
            raise ValueError(
                f"{interaction_type} is not a valid option for interaction_type, valid options are 'click' or 'move'"
            )
                
        