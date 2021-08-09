import numpy as np
import numpy.ma as ma
import scipy.stats as stats
import math

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
    """Class to plot various results from H2 Excitation diagram fitting.
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
                      'color':None}
        kwargs_opts.update(kwargs)
        cdavg = self._tool.average_column_density(norm=norm, position=position, size=size, line=False)
        energies = self._tool.energies(line=False)
        energy = np.array([c for c in energies.values()])
        colden = np.array([c.data for c in cdavg.values()])
        error = np.array([c.error for c in cdavg.values()])
        sigma = LOGE*error/colden
        self._figure,self._axis  = self._plt.subplots(figsize=kwargs_opts['figsize'])
        if self._tool.opr_fitted and show_fit:
            _label = "LTE"
        else:
            _label = '$'+self._label+'$ data'
        ec = self._axis.errorbar(energy,np.log10(colden),yerr=sigma,
                            fmt="o", capsize=kwargs_opts['capsize'],
                            label=_label, lw=kwargs_opts['linewidth'],
                            ms=kwargs_opts['markersize'],color=kwargs_opts['color'])
        if self._tool.opr_fitted and show_fit:
            # Plot only the odd-J ones!
            cddn = colden*self._tool._canonical_opr/self._tool.opr
            odd_index = np.where([isOdd(c) for c in cdavg.keys()])
            #color = ec.lines[0].get_color() # want these to be same color as data
            self._axis.errorbar(x=energy[odd_index], 
                                y=np.log10(cddn[odd_index]),marker="^",
                                label=f"OPR = {self._tool.opr.value:.2f}",
                                yerr=sigma[odd_index], 
                                capsize=2*kwargs_opts['capsize'],
                                linestyle='none',color='k',
                                lw=kwargs_opts['linewidth'],
                                ms=kwargs_opts['markersize'])
        self._axis.set_xlabel("$E_u/k$ (K)")
        if norm:
            self._axis.set_ylabel("log $(N_u/g_u) ~({\\rm cm}^{-2})$")
        else:
            self._axis.set_ylabel("log $(N_u) ~({\\rm cm}^{-2})$")
        # label the points with e.g. J=2,3,4...
        first=True
        for lab in sorted(cdavg):
            if first: 
                ss="J="+str(lab)
                first=False
            else: 
                ss=str(lab)
            self._axis.text(x=energies[lab]+100,y=np.log10(cdavg[lab]),s=ss)
        handles,labels=self._axis.get_legend_handles_labels()
        if show_fit:
            tt = self._tool
            if tt.fit_result is None:
                raise ValueError("No fit to show. Have you run the fit in your H2ExcitationFit?")
            # Doesn't work
            #if kwargs_opts['color'] is not None:
            #    self.colorcycle(self._CB_color_cycle[1:])
            x_fit = np.linspace(0,max(energy), 30)  
            outpar = tt.fit_result.params.valuesdict()
            labcold = r"$T_{cold}=$" + f"{tt.tcold.value:3.0f}" +r"$\pm$" + f"{tt.tcold.error:.1f} {tt.tcold.unit}"
            labhot= r"$T_{hot}=$" + f"{tt.thot.value:3.0f}"+ r"$\pm$" + f"{tt.thot.error:.1f} {tt.thot.unit}"
            labnh = r"$N("+self._label+")=" + float_formatter(tt.total_colden,2)+"$"
            self._axis.plot(x_fit,tt._one_line(x_fit, outpar['m1'], 
                            outpar['n1']), '.' ,label=labcold,
                            lw=kwargs_opts['linewidth'])
            self._axis.plot(x_fit,tt._one_line(x_fit, outpar['m2'], 
                            outpar['n2']), '.', label=labhot,
                            lw=kwargs_opts['linewidth'])
            opr_p = tt._fitresult.params['opr']
            self._axis.plot(x_fit, tt.fit_result.eval(x=x_fit,fit_opr=False), label="fit")
            handles,labels=self._axis.get_legend_handles_labels()
            #kluge to ensure N(H2) label is last
            phantom = self._axis.plot([],marker="", markersize=0,ls="",lw=0)
            handles.append(phantom[0])
            labels.append(labnh)
        self._axis.set_xlim(kwargs_opts['xmin'],kwargs_opts['xmax'])
        self._axis.set_ylim(kwargs_opts['ymin'],kwargs_opts['ymax'])
        self._axis.xaxis.set_major_locator(MultipleLocator(1000))
        self._axis.yaxis.set_major_locator(MultipleLocator(1))
        self._axis.xaxis.set_minor_locator(MultipleLocator(200))
        self._axis.yaxis.set_minor_locator(MultipleLocator(0.2))
        if kwargs_opts['grid']:
            self._axis.grid(b=True,which='both',axis='both',lw=kwargs_opts['linewidth'],
                            color='k',alpha=0.33)
            
        self._axis.legend(handles,labels)
