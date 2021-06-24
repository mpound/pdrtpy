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
from ..pdrutils import to,float_formatter,LOGE

class ExcitationPlot(PlotBase):
    """Class to plot various results from H2 Excitation diagram fitting.
    """
    def __init__(self,tool,label):
        super().__init__(tool)
        self._xlim = []
        self._ylim = []
        self._label = label

    def plot_diagram(self,position,size,norm=True,show_fit=False,test=True,**kwargs):
        r"""Plot the excitation diagram
        :param position: The position of the cutout array's center with respect to the data array. The position can be specified either as a `(x, y)` tuple of pixel coordinates or a :class:`~astropy.coordinates.SkyCoord`, which will use the :class:`~astropy.wcs.WCS` of the ::class:`~pdrtpy.measurement.Measurement`s added to this tool. See :class:`~astropy.nddata.utils.Cutout2D`.
        :type position: tuple or :class:`astropy.coordinates.SkyCoord`
        :param size: The size of the cutout array along each axis. If size is a scalar number or a scalar :class:`~astropy.units.Quantity`, then a square cutout of size will be created. If `size` has two elements, they should be in `(ny, nx)` order. Scalar numbers in size are assumed to be in units of pixels. `size` can also be a :class:`~astropy.units.Quantity` object or contain :class:`~astropy.units.Quantity` objects. Such :class:`~astropy.units.Quantity` objects must be in pixel or angular units. For all cases, size will be converted to an integer number of pixels, rounding the the nearest integer. See the mode keyword for additional details on the final cutout size.
        :type size: int, array_like, or :class:`astropy.units.Quantity`
        :param norm: if True, normalize the column densities by the 
                       statistical weight of the upper state, :math:`g_u`.  
        :type norm: bool
        """
        kwargs_opts = {'xmin':0.0,
                      'xmax':5000.0,
                      'ymax':22,
                      'ymin': 15,
                      'grid' :None,
                      'figsize':(10,7)}
        kwargs_opts.update(kwargs)
        cdavg = self._tool.average_column_density(norm=norm, position=position, size=size, line=False, test=test)
        energies = self._tool.energies(line=False)
        energy = np.array([c for c in energies.values()])
        #print("E ",energy)
        colden = np.array([c.data for c in cdavg.values()])
        #print("N ",colden)
        error = np.array([c.error for c in cdavg.values()])
        sigma = LOGE*error/colden
        self._figure,self._axis  = self._plt.subplots(figsize=kwargs_opts['figsize']) ##nrows=1,ncols=1)
        self._axis.errorbar(energy,np.log10(colden),yerr=sigma,
                            fmt="o", capsize=1,label=self._label+' data')
        if self._tool.opr_fitted:
            cdn = self._tool.average_column_density(norm=norm, position=position, size=size, line=False, test=False)
            cddn = np.array([c.data for c in cdn.values()])
            self._axis.scatter(x=energy,y=np.log10(cddn),marker="^",label="opr=3")
        self._axis.set_xlabel("$E_u/k$ (K)")
        if norm:
            self._axis.set_ylabel("log $(N_u/g_u) ~({\\rm cm}^{-2})$")
        else:
            self._axis.set_ylabel("log $(N_u) ~({\\rm cm}^{-2})$")
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
            if tt.fit_params is None:
                raise ValueError("No fit to show. Have you run the fit in your H2ExcitationTool?")

            #@TODO default limits should be based on input data. 
            x_fit = np.linspace(1, 5100, 30)  
            ma1, na1, ma2, na2 = tt.fit_params._params[0:4]
            #@TODO improve float_formatter to optionally handle measurement errors
            labcold = r"$T_{cold}=$"+f"{tt.tcold.value:3.0f}"+r"$\pm$"+f"{tt.tcold.error:.1f} {tt.tcold.unit}"
            labhot= r"$T_{hot}=$"+f"{tt.thot.value:3.0f}"+r"$\pm$"+f"{tt.thot.error:.1f} {tt.thot.unit}"
            labnh = r"$N("+self._label+")="+float_formatter(tt.total_colden,2)+"$"
            self._axis.plot(x_fit,tt._one_line(x_fit, ma1,na1),'.',label=labcold)
            self._axis.plot(x_fit,tt._one_line(x_fit, ma2,na2),'.',label=labhot)

            if tt.opr_fitted:
                tt._opr_mask = False*np.ones(x_fit.size)
                self._axis.plot(x_fit,tt._exc_func_opr(x_fit,
                            *tt.fit_params._params),label="sum")
                ma1, na1, ma2, na2= [-2.06876425e-03,  2.05430745e+01, -6.26653322e-04,  1.90774300e+01]
                self._axis.plot(x_fit,tt._one_line(x_fit, ma1,na1),'.',label="C1")
                self._axis.plot(x_fit,tt._one_line(x_fit, ma2,na2),'.',label="C2")      
            else:
                self._axis.plot(x_fit,tt._exc_func(x_fit,
                            *tt.fit_params._params[0:4]),label="sum")       
            print("PARAMS: ",tt.fit_params._params)
            handles,labels=self._axis.get_legend_handles_labels()

            #kluge to ensure N(H2) label is last
            phantom = self._axis.plot([],marker="", markersize=0,ls="",lw=0)
            handles.append(phantom[0])
            labels.append(labnh)
            if tt.opr_fitted:
                labopr = f"Fitted OPR={tt.opr:2.1f}"+r"$\pm$"+f"{tt.fit_params._perr[4]:0.1f}"
                ph2 = self._axis.plot([],marker="", markersize=0,ls="",lw=0)
                handles.append(ph2[0])
                labels.append(labopr)
        self._axis.set_xlim(kwargs_opts['xmin'],kwargs_opts['xmax'])
        self._axis.set_ylim(kwargs_opts['ymin'],kwargs_opts['ymax'])
        self._axis.xaxis.set_major_locator(MultipleLocator(1000))
        self._axis.yaxis.set_major_locator(MultipleLocator(1))
        self._axis.xaxis.set_minor_locator(MultipleLocator(200))
        self._axis.yaxis.set_minor_locator(MultipleLocator(0.2))
        if kwargs_opts['grid'] is not None:
            self._axis.grid(b=True,which='both',axis='both',lw=1,color='k',alpha=0.33)
            
        self._axis.legend(handles,labels)
