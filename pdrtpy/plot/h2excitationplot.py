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
from ..pdrutils import to

class H2ExcitationPlot(PlotBase):
    """Class to plot various results from H2 Excitation diagram fitting.
    """
    def __init__(self,tool):
        super().__init__(tool)
        self._xlim = []
        self._ylim = []

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
        loge = math.log10(math.e)
        cdavg = self._tool.average_column_density(norm=norm,position=position,size=size,line=False,test=test)
        energies = self._tool.energies(line=False)
        energy = np.array([c for c in energies.values()])
        #print("E ",energy)
        colden = np.array([c.data for c in cdavg.values()])
        #print("N ",colden)
        error = np.array([c.error for c in cdavg.values()])
        sigma = loge*error/colden
        self._figure,self._axis =self._plt.subplots(nrows=1,ncols=1,**kwargs)
        self._axis.errorbar(energy,np.log10(colden),yerr=sigma,fmt="o", capsize=1,label='$H_2$ data')
        self._axis.set_xlabel("$E_u/k$ (K)")
        self._axis.set_ylabel("log $(N_u/g_u) ~({\\rm cm}^{-2})$")
        first=True
        for lab in sorted(cdavg):
            if first: 
                ss="J="+str(lab)
                first=False
            else: 
                ss=str(lab)
            self._axis.text(x=energies[lab]+100,y=np.log10(cdavg[lab]),s=ss)
        if show_fit:
            tt = self._tool
            x_fit = np.linspace(1, 5100, 30)  
            ma1, na1, ma2, na2 = tt._fitted_params[2]
            om1, on1, om2, on2 = tt._fitted_params[0]
            labcold = r"$T_{cold}=$"+f"{tt._tcold:3.0f}"
            labhot= r"$T_{hot}=$"+f"{tt._thot:3.0f}"

            self._axis.plot(x_fit,tt._one_line(x_fit, ma1,na1),'.',label=labcold)
            self._axis.plot(x_fit,tt._one_line(x_fit, ma2,na2),'.',label=labhot)
            self._axis.plot(x_fit,tt._x_lin(x_fit,*tt._fitted_params[2]),label="sum")

            self._axis.set_xlim(0,5000)
            self._axis.set_ylim(15,22)
            self._axis.xaxis.set_major_locator(MultipleLocator(1000))
            self._axis.yaxis.set_major_locator(MultipleLocator(1))
            self._axis.xaxis.set_minor_locator(MultipleLocator(200))
            self._axis.yaxis.set_minor_locator(MultipleLocator(0.2))
            
        
        self._axis.legend()
        

