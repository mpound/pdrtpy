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

from .plotbase import PlotBase
from ..pdrutils import to

class H2ExcitationPlot(PlotBase):
    """Class to plot various results from H2 Excitation diagram fitting.
    """
    def __init__(self,tool):
        super().__init__(tool)
        self._xlim = []
        self._ylim = []

    def plot_diagram(self,x,y,xsize,ysize,norm=True):
        """Plot the excitation diagram

           :param norm: if True, normalize the column densities by the 
                       statistical weight of the upper state, :math:`g_u`.  
           :type norm: bool
           :param x: bottom left corner x 
           :type x: int
           :param y: bottom left corner y 
           :type y: int
           :param xsize: box width, pixels
           :type xsize: int
           :param ysize: box height, pixels
           :type ysize: int
           :param line: if True, the returned dictionary index is the Line name, otherwise it is the upper state :math:`J` number.  
           :type line: bool
 
        """
        cdavg = self._tool.average_column_density(norm,x,y,xsize,ysize,line=False)
        cdval = list(cdavg.values())*self._tool._cd_units
        energies = list(self._tool.energies(line=False).values())*u.K
        ax=self._plt.subplot(111)
        ax.scatter(x=energies,y=cdavg)
        

