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
    def __init__(self,tool,**kwargs):
        super().__init__(tool)
        self.figure = None
        self.axis = None
        self._xlim = [None,None]
        self._ylim = [None,None]
        self._plotkwargs = kwargs

    def _plotimage(self,data,units,cmap,image,contours,levels,norm):
        k=to(units,data)
        km = ma.masked_invalid(k)
        min_ = km.min()
        max_ = km.max()
        ax=self._plt.subplot(111,projection=k.wcs,aspect='equal')
        if norm == 'simple':
            normalizer = simple_norm(km, min_cut=min_,max_cut=max_, stretch='log', clip=false)
        elif norm == 'zscale':
            normalizer = self._zscale(km)
        else: 
            normalizer = norm
        
        if image: 
            current_cmap = mcm.get_cmap(cmap)
            current_cmap.set_bad(color='white',alpha=1)
            self._plt.imshow(km,cmap=current_cmap,origin='lower',norm=normalizer)
            self._plt.colorbar()
        if contours:
            if image==false: colors='black'
            else: colors='white'
            if levels is none:
                # figure out some autolevels 
                steps='log'
                contourset = ax.contour(km, levels=self._autolevels(km,steps),colors=colors)
            else:
                contourset = ax.contour(km, levels=levels, colors=colors)
                # todo: add contour level labelling
                # See https://matplotlib.org/3.1.1/gallery/images_contours_and_fields/contour_label_demo.html
        if title is not None: self._plt.title(title)
        self._plt.xlabel(k.wcs.wcs.lngtyp)
        self._plt.ylabel(k.wcs.wcs.lattyp)

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
        

