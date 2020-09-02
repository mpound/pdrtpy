"""Base class for tool plotters
"""
import numpy as np

import matplotlib.axes as maxes
from mpl_toolkits.axes_grid1 import make_axes_locatable

from astropy.visualization import simple_norm, ZScaleInterval , ImageNormalize
from astropy.visualization.stretch import LinearStretch
from astropy.visualization.stretch import SinhStretch,  LinearStretch
from matplotlib.colors import LogNorm

from ..pdrutils import to

rad_title = dict()
rad_title['Habing'] = '$G_0$'
rad_title['Draine'] = '$\chi$'
rad_title['Mathis'] = 'ISRF$_{Mathis}$'

class PlotBase:
    def __init__(self,tool):
        import matplotlib.pyplot 
        self._plt = matplotlib.pyplot
        self._figure = None
        self._axis = None
        self._tool = tool
        self._valid_norms = [ 'simple', 'zscale', 'log' ]
        #print("Done PlotBase")

    def _autolevels(self,data,steps='log',numlevels=None,verbose=False):
        """Compute contour levels automatically based on data. 

        :param data: The data to contour
        :type data: numpy.ndarray, astropy.io.fits HDU or CCDData
        :param steps: The type of steps to compute. "log" for logarithmic, or "lin" for linear. Defaut: log
        :type steps: str
        :param numlevels: The number of contour levels to compute. Default: None which means autocompute the number of levels which typically gives about 10 levels.
        :type numlevels: int
        :param verbose: Print the computed levels. Default: False
        :type verbose: boolean
        :returns:  numpy.array containing level values
        """
 
        # tip of the hat to the WIP autolevels code lev.
        # http://admit.astro.umd.edu/wip ,  wip/src/plot/levels.c
        # CVS at http://www.astro.umd.edu/~teuben/miriad/install.html
        #print(type(data))
        max_ =data.max()
        min_ = data.min()
        if min_ == 0: min_ = 1E-10
        #print("Auto contour levels: min %f max %f"%(min_,max_))
        if numlevels is None:
            numlevels = int(0.5+3*(np.log(max_)-np.log(min_))/np.log(10))
        #print("levels start %d levels"%numlevels)
        # force number of levels to be between 5 and 15
        numlevels = max(numlevels,5)
        numlevels = min(numlevels,15)
    
        if steps[0:3] == 'lin':
            slope = (max_ - min_)/(numlevels-1)
            levels = np.array([min_+slope*j for j in range(0,numlevels)])
        elif steps[0:3] == 'log':
            # if data minimum is non-positive (shouldn't happen for models),
            #, min_cut=min_,max_cut=max_, stretch='log', clip=False) start log contours at lgo10(1) = 0
            if min_ <= 0: min_=1
            slope = np.log10(max_/min_)/(numlevels - 1)
            levels = np.array([min_ * np.power(10,slope*j) for j in range(0,numlevels)])
        else:
           raise Exception("steps must be 'lin' or 'log'")
        if verbose:
            print("Computed %d contour autolevels: %s"%(numlevels,levels))
        return levels
        
    def _zscale(self,image,contrast=0.25):
        """Normalization object using Zscale algorithm
           See :mod:`astropy.visualization.ZScaleInterval`
        
        :param image: the image object
        :type image: :mod:`astropy.io.fits` HDU or CCDData
        :param contrast: The scaling factor (between 0 and 1) for determining the minimum and maximum value. Larger values increase the difference between the minimum and maximum values used for display. Defaults to 0.25.  
        :type contrast: float
        :returns: :mod:`astropy.visualization.normalization` object
        """
        # clip=False required or NaNs get max color value, see https://github.com/astropy/astropy/issues/8165
        norm = ImageNormalize(data=image,interval=ZScaleInterval(contrast=contrast),stretch=LinearStretch(),clip=False)
        return norm

    def _get_norm(self,norm,km,min,max):
        if type(norm) == str: 
            norm = norm.lower()
            if norm not in self._valid_norms:
                raise Exception("Unrecognized normalization %s. Valid values are %s"%(norm,self._valid_norms))
        if norm == 'simple':
            return simple_norm(km, min_cut=min,max_cut=max, stretch='log', clip=False)
        elif norm == 'zscale':
            return self._zscale(km)
        elif norm == 'log':
            return LogNorm(vmin=min,vmax=max,clip=False)
        else: 
            return norm

    def _wcs_colorbar(self,image, axis, pos="right", width="10%",pad=0.15,orientation="vertical"):
        """Create a colorbar for a subplot with WCSAxes 
           (as opposed to matplolib Axes).  There are some side-effects of
           using WCS projection that need to be ameliorated.  Also for 
           subplots, we want the colorbars to have the same height as the 
           plot, which is not the default behavior.

           :param image: the mappable object for the plot. Must not be masked.
           :type image: numpy.ndarray, astropy.io.fits HDU or CCDData
           :param axis: which Axes object for the plot
           :type axis:  matplotlib.axis.Axes
           :param pos: colorbar position: ["left"|"right"|"bottom"|"top"]. Default: right
           :type pos: str
           :param width: width of the colorbar in terms of percent width of the plot.
           :type width: str 
           :param pad: padding between colorbar and plot, in inches.
           :type pad: float
           :param orientation: orientation of colorbar ["vertical" | "horizontal" ]
           :type orientation: str
        """
        divider = make_axes_locatable(axis)
        # See https://stackoverflow.com/questions/47060939/matplotlib-colorbar-and-wcs-projection
        cax = divider.append_axes(pos, size=width, pad=pad, axes_class=maxes.Axes)
        cax.yaxis.set_ticks_position(pos)
        return self._figure.colorbar(image,ax=axis,cax=cax)

    def savefig(fname,**kwargs):
        """Save the current figure to a file.

           :param fname: filename to save in
           :type fname: str

           :Keyword Arguments:

           Additional arguments (\*\*kwargs) are passed to :meth:`matplotlib.pyplot.savefig`.

        """
        self._figure.savefig(fname=fname,**kwargs)
