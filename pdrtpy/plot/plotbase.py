#todo: Look into seaborn https://seaborn.pydata.org
# Also https://docs.bokeh.org/en
# especially for coloring and style

import numpy as np

import matplotlib.axes as maxes
from mpl_toolkits.axes_grid1 import make_axes_locatable

from astropy.visualization import ZScaleInterval, ImageNormalize
from astropy.visualization.stretch import LinearStretch

from .pdrutils import to

rad_title = dict()
rad_title['Habing'] = '$G_0$'
rad_title['Draine'] = '$\chi$'
rad_title['Mathis'] = 'ISRF$_{Mathis}$'

class PlotBase:
    def __init__(self,tool):
        import matplotlib.pyplot as plt
        self._plt = matplotlib.pyplot
        self._figure = None
        self._axis = None
        self._tool = tool
        #print("Done PlotBase")

    def _autolevels(self,data,steps='log',numlevels=None):
        """Compute contour levels automatically based on data. 

        :param data: The data to contour
        :type data: numpy.ndarray, astropy.io.fits HDU or CCDData
        :param steps: The type of steps to compute. "log" for logarithmic, or "lin" for linear. Defaut: log
        :type steps: str
        :param numlevels: The number of contour levels to compute. Default: None which means autocompute the number of levels which typically gives about 10 levels.
        :type numlevels: int
        :returns:  numpy.array containing level values
        """
 
        # tip of the hat to the WIP autolevels code lev.
        # http://admit.astro.umd.edu/wip ,  wip/src/plot/levels.c
        # CVS at http://www.astro.umd.edu/~teuben/miriad/install.html
        #print(type(data))
        max_ =data.max()
        min_ = data.min()
        #print("autolev min %f max %f"%(min_,max_))
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
        print("Computed %d contour autolevels: %s"%(numlevels,levels))
        return levels
        
    def _zscale(self,image):
        """Normalization object using Zscale algorithm
        
        :param image: the image object
        :type image: astropy.io.fits HDU or CCDData
        :returns: astropy.visualization.normalization object
        """
        # clip=False required or NaNs get max color value, see https://github.com/astropy/astropy/issues/8165
        norm= ImageNormalize(data=image,interval=ZScaleInterval(contrast=0.5),stretch=LinearStretch(),clip=False)
        return norm

    def _wcs_colorbar(self,image, axis, pos="right", width="10%",pad=0.15,orientation="vertical"):
        """Create a colorbar for a subplot with WCSAxes 
           (as opposed to matplolib Axes).  There are some side-effects of
           using WCS procjection that need to be ameliorated.  Also for 
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

