import numpy as np

import matplotlib.axes as maxes
from mpl_toolkits.axes_grid1 import make_axes_locatable

from astropy.visualization import simple_norm, ZScaleInterval , ImageNormalize
from astropy.visualization.stretch import LinearStretch, SinhStretch, PowerStretch, AsinhStretch, LogStretch
from matplotlib.colors import LogNorm
from cycler import cycler

from ..pdrutils import to

class PlotBase:
    """Base class for plotting.  

    :param tool:  Reference to a :mod:`~pdrtpy.tool` object or `None`.  This is used for classes that inherit from PlotBase and are coupled to a specific tool, e.g. :class:`~pdrtpy.plot.LineRatioPlot` and :class:`~pdrtpy.tool.LineRatioFit`.
    :type tool: Any class derived from :class:`~pdrtpy.tool.toolbase.ToolBase`
    """
    def __init__(self,tool):
        import matplotlib.pyplot 
        self._plt = matplotlib.pyplot
        # don't use latex in text labels etc by default. 
        # because legends and titles wind up using a different font than axes
        # @TODO figure out how to make them all use the same font (e.g. CMBright)
        self._plt.rcParams["text.usetex"] = False
        self._figure = None
        self._axis = None
        self._tool = tool
        self._valid_norms = [ 'simple', 'zscale', 'log' ]
        self._valid_stretch = [ 'linear', 'sqrt', 'power', 'log', 'asinh']
        # color blind/friendly color cyle courtesy https://gist.github.com/thriveth/8560036
        self._CB_color_cycle = ['#377eb8', '#ff7f00','#4daf4a',
                  '#f781bf', '#a65628', '#984ea3',
                  '#999999', '#e41a1c', '#dede00']
        self.colorcycle(self._CB_color_cycle)

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
           raise ValueError("steps must be 'lin' or 'log'")
        if verbose:
            print("Computed %d contour autolevels: %s"%(numlevels,levels))
        return levels
        
    @property 
    def figure(self):
        """The last figure that was drawn.

           :rtype: :class:`matplotlib.figure.Figure`
        """
        return self._figure

    @property
    def axis(self):
        """The last axis that was drawn.

           :rtype: :class:`matplotlib.axes._subplots.AxesSubplot`
        """
        return self._axis

    def _zscale(self,image,vmin,vmax,stretch,contrast=0.25):
        """Normalization object using Zscale algorithm
           See :mod:`astropy.visualization.ZScaleInterval`
        
        :param image: the image object
        :type image: :mod:`astropy.io.fits` HDU or CCDData
        :param contrast: The scaling factor (between 0 and 1) for determining the minimum and maximum value. Larger values increase the difference between the minimum and maximum values used for display. Defaults to 0.25.  
        :type contrast: float
        :returns: :mod:`astropy.visualization.normalization` object
        """
        # clip=False required or NaNs get max color value, see https://github.com/astropy/astropy/issues/8165
        if stretch == 'linear':
            s=LinearStretch()
        elif stretch == 'sqrt':
            s = SqrtStretch()
        elif stretch == 'power':
            s = PowerStretch(1)
        elif stretch == 'log':
            s = LogStretch(1000)
        elif s == 'asinh':
            stretch = AsinhStretch(0.1)
        else:
            raise ValueError(f'Unknown stretch: {stretch}.')

        norm = ImageNormalize(data=image,vmin=vmin,vmax=vmax,interval=ZScaleInterval(contrast=contrast),stretch=s,clip=False)
        return norm

    def _get_norm(self,norm,km,vmin,vmax,stretch):
        if type(norm) == str: 
            norm = norm.lower()
            if norm not in self._valid_norms:
                raise ValueError("Unrecognized normalization %s. Valid values are %s"%(norm,self._valid_norms))
        if stretch not in self._valid_stretch:
            raise ValueError("Unrecognized stretch %s. Valid values are %s"%(stretch,self._valid_stretch))
        #print("norm cut at %.1e %.1e"%(vmin,vmax))
        if norm == 'simple':
            return simple_norm(km, min_cut=vmin,max_cut=vmax, stretch=stretch, clip=False)
        elif norm == 'zscale':
            return self._zscale(km,vmin,vmax,stretch)
        elif norm == 'log':
            # stretch ignored in this case
            return LogNorm(vmin=vmin,vmax=vmax,clip=False)
        else: 
            return norm

    def _wcs_colorbar(self,image, axis, pos="right", width="10%",pad=0.15,orientation="vertical"):
        """Create a colorbar for a subplot with WCSAxes 
           (as opposed to matplolib Axes).  There are some side-effects of
           using WCS projection that need to be ameliorated.  Also for 
           subplots, we want the colorbars to have the same height as the 
           plot, which is not the default behavior.

           :param image: the mappable object for the plot. Must not be masked.
           :type image: :obj:`numpy.ndarray`,:mod:`astropy.io.fits` HDU or CCDData
           :param axis: which Axes object for the plot
           :type axis:  :class:`matplotlib.axis.Axes`
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

    def savefig(self,fname,**kwargs):
        """Save the current figure to a file.

           :param fname: filename to save in
           :type fname: str

           :Keyword Arguments:

           Additional arguments (\*\*kwargs) are passed to :meth:`matplotlib.pyplot.savefig`. e.g. bbox_inches='tight' for a tight layout.

        """
        kwargs_opts = {'bbox_inches':'tight',
                       'transparent':False,
                       'facecolor':'white'
                      }
        kwargs_opts.update(kwargs)
        self._figure.savefig(fname=fname,**kwargs_opts)

    def usetex(self,use):
        """Control whether plots use LaTeX formatting in axis labels and other text components. This method sets
           matplotlib parameter `rcParams["text.usetex"]` in the local pyplot instance.

           :param use: whether to use LaTeX or not
           :type use: bool
        """
        self._plt.rcParams["text.usetex"] = use
        
    def colorcycle(self,colorcycle):
        """Set the plot color cycle for multi-trace plots.  The default color cycle is optimized for color-blind users. 
        
        :param colorcycle: List of colors to use, typically a list of hex color strings.  This list will be passed to :meth:`matplotlib.pyplot.rc` as the *axes prop_cycle* parameter using :class:`matplotlib.cycler`.
        :type colorcycle: list
        """
        self._plt.rc('axes', prop_cycle=(cycler('color',  colorcycle)))
        
