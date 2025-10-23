from abc import ABC
from copy import copy, deepcopy

# import astropy.version
import matplotlib.axes as maxes
import numpy as np
import numpy.ma as ma
from astropy.visualization import ImageNormalize, ZScaleInterval, simple_norm
from astropy.visualization.stretch import (
    AsinhStretch,
    LinearStretch,
    LogStretch,
    PowerStretch,
    SqrtStretch,
)
from cycler import cycler
from matplotlib.colors import LogNorm
# import matplotlib.cm as mcm
from mpl_toolkits.axes_grid1 import make_axes_locatable

from .. import pdrutils as utils


class PlotBase(ABC):
    """Base class for plotting.

    :param tool:  Reference to a :mod:`~pdrtpy.tool` object or `None`.  This is used for classes that inherit from PlotBase and are coupled to a specific tool, e.g. :class:`~pdrtpy.plot.LineRatioPlot` and :class:`~pdrtpy.tool.LineRatioFit`.
    :type tool: Any class derived from :class:`~pdrtpy.tool.toolbase.ToolBase`
    """

    def __init__(self, tool):
        import matplotlib.pyplot

        self._plt = matplotlib.pyplot
        # don't use latex in text labels etc by default.
        # because legends and titles wind up using a different font than axes
        # @TODO figure out how to make them all use the same font (e.g. CMBright)
        self._plt.rcParams["text.usetex"] = False
        self._figure = None
        self._axis = None
        self._tool = tool
        self._valid_norms = ["simple", "zscale", "log"]
        self._valid_stretch = ["linear", "sqrt", "power", "log", "asinh"]
        # color blind/friendly color cyle courtesy https://gist.github.com/thriveth/8560036
        # also added some from matplotlib 'tableau-colorblind10'
        self._CB_color_cycle = [
            "#377eb8",
            "#ff7f00",
            "#4daf4a",
            "#f781bf",
            "#a65628",
            "#984ea3",
            "#999999",
            "#e41a1c",
            "#dede00",
            "#595959",
            "#5F9ED1",
            "#C85200",
            "#898989",
            "#A2C8EC",
            "#FFBC79",
            "#CFCFCF",
        ]
        self.colorcycle(self._CB_color_cycle)

    def _autolevels(self, data, steps="log", numlevels=None, verbose=False):
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
        # print(type(data))
        max_ = data.max()
        min_ = data.min()
        if min_ <= 0:
            min_ = 1e-10
        # print("Auto contour levels: min %f max %f"%(min_,max_))
        if numlevels is None:
            try:
                numlevels = int(0.5 + 3 * (np.log(max_) - np.log(min_)) / np.log(10))
            except ValueError:
                print(f"Bad numlevels with [min,max]=[{min_},{max_}]")
                raise
        # print("levels start %d levels"%numlevels)
        # force number of levels to be between 5 and 15
        numlevels = max(numlevels, 5)
        numlevels = min(numlevels, 15)

        if steps[0:3] == "lin":
            slope = (max_ - min_) / (numlevels - 1)
            levels = np.array([min_ + slope * j for j in range(0, numlevels)])
        elif steps[0:3] == "log":
            # if data minimum is non-positive (shouldn't happen for models),
            # , min_cut=min_,max_cut=max_, stretch='log', clip=False) start log contours at lgo10(1) = 0
            if min_ <= 0:
                min_ = 1
            slope = np.log10(max_ / min_) / (numlevels - 1)
            levels = np.array([min_ * np.power(10, slope * j) for j in range(0, numlevels)])
        else:
            raise ValueError("steps must be 'lin' or 'log'")
        if verbose:
            print("Computed %d contour autolevels: %s" % (numlevels, levels))
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

    def text(self, x, y, s, fontdict=None, **kwargs):
        """
        Add text to the Axes.  Add the text `s` to the Axes at location `x, y` in data coordinates.
        This calls through to :meth:`matplotlib.pyplot.text`.

        :param x: the horizontal coordinate for the text
        :type x: float
        :param y: the vertical coordinate for the text
        :type y: float
        :param s: the text
        :type s: str
        :param fontdict: A dictionary to override the default text properties. If fontdict is None, the defaults are determined by rcParams.
        :type fontdict: dict
        :param \*\*kwargs: Other miscellaneous :class:`~matplotlib.text.Text` parameters.
        """
        n = self._plt.text(x, y, s, fontdict, **kwargs)

    def _zscale(self, image, vmin, vmax, stretch, contrast=0.25):
        """Normalization object using Zscale algorithm
           See :mod:`astropy.visualization.ZScaleInterval`

        :param image: the image object
        :type image: :mod:`astropy.io.fits` HDU or CCDData
        :param contrast: The scaling factor (between 0 and 1) for determining the minimum and maximum value. Larger values increase the difference between the minimum and maximum values used for display. Defaults to 0.25.
        :type contrast: float
        :returns: :mod:`astropy.visualization.normalization` object
        """
        # clip=False required or NaNs get max color value, see https://github.com/astropy/astropy/issues/8165
        if stretch == "linear":
            s = LinearStretch()
        elif stretch == "sqrt":
            s = SqrtStretch()
        elif stretch == "power":
            s = PowerStretch(2)
        elif stretch == "log":
            s = LogStretch(1000)
        elif stretch == "asinh":
            s = AsinhStretch(0.1)
        else:
            raise ValueError(f"Unknown stretch: {stretch}.")

        norm = ImageNormalize(
            data=image, vmin=vmin, vmax=vmax, interval=ZScaleInterval(contrast=contrast), stretch=s, clip=False
        )
        return norm

    def _get_norm(self, norm, km, vmin, vmax, stretch):
        """Get a Normalization object

        :param norm: The normalization time ( 'simple', 'zscale', 'log' )
        :type norm: str
        :param km: the image object
        :type km: :mod:`astropy.io.fits` HDU or CCDData
        :param vmin: the image minimum to use
        :type vmin: float
        :param vmax: the image maximum to use
        :type vmax: float
        :param stretch: the stretch to use (linear,log,power, asinh)
        :type stretch: str
        :returns: :mod:`astropy.visualization.normalization` object
        """
        if isinstance(norm, str):
            norm = norm.lower()
            if norm not in self._valid_norms:
                raise ValueError("Unrecognized normalization %s. Valid values are %s" % (norm, self._valid_norms))
        if stretch not in self._valid_stretch:
            raise ValueError("Unrecognized stretch %s. Valid values are %s" % (stretch, self._valid_stretch))
        # print("norm cut at %.1e %.1e"%(vmin,vmax))
        if norm == "simple":
            # astropy made a non-backwards compatible argument name change.
            # if astropy.version.major > 6 or astropy.version.version[0:3] == "6.1":
            #    return simple_norm(km, vmin=vmin, vmax=vmax, stretch=stretch, clip=False)
            # else:
            # @deprecated_renamed_argument should fix this in astropy 6.1+
            return simple_norm(km, min_cut=vmin, max_cut=vmax, stretch=stretch, clip=False)
        elif norm == "zscale":
            return self._zscale(km, vmin, vmax, stretch)
        elif norm == "log":
            # stretch ignored in this case
            return LogNorm(vmin=vmin, vmax=vmax, clip=False)
        else:
            return norm

    def _wcs_colorbar(self, image, axis, pos="right", width="5%", pad=0.05, orientation="vertical"):
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
        # This makes the colorbar the correct height but then offsets it from the x axis by a large amount.
        # Changing pad, even to a negative number, does not affect this.:w
        # ax_cb = divider.new_horizontal(size=width,pad=pad)
        # ax_cb.yaxis.set_ticks_position(pos)
        # self._figure.add_axes(ax_cb)
        cax = divider.append_axes(pos, size=width, pad=pad, axes_class=maxes.Axes)
        cax.yaxis.set_ticks_position(pos)
        return self._figure.colorbar(image, ax=axis, cax=cax, orientation=orientation)

    def savefig(self, fname, **kwargs):
        """Save the current figure to a file.

        :param fname: filename to save in
        :type fname: str

        :Keyword Arguments:

        Additional arguments (\*\*kwargs) are passed to :meth:`matplotlib.pyplot.savefig`. e.g., **bbox_inches='tight'** for a tight layout.

        """
        kwargs_opts = {"bbox_inches": "tight", "transparent": False, "facecolor": "white"}
        kwargs_opts.update(kwargs)
        self._figure.savefig(fname=fname, **kwargs_opts)

    def usetex(self, use):
        """Control whether plots delegate rendering of fancy text components in axis labels and elsewhere to the system version of LaTeX or use matplotlib's rendering. This method sets
        matplotlib parameter `rcParams["text.usetex"]` in the local pyplot instance.  Note: You must have LaTeX installed on your system if setting this to True or an exception will be raised when you try to plot.

        :param use: whether to use LaTeX or not
        :type use: bool
        """
        self._plt.rcParams["text.usetex"] = use

    def colorcycle(self, colorcycle):
        """Set the plot color cycle for multi-trace plots.  The default color cycle is optimized for color-blind users.

        :param colorcycle: List of colors to use, typically a list of hex color strings.  This list will be passed to :meth:`matplotlib.pyplot.rc` as the *axes prop_cycle* parameter using :class:`matplotlib.cycler`.
        :type colorcycle: list
        """
        self._plt.rc("axes", prop_cycle=(cycler("color", colorcycle)))

    def reset_colorcycle(self):
        """Reset the color cycle to the default color-blind friendly one"""
        self.colorcycle(self._CB_color_cycle)

    def _plot(self, data, **kwargs):
        """generic plotting method used by other plot methods"""

        test = kwargs.pop("test", False)
        kwargs_plot = {"show": "data"}  # or 'mask' or 'error'

        kwargs_opts = {
            "units": None,
            "image": True,
            "colorbar": True,
            "contours": True,
            "label": False,
            "title": None,
            "log": False,
            "axis": None,
        }

        kwargs_contour = {"levels": None, "colors": ["white"], "linewidths": 1.0}

        # Merge in any keys the user provided, overriding defaults.
        kwargs_contour.update(kwargs)
        kwargs_opts.update(kwargs)
        kwargs_plot.update(kwargs)

        _data = deepcopy(data)  # default is show the data

        if kwargs_plot["show"] == "error":
            _data = deepcopy(data)
            _data.data = _data.error
        # do the log here, because we won't take log of a mask.
        if kwargs_opts["log"]:
            _data.data = np.log10(_data.data)
        kwargs_opts.pop("log", None)
        kwargs.pop("log", None)
        if kwargs_plot["show"] == "mask":
            _data = deepcopy(data)
            _data.data = _data.mask
            # can't contour a boolean
            kwargs_opts["contours"] = False

        if self._tool._modelnaxis == 2 or len(_data.shape) == 2:
            if kwargs_opts["units"] is not None:
                k = utils.to(kwargs_opts["units"], _data)
            else:
                k = _data
        elif self._tool._modelnaxis == 3:
            if kwargs_opts["units"] is not None:
                k = utils.to(kwargs_opts["units"], _data[0, :, :])
            else:
                k = _data[0, :, :]
        else:
            raise Exception("Unexpected model naxis: %d" % self._tool._modelnaxis)

        km = ma.masked_invalid(k)
        if getattr(k, "mask", None) is not None:
            km.mask = np.logical_or(k.mask, km.mask)
        # make sure nans don't affect the color map
        min_ = np.nanmin(km)
        max_ = np.nanmax(km)

        kwargs_imshow = {
            "origin": "lower",
            "norm": "simple",
            "stretch": "linear",
            "vmin": min_,
            "vmax": max_,
            "cmap": "plasma",
            "aspect": "auto",
        }

        kwargs_subplot = {
            "nrows": 1,
            "ncols": 1,
            "index": 1,
            "reset": True,
            "constrained_layout": False,  # this appears to have no effect
        }

        # delay merge until min_ and max_ are known
        kwargs_imshow.update(kwargs)
        kwargs_imshow["norm"] = self._get_norm(
            kwargs_imshow["norm"], km, kwargs_imshow["vmin"], kwargs_imshow["vmax"], kwargs_imshow["stretch"]
        )

        kwargs_subplot.update(kwargs)
        # swap ncols and nrows in figsize to preserve aspect ratio
        kwargs_subplot["figsize"] = kwargs.get("figsize", (kwargs_subplot["ncols"] * 5, kwargs_subplot["nrows"] * 5))

        axidx = kwargs_subplot["index"] - 1
        if kwargs_subplot["reset"] and kwargs_opts["axis"] is None:
            self._figure, self._axis = self._plt.subplots(
                kwargs_subplot["nrows"],
                kwargs_subplot["ncols"],
                figsize=kwargs_subplot["figsize"],
                subplot_kw={"projection": k.wcs, "aspect": kwargs_imshow["aspect"]},
                constrained_layout=kwargs_subplot["constrained_layout"],
            )

        if kwargs_opts["axis"] is not None:
            self._axis = kwargs_opts["axis"]
        if type(self._axis) is not np.ndarray:
            self._axis = np.array([self._axis])
        for a in self._axis:
            a.tick_params(axis="both", direction="in")  # axes vs axis???
            if hasattr(a, "coords"):
                for c in a.coords:
                    c.display_minor_ticks(True)
        if kwargs_opts["image"]:
            # current_cmap = copy(mcm.get_cmap(kwargs_imshow['cmap']))
            current_cmap = copy(self._plt.get_cmap(kwargs_imshow["cmap"]))
            current_cmap.set_bad(color="white", alpha=1)
            # suppress errors and warnings about unused keywords
            # @todo need a better solution for this, it is not scalable.
            # push onto a stack? or pop everything that is NOT related to imshow.
            for kx in [
                "units",
                "image",
                "contours",
                "label",
                "title",
                "linewidths",
                "levels",
                "nrows",
                "ncols",
                "test",
                "index",
                "reset",
                "colors",
                "colorbar",
                "show",
                "axis",
                "yaxis_unit",
                "xaxis_unit",
                "bbox_to_anchor",
                "loc",
                "constrained_layout",
                "figsize",
                "stretch",
                "legend",
                "markersize",
                "show_fit",
            ]:
                kwargs_imshow.pop(kx, None)
            # eliminate deprecation warning.  vmin,vmax are passed to Normalization object.
            if kwargs_imshow["norm"] is not None:
                kwargs_imshow.pop("vmin", None)
                kwargs_imshow.pop("vmax", None)
            im = self._axis[axidx].imshow(km, **kwargs_imshow)
            if kwargs_opts["colorbar"]:
                self._wcs_colorbar(im, self._axis[axidx])
                # reset the axis so that users can call plot._plt.whatever()
                self._plt.sca(self._axis[axidx])

        if kwargs_opts["contours"]:
            if kwargs_contour["levels"] is None:
                # Figure out some autolevels
                kwargs_contour["levels"] = self._autolevels(km, "log")

            # suppress errors and warnings about unused keywords
            for kx in [
                "units",
                "image",
                "contours",
                "label",
                "title",
                "cmap",
                "aspect",
                "colorbar",
                "reset",
                "nrows",
                "ncols",
                "index",
                "show",
                "yaxis_unit",
                "xaxis_unit",
                "norm",
                "constrained_layout",
                "figsize",
                "stretch",
                "legend",
                "markersize",
                "show_fit",
            ]:
                kwargs_contour.pop(kx, None)

            contourset = self._axis[axidx].contour(km, **kwargs_contour)
            if kwargs_opts["label"]:
                self._axis[axidx].clabel(contourset, contourset.levels, inline=True, fmt="%1.1e")

        if kwargs_opts["title"] is not None:
            # self.figure.subplots_adjust(top=0.95)
            # self._axis[axidx].set_title(kwargs_opts['title'])
            # Using ax.set_title causes the title to be cut off.  No amount of
            # diddling with tight_layout, constrained_layout, subplot adjusting, etc
            # would affect this.  However using Figure.suptitle seems to work.
            self.figure.suptitle(kwargs_opts["title"], y=0.95)

        if k.wcs is not None:
            self._axis[axidx].set_xlabel(k.wcs.wcs.lngtyp)
            self._axis[axidx].set_ylabel(k.wcs.wcs.lattyp)
