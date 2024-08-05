
Plot Keywords
-------------

To manage the plots, the methods in ``Plot`` classes take keywords (\*\*kwargs) that turn on or off various options, specify plot units, or map to matplotlib's :meth:`~matplotlib.axes.Axes.plot`, :meth:`~matplotlib.axes.Axes.imshow`, :meth:`~matplotlib.axes.Axes.contour` keywords.  The methods have reasonable defaults, so try them with no keywords to see what they do before modifying keywords.

     * *units* (``str`` or :class:`astropy.units.Unit`) image data units to use in the plot. This can be either a string such as, 'cm^-3' or 'Habing', or it can be an :class:`astropy.units.Unit`.  Data will be converted to the desired unit.   Note these are **not** the axis units, but the image data units.  Modifying axis units is implemented via the `xaxis_unit` and `yaxis_unit` keywords.

     * *image* (``bool``) whether or not to display the image map (imshow).

     * *show* (``str``) which quantity to display in the Measurement, one of 'data', 'error', 'mask'.  For example, this can be used to plot the errors in observed ratios. Default: 'data'


     * *cmap* (``str``) colormap name, Default: 'plasma'

     * *colorbar* (``str``) whether or not to display colorbar

     * *colors* (``str``) color of the contours. Default: 'whitecolor of the contours. Default: 'white'

     * *contours* (``bool``), whether or not to plot contours

     * *label* (``bool``), whether or not to label contours

     * *linewidths* (``float or sequence of float``), the line width in points, Default: 1.0

     * *legend* (``bool``) Draw a legend on the plot. If False, a title is drawn above the plot with the value of the *title* keyword

     * *bbox_to_anchor* (``tuple``) The `matplotlib` legend keyword for controlling the placement of the legend. See the `matplotlib Legend Guide <https://matplotlib.org/stable/tutorials/intermediate/legend_guide.html>`_

     * *loc* (``str``)  The `matplotlib` legend keyword for controlling the location of the legend. See :meth:`~matplotlib.axes.Axes.legend`.

     * *levels* (``int`` or array-like) Determines the number and positions of the contour lines / regions.  If an int n, use n data intervals; i.e. draw n+1 contour lines. The level heights are automatically chosen.  If array-like, draw contour lines at the specified levels. The values must be in increasing order.

     * *measurements* (array-like) A list of single pixel Measurements that can be contoured over a model ratio or intensity map.

     * *meas_color* (array of str) A list of colors to use when overlaying Measurement contours. There should be one color for each Measurement in the *measurement* keyword.  The Default of None will use a color-blind friendly color cycle.

     * *norm* (``str`` or :mod:`astropy.visualization` normalization object) The normalization to use in the image. The string 'simple' will normalize with :func:`~astropy.visualization.simple_norm` and 'zscale' will normalize with IRAF's zscale algorithm.  See :class:`~astropy.visualization.ZScaleInterval`.

     * *stretch* (``str``)  {'linear', 'sqrt', 'power', 'log', 'asinh'}. The stretch function to apply to the image for simple_norm.  The Default is 'linear'.

     * *aspect* (``str``) aspect ratio, 'equal' or 'auto' are typical defaults.

     * *origin* (``str``) Origin of the image. Default: 'lower'

     * *title* (``str``) A title for the plot.  LaTeX allowed.

     * *vmin*  (``float``) Minimum value for colormap normalization

     * *vmax*  (``float``) Maximum value for colormap normalization

     * *xaxis_unit* (``str`` or :class:`astropy.units.Unit`) X axis (density) units to use when plotting models, such as in :meth:`~pdrtpy.plot.lineratioplot.LineRatioPlot.overlay_all_ratios` or :meth:`~pdrtpy.plot.lineratioplot.LineRatioPlot.modelratio`.  If None, the native model axis units are used.

     * *yaxis_unit* (``str`` or :class:`astropy.units.Unit`) Y axis (density) units to use when plotting models, such as in :meth:`~pdrtpy.plot.lineratioplot.LineRatioPlot.overlay_all_ratios` or :meth:`~pdrtpy.plot.lineratioplot.LineRatioPlot.modelratio`.  If None, the native model axis units are used.

The following keywords are available, but you probably won't touch.

     * *nrows* (``int``) Number of rows in the subplot

     * *ncols* (``int``) Number of columns in the subplot

     * *index* (``int``) Index of the subplot

     * *reset* (``bool``) Whether or not to reset the figure.

Providing keywords other than these has undefined results, but may just work!
