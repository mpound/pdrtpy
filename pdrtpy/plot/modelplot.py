import warnings
from copy import deepcopy

import astropy.units as u
import astropy.wcs as wcs
import numpy as np
import numpy.ma as ma
from astropy import log
from matplotlib import ticker
from matplotlib.lines import Line2D

from .. import utils
from ..measurement import Measurement
from .plotbase import PlotBase

log.setLevel("WARNING")  # see issue 163


class ModelPlot(PlotBase):
    """Tool for exploring sets of models.

    Can plot individual intensity or ratio models, phase-space diagrams, and
    optionally overlay observations. Units are seamlessly transformed, so you
    can plot in Habing units, Draine units, or any conformable quantity.
    ModelPlot does not require model fitting with
    :class:`~pdrtpy.tool.lineratiofit.LineRatioFit` first.

    The methods of this class can take a variety of optional keywords. See the
    general `Plot Keywords`_ documentation.
    """

    def __init__(self, modelset, figure=None, axis=None):
        """
        Parameters
        ----------
        modelset : :class:`~pdrtpy.modelset.ModelSet`
            The set of models to use in these plots.
        """
        super().__init__(tool=None)
        self._modelset = modelset
        self._figure = figure
        self._axis = axis

    def plot(self, identifier, **kwargs):
        """Plot a model intensity or ratio.

        Parameters
        ----------
        identifier : str
            Identifier tag for the model to plot, e.g., ``"CII_158"``, ``"OI_145"``, ``"CO_43/CO_21"``.

        See Also
        --------
        :meth:`~pdrtpy.modelset.ModelSet.supported_lines` : for a list of available identifier tags.
        """
        kwargs_opts = {"measurements": None}
        kwargs_opts.update(kwargs)
        if "/" in identifier:
            self.ratio(identifier, **kwargs_opts)
        else:
            self.intensity(identifier, **kwargs_opts)

    def ratio(self, identifier, **kwargs):
        """Plot a model ratio.

        Parameters
        ----------
        identifier : str
            Identifier tag for the model to plot, e.g., ``"OI_63+CII_158/FIR"``, ``"CO_43/CO_21"``.

        See Also
        --------
        :meth:`~pdrtpy.modelset.ModelSet.supported_ratios` : for a list of available identifier tags.
        """
        ms = self._modelset
        model = ms.get_model(identifier)
        kwargs_opts = {
            "title": ms.table.loc[identifier]["title"],
            "colorbar": True,
            "contours": True,
            "colors": ["white"],
            "meas_color": [self._CB_color_cycle[0]],
            "legend": True,
            "bbox_to_anchor": None,
            "loc": "upper center",
            "image": True,
            "measurements": None,
        }
        kwargs_opts.update(kwargs)

        # make a sensible choice about contours if image is not shown
        if not kwargs_opts["image"] and kwargs_opts["colors"][0] == "white":
            kwargs_opts["colors"][0] = "black"

        self._plot_no_wcs(model, **kwargs_opts)
        if kwargs_opts["legend"]:
            lines = list()
            labels = list()
            if kwargs_opts["contours"]:
                lines.append(Line2D([0], [0], color=kwargs_opts["colors"][0], linewidth=3, linestyle="-"))
                labels.append(f"{self._modelset.name} model")
            if kwargs_opts["measurements"] is not None:
                lencol = len(kwargs_opts["meas_color"])
                if lencol > 1:
                    for i in range(lencol):
                        lines.append(Line2D([0], [0], color=kwargs_opts["meas_color"][i], linewidth=3, linestyle="-"))
                        labels.append(f"observed #{i}")
                else:
                    lines.append(Line2D([0], [0], color=kwargs_opts["meas_color"][0], linewidth=3, linestyle="-"))
            self._axis[0].legend(
                lines,
                labels,
                loc=kwargs_opts["loc"],
                bbox_to_anchor=kwargs_opts["bbox_to_anchor"],
                title=kwargs_opts["title"],
            )

    def intensity(self, identifier, **kwargs):
        """Plot a model intensity.

        Parameters
        ----------
        identifier : str
            Identifier tag for the model to plot, e.g., ``"OI_63"``, ``"CII_158"``, ``"CO_10"``.

        See Also
        --------
        :meth:`~pdrtpy.modelset.ModelSet.supported_intensities` : for a list of available identifier tags.
        """
        ms = self._modelset
        meas = kwargs.get("measurements", None)
        model = ms.get_models([identifier], model_type="intensity")
        if meas is not None:
            if type(meas[0]) is not Measurement:
                raise TypeError("measurement keyword value must be a list of Measurements")
            if model[identifier]._unit != meas[0].unit:
                raise TypeError(
                    f"Model and Measurement for {identifier} have different units:"
                    f" ({model[identifier]._unit},{meas[0].unit})"
                )
            if identifier != meas[0].id:
                msg = (
                    f"Identifiers of model {identifier} and supplied Measurement {meas[0].id} do not match. Plotting"
                    " anyway."
                )
                warnings.warn(msg, stacklevel=2)

        kwargs_opts = {
            "title": ms.table.loc[identifier]["title"],
            "colorbar": True,
            "contours": True,
            "colors": ["white"],
            "meas_color": [self._CB_color_cycle[0]],
            "legend": True,
            "bbox_to_anchor": None,
            "loc": "upper center",
            "image": True,
        }

        kwargs_opts.update(kwargs)
        if not kwargs_opts["image"] and kwargs_opts["colors"][0] == "white":
            kwargs_opts["colors"][0] = "black"
        self._plot_no_wcs(model[identifier], **kwargs_opts)
        if kwargs_opts["legend"]:
            lines = list()
            labels = list()
            if kwargs_opts["contours"]:
                lines.append(Line2D([0], [0], color=kwargs_opts["colors"][0], linewidth=3, linestyle="-"))
                labels.append("model")
            if meas is not None:
                lines.append(Line2D([0], [0], color=kwargs_opts["meas_color"][0], linewidth=3, linestyle="-"))
                labels.append("observed")
            # maybe loc should be 'best' but then it bounces around
            self._axis[0].legend(
                lines,
                labels,
                loc=kwargs_opts["loc"],
                bbox_to_anchor=kwargs_opts["bbox_to_anchor"],
                title=kwargs_opts["title"],
            )

    def overlay(self, measurements, **kwargs):
        r"""Overlay one or more single-pixel measurements in the model space :math:`(n,F_{FUV})`.

        Parameters
        ----------
        measurements : list of :class:`~pdrtpy.measurement.Measurement`
            A list of one or more Measurements to overlay.
        shading : float, optional
            Controls how measurements and errors are drawn. If 0, Measurements
            will be drawn as solid contours for the value and dashed for the
            +/- errors. If between 0 and 1, Measurements are drawn as filled
            contours representing the size of the errors (see
            :func:`matplotlib.pyplot.contourf`) with alpha set to the
            ``shading`` value. Default: 0.4.
        """

        kwargs_opts = {
            "units": None,
            "image": False,
            "contours": False,
            "meas_color": self._CB_color_cycle,
            "levels": None,
            "label": False,
            "linewidths": 1.0,
            "ncols": 1,
            "norm": None,
            "title": None,
            "reset": True,
            "legend": True,
            "bbox_to_anchor": None,
            "loc": "upper center",
            "shading": 0.4,
        }

        kwargs_opts.update(kwargs)
        if kwargs_opts["shading"] < 0 or kwargs_opts["shading"] > 1:
            raise ValueError("Shading must be between 0 and 1 inclusive")
        ids = [m.id for m in measurements]
        meas = dict(zip(ids, measurements, strict=False))
        models = [self._modelset.get_model(i) for i in ids]
        # need to trim model grids if H2 is present
        if utils._has_H2(ids) and self._modelset.is_wk2006:
            warnings.warn("Trimming all model grids to match H2 grid: log(n) = 1-5, log(G0) = 1-5", stacklevel=2)
            utils._trim_all_to_H2(models)
        i = 0
        nratio = 0
        nintensity = 0
        for val in models:
            if np.size(meas[val.id].data) != 1:
                raise ValueError(
                    f"Can't plot {val.id}. This method only works with single pixel Measurements [len(measurement.data)"
                    " must be 1]"
                )
            if i > 0:
                kwargs_opts["reset"] = False
            # pass the index of the contour color to use via the "secret" colorcounter keyword.
            self._plot_no_wcs(
                val, header=None, measurements=[utils.convert_if_necessary(meas[val.id])], colorcounter=i, **kwargs_opts
            )
            if val.modeltype == "ratio":
                nratio = nratio + 1
            if val.modeltype == "intensity":
                nintensity = nintensity + 1
            i = i + 1
        if kwargs_opts["legend"]:
            if nratio == 0 and nintensity > 0:
                word = "Intensities"
            elif nratio > 0 and nintensity == 0:
                word = "Ratios"
            else:
                word = "Values"
            lines = [Line2D([0], [0], color=c, linewidth=3, linestyle="-") for c in kwargs_opts["meas_color"][0:i]]
            labels = [k.title for k in models]
            self._plt.legend(
                lines,
                labels,
                loc=kwargs_opts["loc"],
                bbox_to_anchor=kwargs_opts["bbox_to_anchor"],
                title="Observed " + word,
            )

    def isoplot(self, identifier, plotnaxis, nax_clip=None, **kwargs):
        """Plot lines of constant model parameter as a function of the other model parameter and a model intensity or ratio.

        Parameters
        ----------
        identifier : str
            Identifier tag for the model to plot, e.g., ``"OI_63/CO_21"`` or ``"CII_158"``.
        plotnaxis : int
            Which NAXIS to use to compute lines of constant value. Since models
            have two axes, this must be either 1 or 2.
        nax_clip : array-like of :class:`~astropy.units.Quantity`, optional
            The range of model parameters on NAXIS{plotnaxis} to show.
            e.g. ``[10, 1E7]*Unit("cm-3")``. Default: None (full range).
        step : int, optional
            Allows skipping lines of constant value, e.g. plot every ``step``-th
            value. Useful when parameter space is crowded. Default: 1.
        """
        kwargs_opts = {
            "errorbar": False,
            "fmt": None,
            "label": None,
            "legend": True,
            "title": None,
            "grid": True,
            "figsize": (8, 5),
            "linewidth": 2.0,
            "aspect": "auto",
            "step": 1,
            "logx": False,
            "logy": False,
            "measurements": None,
            "bbox_to_anchor": (1.024, 1),
            "loc": "upper left",
            "xaxis_unit": None,
            "yaxis_unit": None,
            "test": False,
        }
        kwargs_opts.update(kwargs)
        model = self._modelset.get_model(identifier)
        self._figure, self._axis = self._plt.subplots(nrows=1, ncols=1, figsize=kwargs_opts["figsize"])
        if plotnaxis != 1 and plotnaxis != 2:
            raise ValueError("plotnaxis must be 1 or 2")
        if plotnaxis == 1:
            otherindex = 1
        if plotnaxis == 2:
            otherindex = 0
        pindex = plotnaxis - 1
        naxis_is_log = ["log" in c for c in model.wcs.wcs.ctype]
        naxis = list(self._get_xy_from_wcs(model, quantity=True, linear=True))
        naxislog = self._get_xy_from_wcs(model, quantity=True, linear=False)
        if nax_clip is None:
            nax_clip = [naxis[pindex][0].value, naxis[pindex][-1].value] * naxis[pindex].unit
        dcc = nax_clip.to(naxis[pindex].unit)
        # Select the model naxis *indices* within the NAX limits
        xi = np.where((naxis[pindex] >= dcc[0]) & (naxis[pindex] <= dcc[1]))[0]
        if naxis_is_log[pindex]:
            # Create an array containing *indices* of the range of naxis[pindex] alues.
            # -5,12 is a large enough logarithmic range to contain any model we have
            # this picks out integer values of log(naxis).
            x2 = np.hstack([np.where((np.round(naxislog[pindex].value, 1)) == i)[0] for i in np.arange(-5, 12)])
        else:
            x2 = np.arange(model.wcs._naxis[pindex])

        xi2 = np.intersect1d(xi, x2)
        naxis[otherindex], xlabel = utils.rescale_axis_units(
            naxis[otherindex],
            naxis[otherindex].unit,
            model.wcs.wcs.ctype[otherindex],
            kwargs_opts["xaxis_unit"],
            loglabel=False,
        )
        # @TODO possibly support plotaxis_unit
        lines = []
        # convert the yaxis if requested. The yaxis is the pixel values in the model,
        # which have units as specified in BUNIT header item.
        bunit = 1.0 * u.Unit(model.header.get("BUNIT", ""))
        bscale = 1.0
        if kwargs_opts["yaxis_unit"] is not None:
            bunit = bunit.to(kwargs_opts["yaxis_unit"])
            bscale = bunit.value
        if bunit != u.dimensionless_unscaled:
            ylabel = rf"{model.title} [{bunit.unit:latex_inline}]"
        else:
            ylabel = model.title
        for j in xi2[:: kwargs_opts["step"]]:
            if plotnaxis == 1:
                yy = bscale * model[0 : len(naxis[otherindex]), j]
            else:
                yy = bscale * model[j, 0 : len(naxis[otherindex])]
            if naxis_is_log[pindex]:
                label = f"{np.round(np.log10(naxis[pindex][j].to(nax_clip.unit).value), 1):.1f}"
            else:
                label = f"{np.round(naxis[pindex][j].to(nax_clip.unit).value, 0):.0f}"
            lines.extend(self._axis.plot(naxis[otherindex], yy, label=label))
        # Format the plot in a sensible way
        self._axis.set_ylabel(ylabel)
        self._axis.set_xlabel(xlabel)
        self._axis.set_aspect(kwargs_opts["aspect"])
        self._axis.ticklabel_format(axis="both", useMathText=True)
        if kwargs_opts["logx"]:
            self._axis.set_xscale("log")
        if kwargs_opts["logy"]:
            self._axis.set_yscale("log")
        self._set_standard_ticks(self._axis)
        if kwargs_opts["grid"]:
            self._draw_grid(self._axis, kwargs_opts["linewidth"])
        if kwargs_opts["legend"]:
            title1 = model.wcs.wcs.ctype[pindex]
            if "_" in title1:
                title1 = r"${\rm " + title1 + "}$"
            unit1 = "[" + nax_clip.unit.to_string("latex_inline") + "]"
            handles, labels = self._axis.get_legend_handles_labels()
            phantom = self._make_phantom_handles(self._axis, 2)
            labels = [title1, unit1, *labels]
            handles = phantom + lines
            leg = self._axis.legend(
                handles,
                labels,
                ncol=1,
                markerfirst=True,
                bbox_to_anchor=kwargs_opts["bbox_to_anchor"],
                loc=kwargs_opts["loc"],
            )
            self._zero_legend_header_widths(leg)
        if kwargs_opts["title"] is not None:
            self._axis.set_title(kwargs_opts["title"])

    def phasespace(
        self,
        identifiers,
        nax1_clip=None,
        nax2_clip=None,
        reciprocal=None,
        **kwargs,
    ):
        r"""Plot lines of constant density and radiation field on a ratio-ratio, ratio-intensity, or intensity-intensity map.

        Parameters
        ----------
        identifiers : list of str
            List of two identifier tags for the model to plot, e.g.,
            ``["OI_63/CO_21", "CII_158"]``.
        nax1_clip : array-like of :class:`~astropy.units.Quantity`, optional
            Range of model densities on NAXIS1 to show. For most models, NAXIS1
            is hydrogen number density :math:`n_H` in :math:`{\rm cm}^{-3}`.
            For ionized gas models, it is electron temperature :math:`T_e` in K.
            Default: ``[10, 1E7]*Unit("cm-3")``.
        nax2_clip : array-like of :class:`~astropy.units.Quantity`, optional
            Range of model parameters on NAXIS2. For most models, NAXIS2 is
            radiation field intensities in Habing or cgs units. For ionized gas
            models, it is electron volume density :math:`n_e`.
            Default: ``[10, 1E6]*utils.habing_unit``.
        reciprocal : array-like of bool, optional
            Whether to plot the reciprocal of the model on each axis. e.g.
            ``[False, True]`` means don't flip the X axis but flip the Y axis.
            Default: ``[False, False]``.
        measurements : array-like of :class:`~pdrtpy.measurement.Measurement`, optional
            A list of two Measurements, one per identifier, to plot as data
            points on the grid. Pairs must be given as ``[m1x, m1y, m2x, m2y, ...]``.
            Default: None.
        errorbar : bool, optional
            Plot error bars when given measurements. Default: True.
        fmt : array of str, optional
            Plot format for each Measurement pair. See :meth:`matplotlib.axes.Axes.plot`.
            Default: ``'sk'`` for all points.
        label : array of str, optional
            Legend label(s) for each Measurement pair. Default: ``'data'`` for all.
        legend : bool, optional
            Draw a legend on the plot. Default: True.
        title : str, optional
            Title to draw on the plot. Default: None.
        linewidth : float, optional
            Line width.
        grid : bool, optional
            Show grid. Default: True.
        figsize : 2-tuple of float, optional
            Figure dimensions (width, height) in inches. Default: ``(8, 5)``.
        capsize : float, optional
            End cap length of errorbars in points. Default: 3.
        markersize : float, optional
            Size of data point marker in points. Default: 8.
        """
        kwargs_opts = {
            "errorbar": False,
            "fmt": None,
            "label": None,
            "legend": True,
            "measurements": None,
            "title": None,
            "grid": True,
            "figsize": (8, 5),
            "linewidth": 2.0,
            "capsize": 5.0,
            "markersize": 8.0,
            "aspect": "auto",
            "bbox_to_anchor": (1.024, 1),
            "loc": "upper left",
        }

        kwargs_opts.update(kwargs)
        if nax1_clip is None:
            nax1_clip = [10, 1e7] * u.Unit("cm-3")
        if nax2_clip is None:
            nax2_clip = [10, 1e6] * utils.habing_unit
        if reciprocal is None:
            reciprocal = [False, False]
        # various input checks
        if len(list(identifiers)) != 2:
            raise ValueError("Length of identifiers list must be exactly 2")
        models = [self._modelset.get_model(i) for i in identifiers]
        mids = [m.id for m in models]
        mdict = dict(zip(mids, models, strict=False))
        if kwargs_opts["measurements"] is not None:
            for m in kwargs_opts["measurements"]:
                if type(m) is not Measurement:
                    raise TypeError("measurement keyword value must be a list of Measurements")
                if m.id not in mids:
                    raise TypeError(f"Can't find measurement identifier {m.id} in model list {mids}.")
                if mdict[m.id]._unit != m.unit:
                    raise TypeError(
                        f"Model and Measurement for {m.id} have different units: ({mdict[m.id]._unit},{m.unit})"
                    )

        # Now we need to find the model points that fall within the user-specified NAXIS limits.
        # First get the x,y of the models
        xlog, ylog = self._get_xy_from_wcs(models[0], quantity=True, linear=False)
        xlin, ylin = self._get_xy_from_wcs(models[0], quantity=True, linear=True)
        x_is_log = "log" in models[0].wcs.wcs.ctype[0]
        y_is_log = "log" in models[0].wcs.wcs.ctype[1]
        # linear and log units are same so doesn't matter which is used for conversion
        dcc = nax1_clip.to(xlog.unit)
        rcc = nax2_clip.to(ylog.unit)
        # Select the model x,y *indices* within the NAX limits
        xi = np.where((xlin >= dcc[0]) & (xlin <= dcc[1]))[0]
        yi = np.where((ylin >= rcc[0]) & (ylin <= rcc[1]))[0]
        # Create an array containing *indices* of the range of x,y values.
        # -5,12 is a large enough logarithmic range to contain any model we have
        if x_is_log:
            x2 = np.hstack([np.where((np.round(xlog.value, 1)) == i)[0] for i in np.arange(-5, 12)])
        else:
            x2 = np.arange(len(xlin))
        # for 2020 models FUV is not an integral value in erg s-1 cm-2
        # so rounding is necessary.
        if y_is_log:
            y2 = np.hstack([np.where((np.round(ylog.value, 1)) == i)[0] for i in np.arange(-5, 12)])
        else:
            y2 = np.arange(len(ylin))
        # Intersection of these two arrays contain the indices of desired model plot points.
        xi2 = np.intersect1d(xi, x2)
        yi2 = np.intersect1d(yi, y2)
        self._figure, self._axis = self._plt.subplots(nrows=1, ncols=1, figsize=kwargs_opts["figsize"])
        linesN = []
        linesG = []
        # Sort out the axes labels depending on whether reciprocal=True or not.
        for j in xi2:
            if x_is_log:
                label = np.round(np.log10(xlin[j].to(nax1_clip.unit).value), 1)
            else:
                label = f"{np.round(xlin[j].to(nax1_clip.unit).value, 0):.0f}"
            if models[0].unit == "":
                m0label = models[0].title
            else:
                m0label = models[0].title + " [" + u.Unit(models[0].unit).to_string("latex_inline") + "]"
            if models[1].unit == "":
                m1label = models[1].title
            else:
                m1label = models[1].title + " [" + u.Unit(models[1].unit).to_string("latex_inline") + "]"
            if reciprocal[0]:
                xx = 1 / models[0][yi2[0] : yi2[-1] + 1, j]
                self._axis.set_xlabel(utils.fliplabel(m0label))
            else:
                xx = models[0][yi2[0] : yi2[-1] + 1, j]
                self._axis.set_xlabel(m0label)
            if reciprocal[1]:
                yy = 1 / models[1][yi2[0] : yi2[-1] + 1, j]
                self._axis.set_ylabel(utils.fliplabel(m1label))
            else:
                yy = models[1][yi2[0] : yi2[-1] + 1, j]
                self._axis.set_ylabel(m1label)
            linesN.extend(self._axis.loglog(xx, yy, label=label, lw=2))

        for j in yi2:
            if y_is_log:
                label = np.round(np.log10(ylin[j].to(nax2_clip.unit).value), 1)
            else:
                label = f"{np.round(ylin[j].to(nax2_clip.unit).value, 0):.0f}"
            if reciprocal[0]:
                xx = 1 / models[0][j, xi2[0] : xi2[-1] + 1]
            else:
                xx = models[0][j, xi2[0] : xi2[-1] + 1]
            if reciprocal[1]:
                yy = 1 / models[1][j, xi2[0] : xi2[-1] + 1]
            else:
                yy = models[1][j, xi2[0] : xi2[-1] + 1]
            linesG.extend(self._axis.loglog(xx, yy, label=label, lw=2, ls="--"))

        # plot the input measurement with error bar. Keywords are
        # [m1x,m1y,m2x,m2y,...]
        # [fmt1,fmt2,...]
        # [label1,label2,...]
        if kwargs_opts["measurements"] is not None:
            l_meas = len(kwargs_opts["measurements"])
            if l_meas % 2 != 0:
                msg = f"Number of Measurements must be even. You provided {l_meas}"
                raise ValueError(msg)
            n_meas = int(l_meas / 2)
            fmt = kwargs_opts["fmt"]
            # Set the default format to black squares.
            if fmt is None:
                fmt = np.full([n_meas], "sk")
            elif len(fmt) != n_meas:
                msg = f"Number of plot formats {len(fmt)} doesn't match number of Measurement pairs {n_meas}."
                raise ValueError(msg)
            label = kwargs_opts["label"]
            if label is None:
                label = ["data"]
            elif len(label) != n_meas:
                msg = f"Number of data labels {len(label)} doesn't match number of Measurement pairs {n_meas}."
                raise ValueError(msg)

            args = []
            i = 0
            # ensure measurements are assigned to correct axis, based on their identifiers.
            for k in range(0, l_meas, 2):
                kk = k + 1
                if kwargs_opts["measurements"][k].id == identifiers[0]:
                    _x = kwargs_opts["measurements"][k]
                    _y = kwargs_opts["measurements"][kk]
                else:
                    _x = kwargs_opts["measurements"][kk]
                    _y = kwargs_opts["measurements"][k]
                # collect the args
                args.extend([_x, _y, fmt[i]])
                # Plot the error bars.  Since, unlike axis.loglog(), axis.errorbar() can't
                # take a *args, we have to call this each time.
                # Note use of zorder to ensure points are on top of lines.
                if kwargs_opts["errorbar"]:
                    self._axis.errorbar(
                        x=_x.data,
                        y=_y.data,
                        xerr=_x.error,
                        yerr=_y.error,
                        capsize=kwargs_opts["capsize"],
                        fmt=fmt[i],
                        capthick=2,
                        ls=None,
                        zorder=6,
                        markersize=kwargs_opts["markersize"],
                    )
                i = i + 1
            # the data points
            dataline = self._axis.loglog(*args, zorder=5, markersize=kwargs_opts["markersize"])
            self._axis.set_aspect(kwargs_opts["aspect"])
            self._axis.set_xscale("log")
            self._axis.set_yscale("log")
            self._set_standard_ticks(self._axis)
        if kwargs_opts["grid"]:
            self._draw_grid(self._axis, kwargs_opts["linewidth"])
        if kwargs_opts["legend"]:
            # Manually build the legend. Create the column headers for the legend
            # and blank handles and labels to take up space for the headers and
            # when the number of naxis1 and naxis2 traces
            # are not equal.
            title1 = models[0].wcs.wcs.ctype[0]
            if "_" in title1:
                title1 = r"${\rm " + title1 + "}$"
            unit1 = "[" + nax1_clip.unit.to_string("latex_inline") + "]"
            rs = nax2_clip.unit.to_string()
            rsl = nax2_clip.unit.to_string("latex_inline")
            if "G0" in models[0].wcs.wcs.ctype[1] or "FUV" in models[0].wcs.wcs.ctype[1]:
                title2 = "log(" + utils.get_rad(rs) + ")"
            else:
                title2 = models[0].wcs.wcs.ctype[1]
                if "_" in title2:
                    title2 = r"${\rm " + title2 + "}$"
            unit2 = "[" + rsl + "]"

            handles, labels = self._axis.get_legend_handles_labels()
            phantom = self._make_phantom_handles(self._axis, 2)
            lN = len(linesN)
            lG = len(linesG)
            diff = lN - lG
            adiff = abs(diff)
            phantom2 = self._make_phantom_handles(self._axis, adiff)
            blank = [""] * adiff

            if diff == 0:
                labels.insert(lN, unit2)
                labels.insert(lN, title2)
                labels = [title1, unit1, *labels]
                linesN = phantom + linesN
                linesG = phantom + linesG
            elif diff > 0:  # more densities than radiation fields
                labels.insert(lN, unit2)
                labels.insert(lN, title2)
                labels = [title1, unit1, *labels, *blank]
                linesN = phantom + linesN
                linesG = phantom + linesG + phantom2
            elif diff < 0:  # more radiation fields than densities
                labels = labels[0:lN] + blank + labels[lN:]
                labels.insert(lN + adiff, unit2)
                labels.insert(lN + adiff, title2)
                labels = [title1, unit1, *labels]
                linesN = phantom + linesN + phantom2
                linesG = phantom + linesG
            handles = linesN + linesG

            # add a second legend if user supplied measurements
            if kwargs_opts["measurements"] is not None:
                dl = self._axis.legend(dataline, label, loc="best")
                self._axis.add_artist(dl)

            leg = self._axis.legend(
                handles,
                labels,
                ncol=2,
                markerfirst=True,
                bbox_to_anchor=kwargs_opts["bbox_to_anchor"],
                loc=kwargs_opts["loc"],
            )
            self._zero_legend_header_widths(leg)
        # Put the plot title on if given.
        if kwargs_opts["title"] is not None:
            self._axis.set_title(kwargs_opts["title"])

    def _get_xy_from_wcs(self, data, quantity=False, linear=False):
        """Get the x,y axis vectors from the WCS of the input data.

        Parameters
        ----------
        data : :class:`astropy.io.fits.ImageHDU`, :class:`astropy.nddata.CCDData`, or :class:`~pdrtpy.measurement.Measurement`
            The input image.
        quantity : bool, optional
            If True, return arrays as :class:`astropy.units.Quantity`. Default: False.
        linear : bool, optional
            If True, return arrays in linear space; if False, in log space. Default: False.

        Returns
        -------
        :class:`numpy.ndarray` or :class:`astropy.units.Quantity`
            The axis values as arrays. Values are center of pixel.
        """
        return utils.get_xy_from_wcs(data, quantity, linear)

    # @todo allow data to be an array? see overlay()
    def _plot_no_wcs(self, data, header=None, **kwargs):
        """generic plotting method for images with no WCS, used by other plot methods"""
        measurements = kwargs.pop("measurements", None)
        _dataheader = getattr(data, "header", None)
        if _dataheader is None and header is None:
            raise Exception(
                "Either your data must have a header dictionary or you must provide one via the header parameter"
            )
        # input header supercedes data header, under assumption user knows what they are doing.
        if header is not None:
            _header = deepcopy(header)
            if getattr(data, "wcs", None) is None:
                data.wcs = wcs.WCS(_dataheader)
        else:
            _header = deepcopy(_dataheader)
            # CRxxx might be in wcs and not in header
            if getattr(data, "wcs", None) is not None:
                _header.update(data.wcs.to_header())
            else:
                # needed for get_xy_from_wcs call later.
                data.wcs = wcs.WCS(_header)
                # however, get the usual complaint about non-FITS units
                # in WCS, so remove them here because they aren't needed.
                # We have to convolute ourselves later to add them back in!
                data.wcs.wcs.cunit = ["" for i in range(data.wcs.naxis)]

        kwargs_opts = {
            "units": None,
            "image": True,
            "colorbar": False,
            "contours": True,
            "label": False,
            "title": None,
            "xaxis_unit": None,
            "yaxis_unit": None,
            "logx": True,
            "xlim": None,
            "logy": True,
            "ylim": None,
            "legend": False,
            "meas_color": ["#4daf4a"],
            "shading": 0.4,
            "test": False,
        }

        kwargs_contour = {"levels": None, "colors": ["white"], "linewidths": 1.0}

        # Merge in any keys the user provided, overriding defaults.
        kwargs_contour.update(kwargs)
        kwargs_opts.update(kwargs)

        # if self._tool is not None:
        #     if self._tool._modelnaxis is None and "NAXIS" not in _header:
        #         raise Exception("Image header/WCS has no NAXIS keyword")

        if "NAXIS" not in _header:
            raise Exception("Image header/WCS has no NAXIS keyword")
        else:
            _naxis = _header["NAXIS"]

        if _naxis == 2:
            if kwargs_opts["units"] is not None:
                k = utils.to(kwargs_opts["units"], data)
            else:
                k = data
        elif _naxis == 3:
            if kwargs_opts["units"] is not None:
                k = utils.to(kwargs_opts["units"], data[0, :, :])
            else:
                k = data[0, :, :]
        else:
            raise Exception(f"Unexpected NAXIS value: {_naxis:d}")

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
            "constrained_layout": False,
            "projection": data.wcs,
        }

        # delay merge until min_ and max_ are known
        kwargs_imshow.update(kwargs)
        _norm = self._get_norm(
            kwargs_imshow["norm"], km, kwargs_imshow["vmin"], kwargs_imshow["vmax"], kwargs_imshow["stretch"]
        )

        kwargs_subplot.update(kwargs)
        # swap ncols and nrows in figsize to preserve aspect ratio
        kwargs_subplot["figsize"] = kwargs.get("figsize", (kwargs_subplot["ncols"] * 5, kwargs_subplot["nrows"] * 5))

        axidx = kwargs_subplot["index"] - 1
        if kwargs_subplot["reset"]:
            # @todo can probably consolidate this
            self._figure, self._axis = self._plt.subplots(
                kwargs_subplot["nrows"],
                kwargs_subplot["ncols"],
                figsize=kwargs_subplot["figsize"],
                subplot_kw={"aspect": kwargs_imshow["aspect"]},
                constrained_layout=kwargs_subplot["constrained_layout"],
            )

        # Make sure self._axis is an array because we will index it below.
        if type(self._axis) is not np.ndarray:
            self._axis = np.array([self._axis])

        # When using ncols>1, either the index needs to be 2-d
        # or the axis array needs to be 1-d.  This takes the second approach:
        if len(self._axis.shape) > 1:
            self._axis = self._axis.flatten()

        x, y = self._get_xy_from_wcs(data, quantity=True, linear=True)

        loglabel = not kwargs_opts["logx"]
        x, xlab = utils.rescale_axis_units(x, _header["CUNIT1"], _header["CTYPE1"], kwargs_opts["xaxis_unit"], loglabel)
        loglabel = not kwargs_opts["logy"]
        y, ylab = utils.rescale_axis_units(y, _header["CUNIT2"], _header["CTYPE2"], kwargs_opts["yaxis_unit"], loglabel)
        # Finish up axes details.
        locmaj = ticker.LogLocator(base=10.0, subs=(1.0,), numticks=10)
        locmin = ticker.LogLocator(base=10.0, subs="auto", numticks=10)
        locmaj2 = ticker.LogLocator(base=10.0, subs=(1.0,), numticks=10)
        locmin2 = ticker.LogLocator(base=10.0, subs="auto", numticks=10)
        if kwargs_opts["logx"]:
            # order matters here: set_xscale will set a default LogLocator with ticks every 2 decades, so we need to override that.
            self._axis[axidx].set_xscale("log")
            self._axis[axidx].xaxis.set_major_locator(locmaj)
            self._axis[axidx].xaxis.set_minor_locator(locmin)
            self._axis[axidx].xaxis.set_minor_formatter(ticker.NullFormatter())
        if kwargs_opts["logy"]:
            self._axis[axidx].set_yscale("log")
            self._axis[axidx].yaxis.set_major_locator(locmaj2)
            self._axis[axidx].yaxis.set_minor_locator(locmin2)
            self._axis[axidx].yaxis.set_minor_formatter(ticker.NullFormatter())
        self._axis[axidx].set_ylabel(ylab)
        self._axis[axidx].set_xlabel(xlab)
        if kwargs_opts["xlim"] is not None:
            xlim = kwargs_opts["xlim"]
            self._axis[axidx].set_xlim(left=xlim[0], right=xlim[1])
        if kwargs_opts["ylim"] is not None:
            ylim = kwargs_opts["ylim"]
            self._axis[axidx].set_ylim(bottom=ylim[0], top=ylim[1])

        if kwargs_opts["image"]:
            # pass shading = auto to avoid deprecation warning
            # see https://matplotlib.org/3.3.0/gallery/images_contours_and_fields/pcolormesh_grids.html
            im = self._axis[axidx].pcolormesh(
                x.value, y.value, km, cmap=kwargs_imshow["cmap"], norm=_norm, shading="auto"
            )
            if kwargs_opts["colorbar"]:
                cbar = self._figure.colorbar(im, ax=self._axis[axidx])
                _norm_val = kwargs_imshow["norm"]
                if not (isinstance(_norm_val, str) and _norm_val.lower() == "log"):
                    # avoid AttributeError: 'LogFormatterSciNotation' object has no attribute 'set_powerlimits'
                    cbar.formatter = ticker.ScalarFormatter(useMathText=True)
                    cbar.formatter.set_scientific(True)
                    cbar.formatter.set_powerlimits((0, 0))
                    cbar.update_ticks()
                if "BUNIT" in _header:
                    lstr = u.Unit(_header["BUNIT"]).to_string("latex_inline")
                    cbar.ax.set_ylabel(lstr, rotation=90)

        if kwargs_opts["contours"]:
            if kwargs_contour["levels"] is None:
                # Figure out some autolevels
                kwargs_contour["levels"] = self._autolevels(km, "log")

            # suppress warnings about unused keywords and potential error
            # about cmap not being None. Also norm being a string will cause an error
            # in matplotlib==3.3.1+
            # @todo need a better solution for this, it is not scalable.
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
                "yaxis_unit",
                "xaxis_unit",
                "norm",
                "legend",
                "figsize",
                "constrained_layout",
                "stretch",
            ]:
                kwargs_contour.pop(kx, None)

            warnings.simplefilter("ignore", category=UserWarning)
            contourset = self._axis[axidx].contour(x.value, y.value, km.data, **kwargs_contour)
            warnings.resetwarnings()

            if kwargs_opts["label"]:
                self._axis[axidx].clabel(contourset, contourset.levels, inline=True, fmt="%1.2e")

        if kwargs_opts["title"] is not None and not kwargs_opts["legend"]:
            self._axis[axidx].set_title(kwargs_opts["title"])

        if measurements is None:
            mlen = 0
        else:
            mlen = len(measurements)
        if len(kwargs_opts["meas_color"]) < mlen:
            raise ValueError(
                f"Number of measurement colors (meas_color keyword) must match number of measurements ({mlen})"
            )

        if measurements is not None:
            lstyles = ["--", "-", "--"]
            # for serial calls to plot_no_wcs in an outside method (i.e. lineratioplot.overlay_all_ratios),
            # we need to keep track of the index for the measurement color, otherwise we always
            # select color index 0, resulting in all measurements contours having the same color.
            # this is a kluge but it's all I can think of right now.
            if "colorcounter" in kwargs:
                jj = kwargs["colorcounter"]
            else:
                jj = 0
            for m in measurements:
                # for the case of colorcounter kluge len(m) will always be 1, so we don't
                # run into issues with incrementing of jj interfering with colorcounter.
                colors = [kwargs_opts["meas_color"][jj]] * mlen
                if kwargs_opts["shading"] != 0:
                    self._axis[axidx].contourf(
                        x.value, y.value, k.data, levels=m.levels, colors=colors, alpha=kwargs_opts["shading"]
                    )
                    # Add extra call to plot contour because savefig("file.pdf")
                    # gets zorder of shading vs. contour wrong and contour lines
                    # don't show up. Only a bug when output is pdf. Harumph.
                    # See https://github.com/mpound/pdrtpy/issues/23
                    self._axis[axidx].contour(
                        x.value, y.value, k.data, levels=[m.levels[1]], colors=colors, alpha=kwargs_opts["shading"]
                    )
                else:
                    self._axis[axidx].contour(
                        x.value, y.value, k.data, levels=m.levels, linestyles=lstyles, colors=colors
                    )
                jj = jj + 1
