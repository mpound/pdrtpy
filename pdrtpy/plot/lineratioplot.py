from copy import deepcopy

import astropy.units as u
import scipy.stats as stats
from astropy import log
from matplotlib.lines import Line2D

from .. import utils
from .modelplot import ModelPlot
from .plotbase import PlotBase

log.setLevel("WARNING")  # see issue 163


class LineRatioPlot(PlotBase):
    """LineRatioPlot plots the results of :class:`~pdrtpy.tools.lineratiofit.LineRatioFit`.  It can plot maps of fit results, observations with errors on top of models, chi-square and confidence intervals and more.


    :Keyword Arguments:

    The methods of this class can take a variety of optional keywords.  See the general `Plot Keywords`_ documentation

    """

    def __init__(self, tool):
        """Init method

        :param tool: The line ratio fitting tool that is to be plotted.
        :type tool: `~pdrtpy.tool.LineRatioFit`
        """

        super().__init__(tool)
        self._figure = None
        self._axis = None
        self._modelplot = ModelPlot(self._tool._modelset, self._figure, self._axis)
        self._ratiocolor = []

    def modelintensity(self, id, **kwargs):
        r"""Plot one of the model intensities

        :param id: the intensity identifier, such as `CO_32``.
        :type id: str
        :param \**kwargs: see class documentation above
        :raises KeyError: if is id not in existing model intensities
        """
        ms = self._tool.modelset
        if id not in ms.supported_intensities["intensity label"]:
            raise KeyError(f"{id} is not in the ModelSet of your LineRatioFit")

        model = ms.get_models([id], model_type="intensity")
        kwargs_opts = {"title": self._tool._modelset.table.loc[id]["title"], "colorbar": True}
        kwargs_opts.update(kwargs)
        self._modelplot._plot_no_wcs(model[id], **kwargs_opts)
        self._figure = self._modelplot.figure
        self._axis = self._modelplot.axis

    def modelratio(self, id, **kwargs):
        r"""Plot one of the model ratios

        :param id: the ratio identifier, such as ``CII_158/CO_32``.
        :type id: str
        :param \**kwargs: see class documentation above
        :raises KeyError: if is id not in existing model ratios

        """
        if self._tool._modelratios[id].shape == (1,):  # does this ever occur??
            return self._tool._modelratios[id]

        kwargs_opts = {
            "title": self._tool._modelset.table.loc[id]["title"],
            "units": u.dimensionless_unscaled,
            "colorbar": True,
        }
        kwargs_opts.update(kwargs)
        self._modelplot._plot_no_wcs(self._tool._modelratios[id], **kwargs_opts)
        self._figure = self._modelplot.figure
        self._axis = self._modelplot.axis

    def observedratio(self, id, **kwargs):
        """Plot one of the observed ratios

        :param id: the ratio identifier, such as ``CII_158/CO_32``.
        :type id: - str
        :raises KeyError: if id is not in existing observed ratios
        """
        if self._tool._observedratios[id].shape == (1, 0) or self._tool.has_vectors:
            return self._tool._observedratios[id]

        kwargs_opts = {
            "title": self._tool._modelset.table.loc[id]["title"],
            "units": u.dimensionless_unscaled,
            "colorbar": False,
        }
        kwargs_opts.update(kwargs)
        self._plot(data=self._tool._observedratios[id], **kwargs_opts)

    def density(self, **kwargs):
        """Plot the hydrogen nucleus volume density map that was computed by :class:`~pdrtpy.tool.lineratiofit.LineRatioFit` tool. Default units: cm :math:`^{-3}`"""
        kwargs_opts = {
            "units": "cm^-3",
            "aspect": "equal",
            "image": True,
            "contours": False,
            "label": False,
            "linewidths": 1.0,
            "levels": None,
            "norm": None,
            "title": None,
        }

        kwargs_opts.update(kwargs)

        # handle single pixel or multi-pixel non-map cases.
        if self._tool._density.shape == (1,) or self._tool.has_vectors:
            return utils.to(kwargs_opts["units"], self._tool._density)

        tunit = u.Unit(kwargs_opts["units"])
        if kwargs_opts["title"] is None:
            kwargs_opts["title"] = rf"n [{tunit:latex_inline}]"
        self._plot(self._tool._density, **kwargs_opts)

    def radiation_field(self, **kwargs):
        """Plot the radiation field map that was computed by :class:`~pdrtpy.tool.lineratiofit.LineRatioFit` tool. Default units: Habing."""

        kwargs_opts = {
            "units": "Habing",
            "aspect": "equal",
            "image": True,
            "contours": False,
            "label": False,
            "linewidths": 1.0,
            "levels": None,
            "norm": None,
            "title": None,
        }
        kwargs_opts.update(kwargs)

        # handle single pixel or multi-pixel non-map cases.
        if self._tool.radiation_field.shape == (1,) or self._tool.has_vectors:
            return utils.to(kwargs_opts["units"], self._tool.radiation_field)

        if kwargs_opts["title"] is None:
            rad_title = utils.get_rad(kwargs_opts["units"])
            tunit = u.Unit(kwargs_opts["units"])
            kwargs_opts["title"] = rf"{rad_title} [{tunit:latex_inline}]"

        self._plot(self._tool.radiation_field, **kwargs_opts)

    def _plot_chisq_impl(self, data_fn, title, min_val, label_sym, kwargs):
        """Shared implementation for chisq() and reduced_chisq().

        :param data_fn: bound tool method returning the chi-square data (tool.chisq or tool.reduced_chisq)
        :param title: axis/legend title string (e.g. r'$\\chi^2$ (dof=3)')
        :param min_val: scalar minimum chi-square value to show in the legend label
        :param label_sym: LaTeX symbol for the minimum, e.g. r'$\\chi_{min}^2$'
        :param kwargs: caller's **kwargs dict
        """
        kwargs_opts = {
            "units": None,
            "aspect": "equal",
            "image": True,
            "contours": True,
            "label": False,
            "colors": ["white"],
            "linewidths": 1.0,
            "norm": "simple",
            "stretch": "linear",
            "xaxis_unit": None,
            "yaxis_unit": None,
            "legend": None,
            "bbox_to_anchor": None,
            "loc": "upper center",
            "title": None,
        }
        kwargs_opts.update(kwargs)
        if not kwargs_opts["image"] and kwargs_opts["colors"][0] == "white":
            kwargs_opts["colors"][0] = "black"
        if self._tool.has_vectors:
            raise NotImplementedError("Plotting of chi-square is not yet implemented for vector Measurements.")
        if self._tool.has_maps:
            data = data_fn(min=True)
            if "title" not in kwargs:
                kwargs_opts["title"] = title
            self._plot(data, **kwargs_opts)
        else:
            data = data_fn(min=False)
            self._modelplot._plot_no_wcs(data, header=None, **kwargs_opts)
            if kwargs_opts["xaxis_unit"] is not None:
                x = utils.to(self._tool._density, kwargs_opts["xaxis_unit"]).value
            else:
                x = self._tool._density.value
            if kwargs_opts["yaxis_unit"] is not None:
                y = utils.to(kwargs_opts["yaxis_unit"], self._tool._radiation_field).value
            else:
                y = self._tool._radiation_field.value
            if kwargs_opts["title"] is None:
                kwargs_opts["title"] = title
            label = rf"{label_sym} = {min_val:.2g} @ (n,FUV) = ({x[0]:.2g},{y[0]:.2g})"
            self._modelplot._axis[0].scatter(x, y, c="r", marker="+", s=200, linewidth=2, label=label)
            if kwargs_opts["legend"]:
                self._modelplot._axis[0].legend(
                    title=kwargs_opts["title"], bbox_to_anchor=kwargs_opts["bbox_to_anchor"], loc=kwargs_opts["loc"]
                )
            self._figure = self._modelplot.figure
            self._axis = self._modelplot.axis

    def chisq(self, **kwargs):
        r"""Plot the :math:`\chi^2` map that was computed by the
        :class:`~pdrtpy.tool.lineratiofit.LineRatioFit` tool.
        """
        self._plot_chisq_impl(
            data_fn=self._tool.chisq,
            title=rf"$\chi^2$ (dof={self._tool._dof:d})",
            min_val=self._tool._chisq_min.value[0],
            label_sym=r"$\chi_{min}^2$",
            kwargs=kwargs,
        )

    def reduced_chisq(self, **kwargs):
        r"""Plot the reduced :math:`\chi^2` map that was computed by the
        :class:`~pdrtpy.tool.lineratiofit.LineRatioFit` tool.
        """
        self._plot_chisq_impl(
            data_fn=self._tool.reduced_chisq,
            title=rf"$\chi_\nu^2$ (dof={self._tool._dof:d})",
            min_val=self._tool._reduced_chisq_min.value[0],
            label_sym=r"$\chi_{\nu,min}^2$",
            kwargs=kwargs,
        )

    def show_both(self, units=None, **kwargs):
        """Plot both radiation field and volume density maps computed by the
        :class:`~pdrtpy.tool.lineratiofit.LineRatioFit` tool in a 1x2 panel subplot. Default units: ['Habing','cm^-3']
        """

        if units is None:
            units = ["Habing", "cm^-3"]
        _index = [1, 2]
        _reset = [True, False]

        kwargs_opts = {
            "image": True,
            "aspect": "equal",
            "contours": False,
            "label": False,
            "levels": None,
            "norm": None,
            "title": None,
            "nrows": 1,
            "ncols": 2,
            "index": _index[0],
            "reset": _reset[0],
        }

        kwargs_opts.update(kwargs)

        rf = self.radiation_field(units=units[0], **kwargs_opts)

        kwargs_opts["index"] = _index[1]
        kwargs_opts["reset"] = _reset[1]

        d = self.density(units=units[1], **kwargs_opts)
        # @todo don't return for plots.  print for non-plots
        return (rf, d)

    def confidence_intervals(self, **kwargs):
        r"""Plot the confidence intervals from the :math:`\chi^2` map computed by the
        :class:`~pdrtpy.tool.lineratiofit.LineRatioFit` tool. Default levels:  [50., 68., 80., 95., 99.]

        **Currently only works for single-pixel Measurements**
        """

        if self._tool.has_maps or self._tool.has_vectors:
            raise NotImplementedError("Plotting of confidence intervals is not yet implemented for maps or vectors.")

        kwargs_opts = {
            "units": None,
            "aspect": "auto",
            "image": False,
            "contours": True,
            "label": True,
            "levels": [50.0, 68.0, 80.0, 95.0, 99.0],
            "colors": ["black"],
            "linewidths": 1.0,
            "norm": "simple",
            "stretch": "linear",
            "xaxis_unit": None,
            "yaxis_unit": None,
            "title": "Confidence Intervals",
        }

        # ensure levels are in ascending order
        kwargs_opts["levels"] = sorted(kwargs_opts["levels"])
        kwargs_opts.update(kwargs)

        confidence = deepcopy(self._tool._chisq)
        confidence.data = 100 * stats.distributions.chi2.cdf(confidence.data, self._tool._dof)
        self._tool.confidence = confidence
        self._modelplot._plot_no_wcs(data=confidence, header=None, **kwargs_opts)
        self._figure = self._modelplot.figure
        self._axis = self._modelplot.axis

    def overlay_all_ratios(self, **kwargs):
        """Overlay all the measured ratios and their errors on the :math:`(n,F_{FUV})` space.

        This only works for single-valued Measurements; an overlay for multi-pixel doesn't make sense.
        """
        # NB: could have position and area though.

        if self._tool.has_maps or self._tool.has_vectors:
            raise NotImplementedError("Plotting of ratio overlays is not yet implemented for maps or vectors.")

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
        }

        kwargs_opts.update(kwargs)
        # force this as ncols !=1 makes no sense.
        kwargs_opts["ncols"] = 1

        i = 0
        _measurements = list()
        _models = list()
        # get_model will correctly raise exception if m.id not in ModelSet
        meas_passed = False
        if kwargs_opts.get("measurements", None) is not None:
            # avoid modifying a passed parameter
            _measurements = deepcopy(kwargs_opts["measurements"])
            meas_passed = True
            for m in _measurements:
                if i > 0:
                    kwargs_opts["reset"] = False
                val = self._tool.modelset.get_model(m.id)
                _models.append(val)
                kwargs_opts["measurements"] = [utils.convert_if_necessary(m)]
                self._modelplot._plot_no_wcs(_models[i], header=None, colorcounter=i, **kwargs_opts)
                i = i + 1
            kwargs_opts.pop("measurements")

        for key, val in self._tool._modelratios.items():
            if i > 0:
                kwargs_opts["reset"] = False
            self._modelplot._plot_no_wcs(
                val, header=None, colorcounter=i, **kwargs_opts, measurements=[self._tool._observedratios[key]]
            )
            i = i + 1
        if kwargs_opts["legend"]:
            lines = [Line2D([0], [0], color=c, linewidth=3, linestyle="-") for c in kwargs_opts["meas_color"][0:i]]
            labels = list()
            if meas_passed:
                for m in _measurements:
                    try:
                        tt = self._tool.modelset.get_model(m.id).title
                    except Exception:
                        tt = m.id
                    labels.append(tt)
                title = "Observed Ratios and Intensities"
            else:
                title = "Observed Ratios"
            labels.extend([self._tool._modelratios[k].title for k in self._tool._modelratios])
            _title = kwargs.get("title", None)
            if _title is not None:
                title += " " + _title
            self._plt.legend(
                lines, labels, loc=kwargs_opts["loc"], bbox_to_anchor=kwargs_opts["bbox_to_anchor"], title=title
            )
        self._figure = self._modelplot.figure
        self._axis = self._modelplot.axis

    def ratios_on_models(self, **kwargs):
        """Overlay all the measured ratios and their errors on the individual models for those ratios.  Plots are displayed in multi-column format, controlled the `ncols` keyword. Default: ncols=2

        **Currently only works for single-pixel Measurements**
        """

        if self._tool.has_maps or self._tool.has_vectors:
            raise NotImplementedError("Plotting of ratio overlays is not yet implemented for maps or vectors.")

        kwargs_opts = {
            "units": None,
            "image": True,
            "colorbar": True,
            "contours": True,
            "colors": ["white"],
            "levels": None,
            "label": False,
            "linewidths": 1.0,
            "meas_color": ["#4daf4a"],
            "ncols": 2,
            "norm": "simple",
            "stretch": "linear",
            "index": 1,
            "reset": True,
            "legend": True,
            "bbox_to_anchor": None,
            "loc": "upper center",
        }

        kwargs_opts.update(kwargs)

        kwargs_opts["ncols"] = min(kwargs_opts["ncols"], self._tool.ratiocount)
        kwargs_opts["nrows"] = int(round(self._tool.ratiocount / kwargs_opts["ncols"] + 0.49, 0))
        # defend against meas_color not being a list
        if isinstance(kwargs_opts["meas_color"], str):
            kwargs_opts["meas_color"] = [kwargs_opts["meas_color"]]

        for key, val in self._tool._modelratios.items():
            axidx = kwargs_opts["index"] - 1
            if kwargs_opts["index"] > 1:
                kwargs_opts["reset"] = False
            kwargs_opts["measurements"] = [self._tool._observedratios[key]]
            self._modelplot._plot_no_wcs(val, header=None, **kwargs_opts)
            self._axis = self._modelplot._axis
            self._figure = self._modelplot._figure
            kwargs_opts["index"] = kwargs_opts["index"] + 1
            if kwargs_opts["legend"]:
                if "title" not in kwargs:  # then it was None, and we customize it
                    _title = self._tool._modelratios[key].title
                else:
                    _title = kwargs["title"]
                lines = list()
                labels = list()
                if kwargs_opts["contours"]:
                    lines.append(Line2D([0], [0], color=kwargs_opts["colors"][0], linewidth=3, linestyle="-"))
                    labels.append("model")
                lines.append(Line2D([0], [0], color=kwargs_opts["meas_color"][0], linewidth=3, linestyle="-"))
                labels.append("observed")
                # maybe loc should be 'best' but then it bounces around
                self._axis[axidx].legend(
                    lines, labels, bbox_to_anchor=kwargs_opts["bbox_to_anchor"], loc=kwargs_opts["loc"], title=_title
                )

            # Turn off subplots greater than the number of
            # available ratios
            for i in range(self._tool.ratiocount, len(self._axis)):
                self._axis[i].axis("off")
