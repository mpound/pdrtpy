import numpy as np

from matplotlib.ticker import MultipleLocator

# from cycler import cycler

import astropy.units as u
from astropy.coordinates import SkyCoord
from astropy.nddata import StdDevUncertainty

from ..measurement import Measurement
from .plotbase import PlotBase
from ..pdrutils import float_formatter, LOGE


class ExcitationPlot(PlotBase):
    """
    ExcitationPlot creates excitation diagrams using the results of :class:`~pdrtpy.tool.h2excitationfit.H2ExcitationFit`. It can plot the observed excitation diagram with or without fit results, and allows averaging over user-given spatial areas.
    """

    def __init__(self, tool, label):
        super().__init__(tool)
        self._xlim = []
        self._ylim = []
        self._label = label
        self._logfile = None

    def _sorted_by_vibrational_level(self, measurements):
        # d is a dict of measurements
        ret = dict()
        for m in measurements:
            _key = f'v={self._tool._ac.loc[m]["vu"]}'
            if _key not in ret:
                ret[_key] = []
            ret[_key].append(measurements[m])
        return ret

    def ex_diagram(self, position=None, size=None, norm=True, show_fit=False, **kwargs):
        # @todo position and size might not necessarily match how the fit was done.
        #:type position: tuple or :class:`astropy.coordinates.SkyCoord`
        # or a :class:`~astropy.coordinates.SkyCoord`, which will use the :class:`~astropy.wcs.WCS` of the ::class:`~pdrtpy.measurement.Measurement`s added to this tool.
        r"""Plot the excitation diagram.  For maps of excitation parameters, a position and optional size are required.  To examine the entire map, use :meth:`explore`.

        :param position: The spatial position of excitation diagram.  For spatial averaging this is the cutout array's center with respect to the data array. The position may be specified as an `(x, y)` tuple of pixel coordinates or a SkyCoord coordinate
        :type position: tuple or :class:`~astropy.coordinates.SkyCoord`
        :param size: The size of the cutout array along each axis. If size is a scalar number or a scalar :class:`~astropy.units.Quantity`, then a square cutout of size will be created. If `size` has two elements, they should be in `(ny, nx)` order. Scalar numbers in size are assumed to be in units of pixels. `size` can also be a :class:`~astropy.units.Quantity` object or contain :class:`~astropy.units.Quantity` objects. Such :class:`~astropy.units.Quantity` objects must be in pixel or angular units. For all cases, size will be converted to an integer number of pixels, rounding the the nearest integer.  See :class:`~astropy.nddata.utils.Cutout2D`
        :type size: int, array_like, or :class:`astropy.units.Quantity`
        :param norm: if True, normalize the column densities by the
                       statistical weight of the upper state, :math:`g_u`.
        :type norm: bool
        :param show_fit: Show the most recent fit done the the associated H2ExcitationFit tool.
        :type show_fit: bool
        """
        kwargs_opts = {
            "xmin": 0.0,
            "xmax": None,  #  we use np.max() later if user does not specify
            "ymax": 22,
            "ymin": 15,
            "xlabel": None,
            "ylabel": None,
            "grid": False,
            "figsize": (10, 7),
            "capsize": 3,
            "linewidth": 2.0,
            "markersize": 8,
            "color": None,
            "axis": None,
            "label": None,
            "aspect": "auto",
            "bbox_to_anchor": None,
            "loc": "best",
            "test": False,
        }
        kwargs_opts.update(kwargs)

        print(f"norm={norm} pos={position} size={size}")
        if isinstance(position, SkyCoord):
            position = self._tool.fitresult.get_pixel_from_coord(position)
            print(f"AFTER norm={norm} pos={position} size={size}")
        cdavg = self._tool.average_column_density(
            norm=norm, position=position, size=size, line=True
        )
        # print("CDAVG ",cdavg)
        energies = self._tool.energies(line=True)
        energy = np.array(list(energies.values()))
        colden = np.squeeze(np.array([c.data for c in cdavg.values()]))
        error = np.squeeze(np.array([c.error for c in cdavg.values()]))
        # print("ERROR ",error)
        # print("COLDEN",colden)
        sigma = LOGE * error / colden
        if kwargs_opts["axis"] is None:
            self._figure, self._axis = self._plt.subplots(
                figsize=kwargs_opts["figsize"]
            )
            _axis = self._axis
        else:
            _axis = kwargs_opts["axis"]
        if kwargs_opts["label"] != "v":
            if self._tool.opr_fitted and show_fit:
                _label = "LTE"
            else:
                _label = "$" + self._label + "$ data"
            ec = _axis.errorbar(
                energy,
                np.log10(colden),
                yerr=sigma,
                fmt="o",
                capsize=kwargs_opts["capsize"],
                label=_label,
                lw=kwargs_opts["linewidth"],
                ms=kwargs_opts["markersize"],
                color=kwargs_opts["color"],
            )
        else:
            # return dict of arrays of measuremtents with keys v=0,v=1,v=2 etc
            cdsort = self._sorted_by_vibrational_level(cdavg)
            ensort = self._sorted_by_vibrational_level(energies)
            # print("ENSORT" ,ensort.values())
            # cyc = cycler('color',  self._CB_color_cycle)
            # cyfill = cycler('fillstyle',['full', 'none', 'full', 'none', 'full', 'none', 'full', 'none', 'full'])
            # self._plt.rc('axes', prop_cycle=(cyc+cyfill))

            fmtd = {False: "o", True: "^"}  # there is no cycler for fmt, do it manually
            fmtb = False
            for key in cdsort:
                cs = np.squeeze(np.array([m.value[0] for m in cdsort[key]]))
                es = np.squeeze(np.array([m.error[0] for m in cdsort[key]]))
                ens = np.array([c for c in ensort[key]])
                # print(f"LOG10(CD({key}))={np.log10(cs)}")
                # print(f"E{key} = {ens}")
                sigma = LOGE * es / cs
                ec = _axis.errorbar(
                    ens,
                    np.log10(cs),
                    yerr=sigma,
                    fmt=fmtd[fmtb],
                    capsize=kwargs_opts["capsize"],
                    label=key,
                    lw=kwargs_opts["linewidth"],
                    ms=kwargs_opts["markersize"],
                )
                # fmtb = not fmtb
        tt = self._tool
        if self._tool.opr_fitted and show_fit:
            if position is not None and len(np.shape(tt.opr)) > 1:
                opr_v = tt.opr[position]
                opr_e = tt.opr.error[position]
                # a Measurement.get_as_measurement() would be nice
                opr_p = Measurement(
                    opr_v, uncertainty=StdDevUncertainty(opr_e), unit=""
                )
            else:
                opr_p = tt.opr
            cddn = colden * self._tool._canonical_opr / opr_p
            # Plot only the odd-J ones scaled by fitted OPR
            odd_index = np.where([self._tool._is_ortho(c) for c in cdavg.keys()])
            # color = ec.lines[0].get_color() # want these to be same color as data
            _axis.errorbar(
                x=energy[odd_index],
                y=np.log10(cddn[odd_index]),
                marker="^",
                label=f"OPR = {opr_p:.2f}",
                yerr=sigma[odd_index],
                capsize=2 * kwargs_opts["capsize"],
                linestyle="none",
                color="k",
                lw=kwargs_opts["linewidth"],
                ms=kwargs_opts["markersize"],
            )
        if kwargs_opts["xlabel"] is None:
            _axis.set_xlabel("$E_u/k$ (K)")
        else:
            _axis.set_xlabel(kwargs_opts["xlabel"])
        if kwargs_opts["ylabel"] is None:
            if norm:
                _axis.set_ylabel("log $(N_u/g_u) ~({\\rm cm}^{-2})$")
            else:
                _axis.set_ylabel("log $(N_u) ~({\\rm cm}^{-2})$")
        else:
            _axis.set_ylabel(kwargs_opts["ylabel"])
        first = True
        if kwargs_opts["label"] == "id":
            for lab in sorted(cdavg):
                _axis.text(x=energies[lab] + 100, y=np.log10(cdavg[lab]), s=str(lab))
        elif kwargs_opts["label"] == "j":  # label the points with e.g. J=2,3,4...
            for lab in sorted(cdavg):
                # this fails because the lowest J may not be the first data point.
                # we'd have to sort on Ju of the data. Which isn't even unique
                # if there are multiple vibrational levels.
                # if first:
                #    ss="J="+str(self._tool._ac.loc[lab]["Ju"])
                #    first=False
                # else:
                ss = str(self._tool._ac.loc[lab]["Ju"])
                _axis.text(x=energies[lab] + 100, y=np.log10(cdavg[lab]), s=ss)
        handles, labels = _axis.get_legend_handles_labels()
        print(f"show_fit is {show_fit}")
        if show_fit:
            if tt.fit_result is None:
                raise ValueError(
                    "No fit to show. Have you run the fit in your H2ExcitationFit?"
                )
            if np.shape(tt.fit_result.data) == (1,):
                position = 0
            elif position is None:
                raise ValueError("position must be provided for map fit results")
            if tt.fit_result[position] is None:
                raise ValueError(
                    f"The Excitation Tool was unable to fit pixel {position}. Try show_fit=False"
                )
            x_fit = np.linspace(0, max(energy), 30)
            # @TODO This now depends on tool._numcomponents
            outpar = tt.fit_result[position].params.valuesdict()
            if tt.numcomponents == 2:
                labcold = (
                    r"$T_{cold}=$"
                    + f"{tt.tcold[position]:3.0f}"
                    + r"$\pm$"
                    + f"{tt.tcold.error[position]:.1f} {tt.tcold.unit}"
                )
                # labcold = r"$T_{cold}=$" + f"{tt.tcold[position]:3.1f}"
                # labhot= r"$T_{hot}=$" + f"{tt.thot.value:3.0f}"+ r"$\pm$" + f"{tt.thot.error:.1f} {tt.thot.unit}"
                # labhot= r"$T_{hot}=$" + f"{tt.thot[position]:3.1f}"
                labhot = (
                    r"$T_{hot}=$"
                    + f"{tt.thot[position]:3.0f}"
                    + r"$\pm$"
                    + f"{tt.thot.error[position]:.1f} {tt.thot.unit}"
                )
            elif tt.numcomponents == 1:
                labcold = (
                    r"$T=$"
                    + f"{tt.tcold[position]:3.0f}"
                    + r"$\pm$"
                    + f"{tt.tcold.error[position]:.1f} {tt.tcold.unit}"
                )
            if position == 0:
                labnh = (
                    r"$N("
                    + self._label
                    + ")="
                    + float_formatter(tt.total_colden, 2)
                    + "$"
                )
            else:
                labnh = (
                    r"$N("
                    + self._label
                    + ")="
                    + float_formatter(
                        u.Quantity(tt.total_colden[position], tt.total_colden.unit), 2
                    )
                    + "$"
                )
            _axis.plot(
                x_fit,
                tt._one_line(x_fit, outpar["m1"], outpar["n1"]),
                ".",
                label=labcold,
                lw=kwargs_opts["linewidth"],
            )
            if tt.numcomponents == 2:
                _axis.plot(
                    x_fit,
                    tt._one_line(x_fit, outpar["m2"], outpar["n2"]),
                    ".",
                    label=labhot,
                    lw=kwargs_opts["linewidth"],
                )
            _axis.plot(
                x_fit, tt.fit_result[position].eval(x=x_fit, fit_opr=False), label="fit"
            )
            handles, labels = _axis.get_legend_handles_labels()
            # kluge to ensure N(H2) label is last
            phantom = _axis.plot([], marker="", markersize=0, ls="", lw=0)
            handles.append(phantom[0])
            labels.append(labnh)
        # Scale xaxis with max(energy). Round up to nearest 1000
        if kwargs_opts["xmax"] is None:
            kwargs_opts["xmax"] = np.round(500.0 + energy.max(), -3)
        _axis.set_xlim(kwargs_opts["xmin"], kwargs_opts["xmax"])
        print(f"setting ylim [{kwargs_opts['ymin']},{kwargs_opts['ymax']}]")
        _axis.set_ylim(kwargs_opts["ymin"], kwargs_opts["ymax"])
        # try to make reasonably-spaced xaxis tickmarks.
        # if I were clever, I'd do this with a function
        temperature_range = kwargs_opts["xmax"] - kwargs_opts["xmin"]
        if temperature_range <= 10000:
            _axis.xaxis.set_major_locator(MultipleLocator(1000))
            _axis.xaxis.set_minor_locator(MultipleLocator(200))
        elif temperature_range <= 26000:
            _axis.xaxis.set_major_locator(MultipleLocator(2000))
            _axis.xaxis.set_minor_locator(MultipleLocator(500))
        else:
            _axis.xaxis.set_major_locator(MultipleLocator(6000))
            _axis.xaxis.set_minor_locator(MultipleLocator(2000))
        _axis.yaxis.set_major_locator(MultipleLocator(1))
        _axis.yaxis.set_minor_locator(MultipleLocator(0.2))
        _axis.tick_params(axis="both", direction="in", which="both")
        _axis.tick_params(
            axis="both", bottom=True, top=True, left=True, right=True, which="both"
        )
        if kwargs_opts["grid"]:
            _axis.grid(
                visible=True,
                which="major",
                axis="both",
                lw=kwargs_opts["linewidth"] / 2,
                color="k",
                alpha=0.33,
            )
            _axis.grid(
                visible=True,
                which="minor",
                axis="both",
                lw=kwargs_opts["linewidth"] / 2,
                color="k",
                alpha=0.22,
                linestyle="--",
            )

        _axis.legend(
            handles,
            labels,
            bbox_to_anchor=kwargs_opts["bbox_to_anchor"],
            loc=kwargs_opts["loc"],
        )

    def temperature(self, component, **kwargs):
        """Plot the temperature of hot or cold gas component.

        :param component: 'hot' or 'cold'
        :type component: str
        """
        if component not in self._tool.temperature:
            raise KeyError(
                f"{component} not a valid component. Must be one of {list(self._tool.temperature.keys())}"
            )
        self._plot(self._tool.temperature[component], **kwargs)

    def column_density(self, component, log=True, **kwargs):
        """Plot the column density of hot or cold gas component, or total column density.

        :param component: 'hot', 'cold', or 'total
        :type component: str
        :param log: take the log10 of the column density before plotting
        """
        self._plot(self._tool.colden(component), log=log, **kwargs)

    def opr(self, **kwargs):
        """Plot the ortho-to-para ratio.  This will be a map if the input data are a map, otherwise a float value is returned."""
        if isinstance(self._tool.opr, float):
            return self._tool.opr
        self._plot(self._tool.opr, **kwargs)

    def explore(self, data=None, interaction_type="click", **kwargs):
        """Explore the fitted parameters of a map. A user-requested map is displayed in the left panel and in the right panel is the fitted excitation diagram for a point selected by the user.  The user clicks on a point in the left panel and the right panel will update with the excitation diagram for that point.

        :param data: A reference image to use for the left panel, e.g. the total column density, the cold temperature, etc.  This should be a reference results in the :class:`~pdrtpy.tool.h2excitation.H2Excitation` tool used for this :class:`~pdrtpy.plot.excitationplot.ExcitationPlot` (e.g., *htool.temperature['cold']*)
        :type data: :class:`~pdrtpy.measurement.Measurement`
        :param interaction_type: whether to use mouse click or mouse move to update the right hand panel.   Valid values are 'click' or 'move'.
        :type interaction_type: str
        :param \*\*kwargs: Other parameters passed to :meth:`~pdrtpy.plot.excitationplot.ExcitationPlot._plot`, :meth:`~pdrtpy.plot.excitationplot.ExcitationPlot.ex_diagram`, or matplotlib methods.

            - *units,image, contours, label, title, norm, figsize* -- See the general `Plot Keywords`_ documentation
            - *show_fit* - show the fit in the excitation diagram, Default: True
            - *log* - plot the log10 of the image, can be useful for column density,  Default: False
            - *markersize* - size of the marker displayed where clicked, in points, Default: 20
            - *fmt* - matplotlib format for the marker, Default:. 'r+'
        """

        kwargs_opts = {
            "units": None,
            "image": True,
            "colorbar": True,
            "contours": False,
            "label": False,
            "title": None,
            "norm": "simple",
            "log": False,
            "show_fit": True,
            "figsize": (5, 3),
            "markersize": 20,
            "fmt": "r+",
            "debug": False
        }
        # starting position is middle pixel of image. note // for integer arithmetic
        kwargs_opts.update(kwargs)
        debug = kwargs_opts.pop("debug")
        if debug:
            self._logfile = open("/tmp/test.log", "a")
        position = tuple(np.array(np.shape(data)) // 2)
        # print(position)
        # print(f"fit result at {position} is {self._tool.fit_result[position]}")
        # print(f"NONE? {self._tool.fit_result[position] is None}")
        if self._tool.fit_result[position] is None:
            # find another position where the fit succeeded
            ok = np.where(self._tool.fit_result._data is not None)
            #position = (ok[0][0], ok[1][0])
            position = (ok[1][0], ok[0][0])
        if debug: 
            self._logfile.write(f"Trying to get world coordinates at position {position}\n")
        coord = self._tool.fit_result.get_skycoord(position[0], position[1])
        if debug:
            self._logfile.write(f"Explore using position: {position} world {coord.to_string('hmsdms')} size=1\n")

        self._figure = self._plt.figure(figsize=kwargs_opts["figsize"], clear=True)
        self._axis = np.empty([2], dtype=object)
        self._axis[0] = self._figure.add_subplot(
            121, projection=data.wcs, aspect="auto"
        )
        self._axis[1] = self._figure.add_subplot(122, projection=None, aspect="auto")
        self._axis[1].tick_params(
            "y", labelright=True, labelleft=False
        )  # avoid overlap with colorbar
        self._axis[1].get_yaxis().set_label_position("right")
        fmt = kwargs_opts.pop("fmt", "r+")
        show_fit = kwargs_opts.pop("show_fit")
        self._plot(data, axis=self._axis, index=1, **kwargs_opts)
        self.ex_diagram(
            axis=self._axis[1],
            reset=False,
            position=position,
            size=1,
            norm=True,
            show_fit=show_fit,
            ymin=0,
            ymax=30,
        )

        self._marker = self.axis[0].plot(
            position[0], position[1], fmt, markersize=kwargs_opts["markersize"]
        )

        def update_lines(event):
            self._logfile = None
            try:
                if debug:
                    self._logfile = open("/tmp/test.log", "a")
                if debug:
                    self._logfile.write(
                    f"event.inaxes = {event.inaxes} x,y={event.xdata,event.ydata}\n"
                    )
                    self._logfile.write(f"event dict: {event.__dict__}")
   
                if (
                    event.inaxes == self._axis[0]
                ):  # the click must be on the left panel (map)
                    position = (int(round(event.ydata)), int(round(event.xdata)))
                    self._marker[0].set_marker("None")
                    self._marker = self.axis[0].plot(
                        position[1],
                        position[0],
                        fmt,
                        markersize=kwargs_opts["markersize"],
                    )
                    self._axis[1].clear()
                    self._axis[1].remove()
                    self._axis[1] = self._figure.add_subplot(
                        122, projection=None, aspect="auto"
                    )
                    self._axis[1].tick_params("y", labelright=True, labelleft=False)
                    self._axis[1].get_yaxis().set_label_position("right")
                    if debug:
                        self._logfile.write("in update calling ex_diagram")
                    self.ex_diagram(
                        axis=self._axis[1],
                        reset=False,
                        position=position,
                        size=1,
                        figsize=(5, 3),
                        norm=True,
                        show_fit=True,
                        ymin=0,
                        ymax=30,
                    )
                    if debug:
                        self._logfile.write(f"pos={position}")
            except Exception as err:
                if self._logfile is None:
                    pass
                else:
                    self._logfile.write("Exception {0}".format(err))

            if self._logfile:
                self._logfile.close()
            self._figure.canvas.draw_idle()

        if interaction_type == "move":
            self._figure.canvas.mpl_connect("motion_notify_event", update_lines)
        elif interaction_type == "click":
            self._figure.canvas.mpl_connect("button_press_event", update_lines)
        else:
            self._plt.close(self._figure)
            raise ValueError(
                f"{interaction_type} is not a valid option for interaction_type, valid options are 'click' or 'move'"
            )
