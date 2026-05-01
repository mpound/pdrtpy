import math
import warnings
from types import SimpleNamespace

import astropy.constants as constants
import astropy.units as u
import numpy as np
from astropy import log
from astropy.nddata import Cutout2D, StdDevUncertainty
from emcee.pbar import get_progress_bar
from lmfit import Parameters  # , fit_report
from lmfit.model import Model  # , ModelResult

from .. import molecule as mol, utils
from ..measurement import Measurement
from .fitmap import FitMap
from .toolbase import ToolBase

log.setLevel("WARNING")


class BaseExcitationFit(ToolBase):
    """
    Base class for creating excitation fitting tools for various species.

    Parameters
    ----------
    molecule : `~pdrtpy.molecule.BaseMolecule`
        The molecule whose transitions will be fit.
    measurements : :class:`~pdrtpy.measurement.Measurement` or dict, optional.
        Input measurements to be fit. If input is a dictionary of measurements, the keys must Measurement identifiers. The default is None.

    """

    def __init__(self, molecule: mol.BaseMolecule, measurements: dict | Measurement = None):

        super().__init__()
        self._molecule = molecule
        # must be set before call to init_measurements
        self._intensity_units = "erg cm^-2 s^-1 sr^-1"
        self._cd_units = "cm^-2"
        self._t_units = "K"
        self._numcomponents = 0  # number of components to fit. user-settable
        self._valid_components = ["hot", "cold", "total"]  # NB: this only allows for 2-component fit
        self._av_interp = None
        if isinstance(measurements, dict) or measurements is None:
            self._measurements = measurements
        else:
            self._init_measurements(measurements)
        self._set_measurementnaxis()
        self._molecule._transition_data = molecule.transition_data
        # @todo we don't really even use this.  CD's are computed on the fly in average_column_density()
        self._column_density = dict()
        self._canonical_opr = molecule.canonical_opr
        self._opr = Measurement(data=[self._canonical_opr], uncertainty=None)
        self._model_functions = {
            1: self._one_component_model,
            2: self._two_component_model,
        }
        self._fitresult = None
        self._temperature = None
        self._total_colden = None
        # position and size that was used for averaging/fit
        self._position = None
        self._size = None
        self._numcomponents = 2

    def _init_measurements(self, m: list):
        r"""Initialize measurements dictionary given a list.

        :param m: list of intensity :class:`~pdrtpy.measurement.Measurement`s in units equivalent to :math:`{\rm erg~cm^{-2}~s^{-1}~sr^{-1}}`
        :type m: list of :class:`~pdrtpy.measurement.Measurement`
        """
        self._measurements = dict()
        for mm in m:
            if not utils.check_units(mm.unit, self._intensity_units):
                raise TypeError(
                    f"Measurement {mm.id} units {mm.unit.to_string()} are not in intensity units equivalent to"
                    f" {self._intensity_units}"
                )
            self._measurements[mm.id] = mm

    def _is_ortho(self, identifier):
        """Determine if a transition is ortho or para.
        Always False if the molecule does not have a variable OPR
        """
        if not self.molecule.opr_can_vary:
            return False
        # identifier is J level
        if isinstance(identifier, int):
            return utils.is_odd(identifier)
        else:  # identifier is e.g., H210S7
            return self._molecule._transition_data.loc[identifier]["Ju"] % 2 != 0

    def _get_ortho_indices(self, ids):
        """Given a list of J values, return the indices of those that are ortho
        transitions (odd J)

        **If the molecule does not have a variable OPR, then all indices for the `ids` are returned.**

        :param ids:
        :type ids: list of str
        :returns: The array indices of the odd J values.
        :rtype: list of int
        """
        if not self.molecule.opr_can_vary:
            # log.warning(f"The molecule {self.molecule.name} does not have a variable OPR.")
            return np.where(self._molecule._transition_data.loc[ids]["Ju"])[0]
        return np.where(self._molecule._transition_data.loc[ids]["Ju"] % 2 != 0)[0]

    def _get_para_indices(self, ids):
        """Given a list of J values, return the indices of those that are para
        transitions (even J)

        **If the molecule does not have a variable OPR, then all indices for the `ids` are returned.**

        :param ids:
        :type ids: list of str
        :returns: The array indices of the even J values.
        :rtype: list of int
        """
        if not self.molecule.opr_can_vary:
            log.warning(f"The molecule {self.molecule.name} does not have a variable OPR.")
            return np.where(self._molecule._transition_data.loc[ids]["Ju"])[0]
        return np.where(self._molecule._transition_data.loc[ids]["Ju"] % 2 == 0)[0]

    ######################################################################################
    # Public user methods for managing measurements and running the fit
    ######################################################################################
    def add_measurement(self, m: Measurement):
        r"""Add an intensity Measurement to internal dictionary used to
        compute the excitation diagram.   This method can also be used
        to safely replace an existing intensity Measurement.

        :param m: A :class:`~pdrtpy.measurement.Measurement` instance containing intensity in units equivalent to :math:`{\rm erg~cm^{-2}~s^{-1}~sr^{-1}}`
        """
        if not utils.check_units(m.unit, self._intensity_units):
            raise TypeError(
                f"Measurement {m.id} units {m.unit.to_string()} are not in intensity units equivalent to"
                f" {self._intensity_units}"
            )
        if self._measurements:
            self._measurements[m.id] = m
            # if there is an existing column density with this ID, remove it
            self._column_density.pop(m.id, None)
        else:
            self._init_measurements(m)

    def remove_measurement(self, identifier: str):
        """Delete a measurement from the internal dictionary used to compute column densities. Any associated column density will also be removed.

        :param identifier: the measurement identifier
        :type identifier: str
        :raises KeyError: if identifier not in existing Measurements
        """
        del self._measurements[identifier]  # we want this to raise a KeyError if id not found
        self._column_density.pop(identifier, None)  # but not this.

    def replace_measurement(self, m: Measurement):
        r"""Safely replace an existing intensity Measurement.  Do not
        change a Measurement in place, use this method.
        Otherwise, the column densities will be inconsistent.

        :param m: A :class:`~pdrtpy.measurement.Measurement` instance containing intensity in units equivalent to :math:`{\rm erg~cm^{-2}~s^{-1}~sr^{-1}}`
        """
        self.add_measurement(m)

    def set_extinction_model(self, model):
        r"""
        Set the model to be used for fitting visual extinction, :math:`A_v`.  This is typically a model
        from the `~dust_extinction` package.

        Parameters
        ----------
        model : `~dust_extinction.baseclasses.BaseExtModel` or `astropy.modeling.Model`
            The model to be used to calculate dust extinction.

        Returns
        -------
        None.

        """
        self._extinction_model = model

    def run(self, position=None, size=None, fit_opr=False, fit_av=False, components=2, **kwargs):
        r"""Fit the :math:`log N_u-E` diagram with two excitation temperatures,
        a ``hot`` :math:`T_{ex}` and a ``cold`` :math:`T_{ex}`.

        If ``position`` and ``size`` are given, the data will be averaged over a spatial box before fitting.  The box is created using :class:`astropy.nddata.utils.Cutout2D`.  If position or size is None, the data are averaged over all pixels.  If the Measurements are single values, these arguments are ignored.

        :param position: The position of the cutout array's center with respect to the data array. The position can be specified either as a `(x, y)` tuple of pixel coordinates.
        :type position: tuple
        :param size: The size of the cutout array along each axis in pixels. If size is a scalar number or a scalar :class:`~astropy.units.Quantity`, then a square cutout of size will be created. If `size` has two elements, they should be in `(nx, ny)` order [*this is the opposite of Cutout2D signature*]. Scalar numbers in size are assumed to be in units of pixels.  Default value of None means use all pixels (position is ignored)
        :type size: int, array_like`
        :param fit_opr: Whether to fit the ortho-to-para ratio or not. If True, the OPR will be varied to determine the best value. If False, the OPR is fixed at the canonical LTE value of 3.
        :type fit_opr: bool
        :param fit_av: Whether to fit the visual extinction. If True, the Av will be varied to determine the best value. If False, the Av is fixed at zero.
        :type fit_av: bool
        """
        # @todo what happens if e.g., fit_av=True and init_av !=0 ?
        kwargs_opts = {
            "mask": None,
            "method": "leastsq",
            "nan_policy": "raise",
            "test": False,
            "verbose": False,
            "init_opr": 3.0,
            "init_av": 0.0,
            # for emcee
            "burn": 0,
            "steps": 1000,
            "nwalkers": 100,
        }
        kwargs_opts.update(kwargs)
        if fit_opr and not self.molecule.opr_can_vary:
            raise ValueError(
                "You can't fit the OPR of a molecule ({self.molecule.name}) in which the OPR doesn't vary."
            )
        if fit_opr and fit_av:
            raise ValueError(
                "You can't fit OPR and Av simultaneously. Pick one."
            )  # at least not unless you have many more points, right?
        if fit_av:
            if self.extinction_model is None:
                raise Exception(
                    f"You must set an excition model before fitting for Av. See {self.__class__.__name__}.set_extinction_model()"
                )
        self._numcomponents = components
        self._init_params()
        self._init_model()
        return self._fit_excitation(position, size, fit_opr, fit_av, **kwargs_opts)

    ###################################################################
    ## Methods having to do with parameter intialization and fitting
    ###################################################################
    def _init_params(self):
        """Initialze model fitting parameters."""
        # fit input parameters
        self._params = Parameters()
        # we have to have opr max be greater than 3 so that fitting will work.
        # the fit algorithm does not like when the initial value is pinned at one
        # of the limits
        # print(f'initializing parameters with nc = {self._numcomponents}')
        self._params.add(
            "opr", value=self.molecule.canonical_opr, min=1.0, max=self.molecule.canonical_opr * 1.2, vary=False
        )
        self._params.add("av", value=0.0, min=0.0, max=100, vary=False)
        self._params.add("m1", value=-1, max=0)
        self._params.add("n1", value=7, min=0, max=30)
        if self._numcomponents == 2:
            self._params.add("m2", value=-1, max=0)
            self._params.add("n2", value=75, min=0, max=30)
        # self._params.pretty_print()

    def _init_model(self):
        """Initialize the lmfit Model class to be used in fitting."""
        # @todo make a separate class that subclasses Model.

        # potentially allow users to change it.
        # print(f'initializing model with nc = {self._numcomponents}')
        self._model = Model(
            self._model_functions[self._numcomponents],
            param_names=list(self._params.keys()),
        )
        # This may be entirely unnecessary
        for p, q in self._params.items():
            self._model.set_param_hint(p, min=q.min, max=q.max, vary=q.vary)
        self._model.make_params()

    def _one_component_model(
        self,
        x,
        m1,
        n1,
        opr,
        av,
        idx=None,
        fit_opr=False,
        fit_av=False,
        extinction_ratio=None,
    ):
        """Function for fitting the excitation curve as a single linear function
        and allowing ortho-to-para ratio and/or visual extinctionto vary.  Para is even J, ortho is odd J.
        :param x: independent axis array
        :param m1: slope of line
        :type m1: float
        :param n1: intercept of line
        :type n1: float
        :param opr: ortho-to-para ratio
        :type opr: float
        :param av: visual extinction in magnitudes
        :type av: float
        :type idx: np.ndarray
        :param idx: list of indices that may have variable opr (odd J transitions)
        :param fit_opr: indicate whether opr will be fit, default False (opr fixed = 3)
        :type fit_opr: bool
        :param fit_av: indicate whether Av will be fit, default False (Av=0)
        :type fit_av: bool
        :param extinction_ratio: The ratio of spectral line wavelength extinction to visual extinction. See set_extinction_law()
        :type extinction_ratio: float
        :return: line in log space: x*m1 + n1 + log10(opr/3.0) -  0.4*extinction_ratio*av*log10(e)

        :rtype: :class:`numpy.ndarray`
        """
        if idx is None:
            idx = []
        idx = [int(i) for i in idx]
        # model is already in log space
        model = x * m1 + n1
        if fit_opr:
            model[idx] += np.log10(opr / self._canonical_opr)
        if fit_av:
            model = model - 0.4 * extinction_ratio * av * utils.LOGE
        return model

    def _one_line(self, x, m1, n1):
        """Return a line.

        :param x: array of x values
        :type x: :class:`numpy.ndarray`
        :param m1: slope of first line
        :type m1: float
        :param n1: intercept of first line
        :type n1: float
        """
        return m1 * x + n1

    def _two_component_model(
        self,
        x,
        m1,
        n1,
        m2,
        n2,
        opr,
        av,
        idx=None,
        fit_opr=False,
        fit_av=False,
        extinction_ratio=None,
    ):
        """Function for fitting the excitation curve as sum of two linear functions
        and allowing ortho-to-para ratio and/or visual extinction to vary.  Para is even J, ortho is odd J.
        :param x: independent axis array
        :param m1: slope of first line
        :type m1: float
        :param n1: intercept of first line
        :type n1: float
        :param m2: slope of second line
        :type m2: float
        :param n2: intercept of second line
        :type n2: float
        :param opr: ortho-to-para ratio
        :type opr: float
        :param av: visual extinction in magnitudes
        :type av: float
        :type idx: np.ndarray
        :param idx: list of indices that may have variable opr (odd J transitions)
        :param fit_opr: indicate whether opr will be fit, default False (opr fixed = 3)
        :type fit_opr: bool
        :param fit_av: indicate whether Av will be fit, default False (Av=0)
        :type fit_av: bool
        :param extinction_ratio: The ratio of spectral line wavelength extinction to visual extinction. See set_extinction_law()
        :type extinction_ratio: float
        :return: Sum of lines in log space:log10(10**(x*m1+n1) + 10**(x*m2+n2)) + log10(opr/3.0) - 0.4*extinction_ratio*av*log10(e)
        :rtype: :class:`numpy.ndarray`
        """
        if idx is None:
            idx = []
        # why are these coming in as floats?
        idx = [int(i) for i in idx]
        # print(f"{x=}, {m1=}, {n1=}, {m2=}, {n2=}")
        y1 = 10 ** (x * m1 + n1)
        y2 = 10 ** (x * m2 + n2)
        # print(f"{y1=}, {y2=}")
        model = np.log10(y1 + y2)
        # We assume that the column densities passed in have been normalized
        # using the canonical OPR=3. Therefore what we are actually fitting is
        # the ratio of the actual OPR to the canonical OPR.
        # For odd J, input x = Nu/(3*(2J+1) where 3=canonical OPR.
        #
        # We want the model-data residual to be small, but if the opr
        # is different from the  canonical value of 3, then data[idx] will
        # be low by a factor of 3/opr.
        # So we must LOWER model[idx] artificially by dividing it by
        # 3/opr, i.e. multiplying by opr/3.  This is equivalent to addition in log-space.
        if fit_opr:
            model[idx] += np.log10(opr / self._canonical_opr)
        if fit_av:
            model = model - 0.4 * extinction_ratio * av * utils.LOGE
        return model

    def _two_lines(self, x, m1, n1, m2, n2):
        """This function is used to partition a fit to data using two lines and
        an inflection point.  Second slope is steeper because slopes are
        negative in excitation diagram.

        :param x: array of x values
        :type x: :class:`numpy.ndarray`
        :param m1: slope of first line
        :type m1: float
        :param n1: intercept of first line
        :type n1: float
        :param m2: slope of second line
        :type m2: float
        :param n2: intercept of second line
        :type n2: float

         See https://stackoverflow.com/questions/48674558/how-to-implement-automatic-model-determination-and-two-state-model-fitting-in-py
        """
        return np.max([m1 * x + n1, m2 * x + n2], axis=0)

    #############################
    # Properties
    #############################
    @property
    def fit_result(self):
        """The result of the fitting procedure which includes fit statistics, variable values and uncertainties, and correlations between variables.

        :rtype:  :class:`lmfit.model.ModelResult`
        """
        return self._fitresult

    @property
    def numcomponents(self):
        """Number of temperature components in the fit

        :rtype: int
        """
        return self._numcomponents

    @property
    def av_fitted(self):
        """Was the visual extinction fitted?

        :returns: True if Av was fitted, False if not
        :rtype: bool
        """
        if self._fitresult is None:
            return False
        return self._params["av"].vary

    @property
    def av(self):
        """The visual extinction

        :returns: The fitted Av if it was determined in the fit, otherwise 0
        :rtype: :class:`~pdrtpy.measurement.Measurement`
        """
        return self._av

    @property
    def opr_fitted(self):
        """Was the ortho-to-para ratio fitted?

        :returns: True if OPR was fitted, False if canonical LTE value was used or this molecule's OPR canno vary.
        :rtype: bool
        """
        if not self.molecule.opr_can_vary:
            return False
        if self._fitresult is None:
            return False
        return self._params["opr"].vary

    @property
    def opr(self):
        """The ortho-to-para ratio (OPR)

        :returns: The fitted OPR if it was determined in the fit, otherwise the canonical LTE OPR
        :rtype: :class:`~pdrtpy.measurement.Measurement`
        """
        return self._opr

    @property
    def molecule(self) -> mol.BaseMolecule:
        """
        The molecule being fitted by this ExcitationFit

        Returns
        -------
        Molecule
            The molecule as represented by the `~pdrtpy.molecule.Molecule` class.

        """
        return self._molecule

    @property
    def intensities(self):
        """The stored intensities. See :meth:`add_measurement`

        :rtype: list of :class:`~pdrtpy.measurement.Measurement`
        """
        return self._measurements

    @property
    def total_colden(self):
        """The fitted total column density

        :rtype: :class:`~pdrtpy.measurement.Measurement`
        """
        if self._numcomponents == 1:
            return self._total_colden["cold"]
        return self._total_colden["hot"] + self._total_colden["cold"]

    @property
    def hot_colden(self):
        """The fitted hot gas total column density

        :rtype: :class:`~pdrtpy.measurement.Measurement`
        """
        return self._total_colden["hot"]

    @property
    def cold_colden(self):
        """The fitted cold gas total column density

        :rtype: :class:`~pdrtpy.measurement.Measurement`
        """
        return self._total_colden["cold"]

    @property
    def tcold(self):
        """The fitted cold gas excitation temperature

        :rtype: :class:`~pdrtpy.measurement.Measurement`
        """
        return self._temperature["cold"]  # self._fitparams.tcold

    @property
    def thot(self):
        """The fitted hot gas excitation temperature

        :rtype: :class:`~pdrtpy.measurement.Measurement`
        """
        return self._temperature["hot"]  # self._fitparams.thot

    @property
    def temperature(self):
        """The fitted gas temperatures, returned in a dictionary with keys 'hot' and 'cold'.
        :rtype: dict
        """
        return self._temperature

    @property
    def extinction_model(self):
        r"""
        The extinction law used when fitting for visual extinction, :math:`A_v`.

        Returns
        -------
        model : `~dust_extinction.baseclasses.BaseExtModel` or `astropy.modeling.Model`
            The model to be used to calculate dust extinction.

        """
        return self._extinction_model

    ############################################
    # Attributes that require a computation
    ############################################

    def colden(self, component):  # ,log=False):
        """The column density of hot or cold gas component, or total column density.

        :param component: 'hot', 'cold', or 'total
        :type component: str

        :rtype: :class:`~pdrtpy.measurement.Measurement`
        """
        #:param log: take the log10 of the column density
        cl = component.lower()
        if cl not in self._valid_components:
            raise KeyError(f"{cl} not a valid component. Must be one of {self._valid_components}")
        if cl == "total":
            return self.total_colden
        else:
            return self._total_colden[cl]

    def column_densities(self, norm=False, unit=utils._CM2, line=True):
        r"""The computed upper state column densities of stored intensities

        :param norm: if True, normalize the column densities by the
                    statistical weight of the upper state, :math:`g_u`.
                    Default: False
        :type norm: bool
        :param unit: The units in which to return the column density. Default: :math:`{\rm cm}^{-2}`
        :type unit: str or :class:`astropy.units.Unit`
        :param line: if True, the dictionary index is the Line name,
                  otherwise it is the upper state :math:`J` number.  Default: False
        :type line: bool

        :returns: dictionary of column densities indexed by upper state :math:`J` number or Line name. Default: False means return indexed by :math:`J`.
        :rtype: dict
        """
        # Compute column densities if needed.
        # Note: this has a gotcha - if user changes an existing intensity
        # Measurement in place, rather than replaceMeasurement(), the colden
        # won't get recomputed. But we warned them!
        # if not self._column_density or (len(self._column_density) != len(self._measurements)):

        # screw it. just always compute them.  Note to self: change this if it becomes computationally intensive
        # suppress ridiculous NDDATA warning about units. See issue #163
        log.setLevel("WARNING")
        self._compute_column_densities(unit=unit, line=line)
        if norm:
            cdnorm = dict()
            for cd in self._column_density:
                if line:
                    denom = self._molecule._transition_data.loc[cd]["gu"]
                else:
                    denom = self._molecule._transition_data.loc["Ju", cd]["gu"]
                    if len(denom) > 0:
                        denom = denom[0]  # ARGH kluge.  Need to get rid of f option as Ju is no longer unique
                # This fails with complaints about units:
                # self._column_density[cd] /= self._molecule._transition_data.loc[cd]["gu"]
                # gu = Measurement(self._molecule._transition_data.loc[cd]["gu"],unit=u.dimensionless_unscaled)
                cdnorm[cd] = self._column_density[cd] / denom
                cdnorm[cd]._identifier = cd
            # return #self._column_density
            return cdnorm
        else:
            return self._column_density

    def average_column_density(
        self,
        position=None,
        size=None,
        norm=True,
        unit=utils._CM2,
        line=True,
        clip=None,
    ):
        r"""Compute the average column density over a spatial box.  The box is created using :class:`astropy.nddata.utils.Cutout2D`.

        :param position: The position of the cutout array's center with respect to the data array. The position can be specified either as a `(x, y)` tuple of pixel coordinates.
        :type position: tuple
        :param size: The size of the cutout array along each axis. If size is a scalar number or a scalar :class:`~astropy.units.Quantity`, then a square cutout of size will be created. If `size` has two elements, they should be in `(nx,ny)` order [*this is the opposite of Cutout2D signature*]. Scalar numbers in size are assumed to be in units of pixels.  Default value of None means use all pixels (position is ignored)
        :type size: int, array_like`
        :param norm: if True, normalize the column densities by the
                       statistical weight of the upper state, :math:`g_u`.  For ortho-:math:`H_2`, :math:`g_u = OPR \times (2J+1)`, for para-:math:`H_2`, :math:`g_u=2J+1`. In LTE, :math:`OPR = 3`.
        :type norm: bool
        :param unit: The units in which to return the column density. Default: :math:`{\rm cm}^{-2}`
        :type unit: str or :class:`astropy.units.Unit`
        :param line: if True, the returned dictionary index is the Line name, otherwise it is the upper state :math:`J` number.
        :type line: bool
        :returns: dictionary of column density Measurements, with keys as :math:`J` number or Line name
        :rtype:  dict
        :param clip: Column density value at which to clip pixels. Pixels with column densities below this value will not be used in the average. Default: a large negative number, which translates to no clipping.
        :type clip: :class:`astropy.units.Quantity`
        """
        # @todo
        # - should default clip = None?
        # suppress ridiculous NDDATA warning about units. See issue #163
        log.setLevel("WARNING")
        # Set norm=False because we normalize below if necessary.
        if position is not None and size is None:
            print("WARNING: ignoring position keyword since no size given")
        if position is None and size is not None:
            raise Exception("You must supply a position in addition to size for cutout")
        if size is not None:
            if np.isscalar(size):
                size = np.array([size, size])
            else:
                # Cutout2D wants (ny,nx)
                size = np.array([size[1], size[0]])

        if clip is None:
            clip = -1e40 * u.Unit("cm-2")
        clip = clip.to("cm-2")
        cdnorm = self.column_densities(norm=norm, unit=unit, line=line)
        cdmeas = dict()
        for cd in cdnorm:
            ca = cdnorm[cd]
            if size is not None:
                if len(size) != len(ca.shape):
                    raise Exception(f"Size dimensions [{len(size)}] don't match measurements [{len(ca.shape)}]")

                # if size[0] > ca.shape[0] or size[1] > ca.shape[1]:
                #    raise Exception(f"Requested cutout size {size} exceeds measurement size {ca.shape}")
                cutout = Cutout2D(ca.data, position, size, ca.wcs, mode="trim", fill_value=np.nan)
                w = Cutout2D(
                    ca.uncertainty.array,
                    position,
                    size,
                    ca.wcs,
                    mode="trim",
                    fill_value=np.nan,
                )
                cddata = np.ma.masked_array(
                    cutout.data,
                    mask=np.ma.mask_or(np.isnan(cutout.data), cutout.data < clip.value),
                )
                weights = np.ma.masked_array(w.data, np.isnan(w.data))
            else:
                cddata = ca.data
                # handle corner case of measurment.data is shape = (1,)
                # and StdDevUncertainty.array is shape = ().
                # They both have only one value but StdDevUncertainty stores
                # its data in a peculiar way.
                # alternative: check that type(ca.uncertainty.array) == np.ndarray would also work.
                if np.shape(ca.data) == (1,) and np.shape(ca.uncertainty.array) == ():
                    weights = np.array([ca.uncertainty.array])
                else:
                    weights = ca.uncertainty.array
            if np.sum(weights) == 0:
                cdavg = np.average(cddata)
            else:
                cdavg = np.average(cddata, weights=weights)
            error = np.nanmean(ca.error) / np.sqrt(ca.error.size)  # -1
            cdmeas[cd] = Measurement(
                data=cdavg,
                uncertainty=StdDevUncertainty(error),
                unit=ca.unit,
                identifier=cd,
            )
        # log.setLevel("INFO")
        return cdmeas

    def energies(self, line=True):
        # @todo remove unit if transition_data is changed to QTable
        r"""Upper state energies of stored intensities, in K.

        :param line: if True, the dictionary index is the Line name,
                  otherwise it is the upper state :math:`J` number.  Default: False
        :type line: bool
        :returns: dictionary indexed by upper state :math:`J` number or Line name. Default: False means return indexed by :math:`J`.
        :rtype: dict
        """
        t = dict()
        if line:
            for m in self._measurements:
                t[m] = self._molecule._transition_data.loc[m]["Tu"]
        else:
            for m in self._measurements:
                t[self._molecule._transition_data.loc[m]["Ju"]] = self._molecule._transition_data.loc[m]["Tu"]
        return t

    def wavelengths(self, line=True, units=False):
        r"""Wavelengths of transitions, in micron (assumed unit using Roueff et al table)

        :param line: if True, the dictionary index is the Line name,
                  otherwise it is the upper state :math:`J` number.  Default: False
        :type line: bool
        :param units: if True, values are returned with units as astropy Quantity
        :type units: bool
        :returns: dictionary indexed by upper state :math:`J` number or Line name. Default: False means return indexed by :math:`J`.
        :rtype: dict
        """
        # @todo remove unit if transition_data is changed to QTable
        t = dict()
        if units:
            x = self._molecule._transition_data["lambda"].unit
        else:
            x = 1
        if line:
            for m in self._measurements:
                t[m] = self._molecule._transition_data.loc[m]["lambda"] * x
        else:
            for m in self._measurements:
                t[self._molecule._transition_data.loc[m]["Ju"]] = self._molecule._transition_data.loc[m]["lambda"] * x
        return t

    def gu(self, id, opr):
        r"""Get the upper state statistical weight :math:`g_u` for the given transition identifer, and, if the transition is odd-:math:`J`, scale the result by the given ortho-to-para ratio.  If the transition is even-:math:`J`, the LTE value is returned.

        :param id: the measurement identifier
        :type id: str
        :param opr:
        :type opr: float
        :raises KeyError: if `id` not in existing Measurements
        :rtype: float
        """
        if not self.molecule.opr_can_vary:
            log.warning(f"The molecule {self.molecule.name} does not have a variable OPR.")
            return self._molecule._transition_data.loc[id]["gu"]
        if utils.is_even(self._molecule._transition_data.loc[id]["Ju"]):
            return self._molecule._transition_data.loc[id]["gu"]
        else:
            # print("Ju=%d scaling by [%.2f/%.2f]=%.2f"%(self._molecule._transition_data.loc[id]["Ju"],opr,self._canonical_opr,opr/self._canonical_opr))
            return self._molecule._transition_data.loc[id]["gu"] * opr / self._canonical_opr

    def intensity(self, colden):
        r"""Given an upper state column density :math:`N_u`, compute the intensity :math:`I`.

           .. math::
                 I = {A \Delta E~N_u \over 4\pi}

        where :math:`A` is the Einstein A coefficient and :math:`\Delta E` is the energy of the transition.

        :param colden: upper state column density
        :type colden: :class:`~pdrtpy.measurement.Measurement`
        :returns: optically thin intensity
        :rtype: :class:`~pdrtpy.measurement.Measurement`
        """
        # colden is N_upper
        # @todo remove unit if transition_data is changed to QTable
        dE = (
            self._molecule._transition_data.loc[colden.id]["dE"]
            * constants.k_B.cgs
            * self._molecule._transition_data["dE"].unit
        )
        A = self._molecule._transition_data.loc[colden.id]["A"] * self._molecule._transition_data["A"].unit
        v = A * dE / (4.0 * math.pi * u.sr)
        val = Measurement(data=v.value, unit=v.unit, identifier=colden.id)
        intensity = val * colden  # error will get propagated
        i = intensity.convert_unit_to(self._intensity_units)
        i._identifier = val.id
        return i

    def upper_colden(self, intensity, unit):
        r"""Compute the column density in upper state :math:`N_u`, given an
        intensity :math:`I` and assuming optically thin emission.
        Units of :math:`I` need to be equivalent to
        :math:`{\rm erg~cm^{-2}~s^{-1}~sr^{-1}}`.

        .. math::
              I &= {A \Delta E~N_u \over 4\pi}

              N_u &= 4\pi {I\over A\Delta E}

        where :math:`A` is the Einstein A coefficient and :math:`\Delta E` is the energy of the transition.

        :param intensity: A :class:`~pdrtpy.measurement.Measurement` instance containing intensity in units equivalent to :math:`{\rm erg~cm^{-2}~s^{-1}~sr^{-1}}`
        :type intensity: :class:`~pdrtpy.measurement.Measurement`
        :param unit: The units in which to return the column density. Default: :math:`{\rm cm}^{-2}`
        :type unit: str or :class:`astropy.units.Unit`
        :returns: a :class:`~pdrtpy.measurement.Measurement` of the column density.
        :rtype: :class:`~pdrtpy.measurement.Measurement`
        """
        # suppress ridiculous NDDATA warning about units. See issue #163
        log.setLevel("WARNING")
        dE = (
            self._molecule._transition_data.loc[intensity.id]["dE"]
            * constants.k_B.cgs
            * self._molecule._transition_data["dE"].unit
        )
        A = self._molecule._transition_data.loc[intensity.id]["A"] * self._molecule._transition_data["A"].unit
        v = 4.0 * math.pi * u.sr / (A * dE)
        val = Measurement(data=v.value, unit=v.unit, identifier=intensity.id)
        N_upper = intensity * val  # error will get propagated
        N_upper = N_upper.convert_unit_to(unit)
        N_upper._identifier = intensity.id
        # log.setLevel("INFO")
        return N_upper

    def _compute_column_densities(self, unit=utils._CM2, line=True):
        r"""Compute all upper level column densities for stored intensity measurements and puts them in a dictionary
        :param unit: The units in which to return the column density. Default: :math:`{\rm }cm^{-2}`
        :type unit: str or :class:`astropy.units.Unit`
        :param line: if True, the dictionary index is the Line name,
                  otherwise it is the upper state :math:`J` number.  Default: False
        :type line: bool

         # should we reutrn something here or just compute them and never store.
         # I'm beginning to think there is no reason to store them.
        #:700returns: dictionary of column densities as:class:`~pdrtpy.measurement.Measurement  indexed by upper state :math:`J` number or Line name. Default: False means return indexed by :math:`J`.
        #:returns: a :class:`~pdrtpy.measurement.Measurement` of the column density.
        """
        self._column_density = dict()
        for m in self._measurements:
            if line:
                index = m
            else:
                index = self._molecule._transition_data.loc[m]["Ju"]
            self._column_density[index] = self.upper_colden(self._measurements[m], unit)

    #########################################
    # Methods before or after fitting
    #########################################
    def _extract_fitted_params(self, fmdata, ffmask, n_pix, param_names):
        """Pull fitted parameter values and stderrs from each pixel's ModelResult.

        Returns a dict mapping each name in ``param_names`` to ``(values, stderrs)``,
        each a length-``n_pix`` flat float array. Masked pixels and missing stderrs
        are NaN-filled, so downstream vectorized math propagates NaN cleanly into
        the output Measurement masks.
        """
        out = {p: (np.full(n_pix, np.nan), np.full(n_pix, np.nan)) for p in param_names}
        for i in range(n_pix):
            if ffmask[i]:
                continue
            params = fmdata[i].params
            for name in param_names:
                p = params[name]
                out[name][0][i] = p.value
                if p.stderr is not None:
                    out[name][1][i] = p.stderr
        return out

    def _flag_bad_stderr_pixels(self, fmdata, ffmask, n_pix, param_names):
        """Mask pixels where any varying parameter has a None stderr and warn.

        Mutates ``ffmask`` in place. Replaces the previous behavior of raising
        on the first bad pixel, which could abort an entire map fit.
        """
        bad = []
        for i in range(n_pix):
            if ffmask[i]:
                continue
            params = fmdata[i].params
            missing = [p for p in param_names if params[p].vary and params[p].stderr is None]
            if missing:
                bad.append((i, missing))
                ffmask[i] = True
        if bad:
            preview = bad[:10]
            suffix = f" (and {len(bad) - 10} more)" if len(bad) > 10 else ""
            warnings.warn(
                f"Could not calculate stderrs for {len(bad)} pixel(s); the "
                f"{self._numcomponents}-temperature model may be inappropriate for these. "
                f"Pixels have been masked. Affected (pixel, params): {preview}{suffix}",
                UserWarning,
                stacklevel=2,
            )

    def _wrap_measurement(self, data, err, unit, fitmap):
        """Build a Measurement from flat or shaped arrays, masking non-finite values."""
        mask = fitmap.mask | np.logical_not(np.isfinite(data))
        return Measurement(
            data=data,
            unit=unit,
            uncertainty=StdDevUncertainty(np.abs(err)),
            wcs=fitmap.wcs,
            mask=mask,
        )

    def _compute_quantities(self, fitmap):
        """Compute temperatures and column densities for the hot and cold gas components.

        Sets ``self._temperature``, ``self._j0_colden``, ``self._total_colden``,
        ``self._opr``, and ``self._av`` from a fitted FitMap.
        """
        self._temperature = dict()
        self._j0_colden = dict()
        self._total_colden = dict()

        if self._numcomponents == 2:
            param_names = ("m1", "n1", "m2", "n2", "opr", "av")
        elif self._numcomponents == 1:
            param_names = ("m1", "n1", "opr", "av")
        else:
            raise Exception(f"Bad numcomponents: {self._numcomponents}")

        n_pix = fitmap.data.size
        map_shape = fitmap.data.shape
        fmdata = fitmap.data.flatten()
        ffmask = fitmap.mask.flatten().copy()

        self._flag_bad_stderr_pixels(fmdata, ffmask, n_pix, param_names)
        extracted = self._extract_fitted_params(fmdata, ffmask, n_pix, param_names)

        if self._numcomponents == 2:
            # Per-pixel cold/hot assignment: cold is the steeper (more negative) slope.
            m1_v, m1_e = extracted["m1"]
            m2_v, m2_e = extracted["m2"]
            n1_v, n1_e = extracted["n1"]
            n2_v, n2_e = extracted["n2"]
            cold_is_2 = m2_v < m1_v
            m_cold = np.where(cold_is_2, m2_v, m1_v)
            m_cold_err = np.where(cold_is_2, m2_e, m1_e)
            n_cold = np.where(cold_is_2, n2_v, n1_v)
            n_cold_err = np.where(cold_is_2, n2_e, n1_e)
            m_hot = np.where(cold_is_2, m1_v, m2_v)
            m_hot_err = np.where(cold_is_2, m1_e, m2_e)
            n_hot = np.where(cold_is_2, n1_v, n2_v)
            n_hot_err = np.where(cold_is_2, n1_e, n2_e)
        else:
            m_cold, m_cold_err = extracted["m1"]
            n_cold, n_cold_err = extracted["n1"]
            m_hot, m_hot_err = m_cold, m_cold_err
            n_hot, n_hot_err = n_cold, n_cold_err

        with np.errstate(invalid="ignore", divide="ignore"):
            tc = (-utils.LOGE / m_cold).reshape(map_shape)
            tc_err = np.abs(tc * (m_cold_err / m_cold).reshape(map_shape))
            th = (-utils.LOGE / m_hot).reshape(map_shape)
            th_err = np.abs(th * (m_hot_err / m_hot).reshape(map_shape))
            nc = (10.0**n_cold).reshape(map_shape)
            nc_err = (utils.LN10 * n_cold_err * (10.0**n_cold)).reshape(map_shape)
            nh = (10.0**n_hot).reshape(map_shape)
            nh_err = (utils.LN10 * n_hot_err * (10.0**n_hot)).reshape(map_shape)

        opr_v = extracted["opr"][0].reshape(map_shape)
        opr_e = extracted["opr"][1].reshape(map_shape)
        av_v = extracted["av"][0].reshape(map_shape)
        av_e = extracted["av"][1].reshape(map_shape)

        self._temperature["cold"] = self._wrap_measurement(tc, tc_err, self._t_units, fitmap)
        self._j0_colden["cold"] = self._wrap_measurement(nc, nc_err, self._cd_units, fitmap)
        if self._numcomponents == 2:
            self._temperature["hot"] = self._wrap_measurement(th, th_err, self._t_units, fitmap)
            self._j0_colden["hot"] = self._wrap_measurement(nh, nh_err, self._cd_units, fitmap)
            self._total_colden["hot"] = self._j0_colden["hot"] * self.molecule.partition_function(self.thot)
        else:
            self._temperature["hot"] = self._temperature["cold"]
            self._j0_colden["hot"] = self._j0_colden["cold"]
        self._total_colden["cold"] = self._j0_colden["cold"] * self.molecule.partition_function(self.tcold)
        if self._numcomponents == 1:
            self._total_colden["hot"] = self._total_colden["cold"]

        self._opr = self._wrap_measurement(opr_v, opr_e, u.dimensionless_unscaled, fitmap)
        self._av = self._wrap_measurement(av_v, av_e, u.dimensionless_unscaled, fitmap)

    def _first_guess(self, x, y):
        r"""The first guess at the fit parameters is done by finding the line between the first
        two (lowest energy) points to determine $T_{cold}and between the last two (highest energy)
        points to determine $T_{hot}. The first guess is needed to ensure the final fit converges.
        The guess doesn't need to be perfect, just in the ballpark.

        :param x: array of energies, $E/k$
        :type x: numpy array
        :param y: array of normalized column densities $N_u/g_u$
        :type y: numpy array
        """

        if self._numcomponents == 2:
            slopecold = (y[2] - y[0]) / (x[2] - x[0])
            intcold = y[1] - slopecold * x[1]
            slopehot = (y[-1] - y[-2]) / (x[-1] - x[-2])
            inthot = y[-1] - slopehot * x[-1]
        else:
            slopecold = (y[-1] - y[0]) / (x[-1] - x[0])
            intcold = y[-1] - slopecold * x[-1]
            slopehot = slopecold
            inthot = intcold
        if np.all(slopecold >= 0):
            # (f"Bad first guess, resetting from {slopecold=} to -0.5")
            slopecold = np.full_like(slopecold, -0.5)
        # print("FG ",type(slopecold),type(slopehot),type(intcold),type(inthot))
        return np.array([slopecold, intcold, slopehot, inthot])

    def _fit_excitation(self, position, size, fit_opr=False, fit_av=False, **kwargs):
        r"""Fit the :math:`log N_u-E` diagram with one or two excitation temperatures.

        A first-pass guess is made by partitioning the data and fitting two lines
        (or one, depending on ``self._numcomponents``). If ``position`` and ``size``
        are both given, the data are averaged over a spatial box (``Cutout2D``)
        before fitting; otherwise every pixel is fit independently.

        :param position: ``(x, y)`` pixel coordinate or :class:`~astropy.coordinates.SkyCoord`.
        :param size: scalar pixel size or ``(nx, ny)`` tuple.
        :param fit_opr: vary the ortho-to-para ratio.
        :param fit_av: vary the visual extinction.
        """
        verbose = kwargs.pop("verbose")
        prep = self._prep_fit_data(
            position,
            size,
            fit_opr,
            fit_av,
            kwargs.pop("init_opr", 3.0),
            kwargs.pop("init_av", 0.0),
            verbose,
        )
        fmdata, fm_mask, count = self._run_pixel_fits(prep, fit_opr, fit_av, kwargs, verbose)
        count = self._cleanup_fits(fmdata, fm_mask, count, verbose)
        warnings.resetwarnings()
        self._reshape_results(fmdata, fm_mask, prep.saveshape, prep.colden_wcs)
        self._compute_quantities(self._fitresult)
        if verbose:
            print(f"fitted {count} of {prep.total} pixels")
            print(f"got {self._excount} exceptions and {self._badfit} bad fits")
        self._position = position
        self._size = size

    def _prep_fit_data(self, position, size, fit_opr, fit_av, init_opr, init_av, verbose):
        """Validate inputs, build energy/column-density vectors, run first-guess.

        Returns a :class:`SimpleNamespace` with everything the pixel loop needs:
        flat ``yr`` (data), ``sig`` (sigma), per-pixel first-guess slopes/intercepts,
        ``saveshape`` for later reshape, ``colden_wcs`` for the FitMap, and the
        precomputed extinction ratios.
        """
        min_points = self._numcomponents * 2
        if fit_opr:
            min_points += 1
        else:
            self._opr = Measurement(data=[self._canonical_opr], uncertainty=None)
        if fit_av:
            min_points += 1
            wavelengths = list(self.wavelengths(line=True).values()) * u.micron
            extinction_ratios = self.extinction_model(wavelengths)
        else:
            self._av = Measurement(data=[0.0], uncertainty=None)
            extinction_ratios = None

        self._params["opr"].vary = fit_opr
        if fit_opr:
            self._params["opr"].value = init_opr
        self._params["av"].vary = fit_av
        if fit_av:
            self._params["av"].value = init_av

        energy = self.energies(line=True)
        _ee = np.array(list(energy.values()))
        if len(_ee) < min_points:
            raise Exception(
                f"You need at least {min_points:d} data points to determine {self._numcomponents}-temperature model"
            )
        if len(_ee) == min_points:
            warnings.warn(
                f"Number of data points is equal to number of free parameters ({min_points:d}). "
                "Fit will be over-constrained",
                stacklevel=2,
            )
        idx = self._get_ortho_indices(list(energy.keys()))

        if position is None or size is None:
            colden = self.column_densities(norm=True, line=True)
        else:
            colden = self.average_column_density(norm=True, position=position, size=size, line=True)

        _cd = np.squeeze(np.array([c.data for c in colden.values()]))
        _er = np.squeeze(np.array([c.error for c in colden.values()]))
        _colden = Measurement(_cd, uncertainty=StdDevUncertainty(_er), unit="cm-2")
        x = _ee
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            y = np.log10(_colden.data)
        sigma = utils.LOGE * _colden.error / _colden.data

        slopecold, intcold, slopehot, inthot = self._first_guess(x, y)
        tcold = -utils.LOGE / slopecold
        thot = -utils.LOGE / slopehot
        if np.shape(tcold) == ():
            tcold = np.array([tcold])
            thot = np.array([thot])
        saveshape = tcold.shape
        if verbose:
            print(f"First guess at excitation temperatures:\n T_cold = {tcold:.1f} K\n T_hot = {thot:.1f} K")

        # flatten any dimensions past 0 so the pixel loop indexes are 1-D
        shp = y.shape
        if len(shp) == 1:
            y = y[:, np.newaxis]
            shp = y.shape
        n_pix = int(np.prod(shp[1:]))

        return SimpleNamespace(
            x=x,
            yr=y.reshape((shp[0], n_pix)),
            sig=sigma.reshape((shp[0], n_pix)),
            idx=idx,
            slopecold=slopecold.flatten(),
            intcold=intcold.flatten(),
            slopehot=slopehot.flatten(),
            inthot=inthot.flatten(),
            extinction_ratios=extinction_ratios,
            saveshape=saveshape,
            total=n_pix,
            colden_wcs=colden[utils.firstkey(colden)].wcs,
        )

    def _run_pixel_fits(self, prep, fit_opr, fit_av, kwargs, verbose):
        """Run lmfit on every pixel. Returns ``(fmdata, fm_mask, count)``.

        Sets ``self._excount`` (ValueError count) and ``self._badfit`` (fits that
        completed but reported success=False).
        """
        total = prep.total
        fmdata = np.empty(total, dtype=object)
        fm_mask = np.full(total, False)
        count = 0
        self._excount = 0
        self._badfit = 0

        # Suppress lmfit's incorrect warning about model parameters during the loop.
        # Caller is responsible for `warnings.resetwarnings()` after we return.
        warnings.simplefilter("ignore", category=UserWarning)

        self._model.set_param_hint("opr", vary=fit_opr)
        self._model.set_param_hint("av", vary=fit_av)

        progress = kwargs.pop("progress", True) if total > 1 else False
        method = kwargs["method"]
        emcee_kwargs = (
            {k: kwargs[k] for k in ("burn", "steps", "nwalkers") if k in kwargs} if method == "emcee" else None
        )

        # lmfit.Model.fit deepcopies the params argument before minimizing, so we
        # can reuse one Parameters object across pixels and just overwrite the
        # per-pixel starting .value entries each iteration.
        p = self._params.copy()

        with get_progress_bar(progress, total, leave=True, position=0) as pbar:
            for i in range(total):
                if not (np.isfinite(prep.yr[:, i]).all() and np.isfinite(prep.sig[:, i]).all()):
                    if verbose:
                        print("Bad fit because NaNs in data")
                    fm_mask[i] = True
                    pbar.update(1)
                    continue
                p["n1"].value = prep.intcold[i]
                p["m1"].value = prep.slopecold[i]
                if self._numcomponents == 2:
                    p["n2"].value = prep.inthot[i]
                    p["m2"].value = prep.slopehot[i]
                try:
                    fmdata[i] = self._model.fit(
                        data=prep.yr[:, i],
                        weights=1.0 / prep.sig[:, i],
                        x=prep.x,
                        params=p,
                        idx=prep.idx,
                        fit_opr=fit_opr,
                        fit_av=fit_av,
                        extinction_ratio=prep.extinction_ratios,
                        method=method,
                        nan_policy=kwargs["nan_policy"],
                        fit_kws=emcee_kwargs,
                    )
                    if fmdata[i].success:
                        count += 1
                    else:
                        if verbose:
                            print(
                                f"Bad fit because 'success' value ({fmdata[i].success}) "
                                f"or errorbars ({fmdata[i].errorbars}) was False."
                            )
                        fm_mask[i] = True
                        self._badfit += 1
                except ValueError as v:
                    print(f"Bad fit because {v}")
                    fm_mask[i] = True
                    self._excount += 1
                pbar.update(1)

        return fmdata, fm_mask, count

    def _cleanup_fits(self, fmdata, fm_mask, count, verbose):
        """Mark pixels that completed but reported a None stderr on a varying
        parameter. Mutates ``fmdata`` and ``fm_mask`` in place. Returns the
        adjusted successful-fit count.
        """
        for ii in range(len(fmdata)):
            fmd = fmdata[ii]
            if fmd is None:
                continue
            badstderr = False
            for p in fmd.params:
                if fmd.params[p].stderr is None and fmd.params[p].vary:
                    if verbose:
                        print(f"Fit completed at pixel {ii} but stderr for parameter {p} is None. Setting mask.")
                        if self._numcomponents == 2:
                            print("Try fitting a single component instead.")
                    fmdata[ii].success = False
                    fm_mask[ii] = True
                    self._badfit += 1
                    badstderr = True
            if badstderr:
                count -= 1
        return count

    def _reshape_results(self, fmdata, fm_mask, saveshape, colden_wcs):
        """Build ``self._fitresult`` from the flat fit arrays."""
        self._fitresult = FitMap(
            fmdata.reshape(saveshape),
            wcs=colden_wcs,
            mask=fm_mask.reshape(saveshape),
            name="result",
        )


# ========================== END BASEEXCITATION FIT ===================================================


# ========================== DERIVED CLASSES FOR SPECIFIC MOLECULES ===================================
class H2ExcitationFit(BaseExcitationFit):
    def __init__(self, measurements: Measurement = None):
        r"""Tool for fitting temperatures, column densities, `A_v`, and ortho-to-para ratio(`OPR`) from an :math:`H_2`
        excitation diagram. It takes as input a set of :math:`H_2` rovibrational line observations with errors
        represented as :class:`~pdrtpy.measurement.Measurement`.

        Often, excitation diagrams show evidence of both "hot" and "cold" gas components, where the cold gas
        dominates the intensity in the low :math:`J` transitions and the hot gas dominates in the high :math:`J` transitions.
        Given data over several transitions, one can fit for :math:`T_{cold}, T_{hot}, N_{total} = N_{cold}+ N_{hot}`,
        and optionally :math:`A_v` or :math:`OPR`. One needs at least 5 points to fit two temperatures and column
        densities (slope and intercept :math:`\times 2`), though one could compute (not fit) them with only 4 points.
        To additionally fit :math:`A_v` or :math:`OPR`, one should have 6 points (5 degrees of freedom).

        Once the fit is done, :class:`~pdrtpy.plot.ExcitationPlot` can be used to view the results.

        :param measurements: Input :math:`H_2` measurements to be fit.
        :type measurements: list of :class:`~pdrtpy.measurement.Measurement`.
        """
        super().__init__(mol.H2(), measurements)


class COExcitationFit(BaseExcitationFit):
    def __init__(self, measurements: Measurement = None):
        r"""Tool for fitting temperatures, column densities, `A_v` from an :math:`^{12}C^{16}O` excitation diagram. It takes as input a set of :math:`H_2` rovibrational line observations with errors represented as :class:`~pdrtpy.measurement.Measurement`.


        Often, excitation diagrams show evidence of both "hot" and "cold" gas components, where the cold gas
        dominates the intensity in the low `J` transitions and the hot gas dominates in the high `J` transitions.
        Given data over several transitions, one can fit for :math:`T_{cold}, T_{hot}, N_{total} = N_{cold}+ N_{hot}`,
        and optionally :math:`A_v`. One needs at least 5 points to fit two temperatures and column
        densities (slope and intercept :math:`\times 2`), though one could compute (not fit) them with only 4 points.
        To additionally fit :math:`A_v`, one should have 6 points (5 degrees of freedom).

        Once the fit is done, :class:`~pdrtpy.plot.ExcitationPlot` can be used to view the results.

        :param measurements: Input :math:`^{12}CO` measurements to be fit.
        :type measurements: list of :class:`~pdrtpy.measurement.Measurement`.
        """
        super().__init__(mol.CO(), measurements)


class C13OExcitationFit(BaseExcitationFit):
    def __init__(self, measurements: Measurement = None):
        r"""Tool for fitting temperatures, column densities, `A_v` from an :math:`^{13}C^{16}O` excitation diagram. It takes as input a set of :math:`H_2` rovibrational line observations with errors represented as :class:`~pdrtpy.measurement.Measurement`.

        Often, excitation diagrams show evidence of both "hot" and "cold" gas components, where the cold gas
        dominates the intensity in the low :math:`J` transitions and the hot gas dominates in the high :math:`J` transitions.
        Given data over several transitions, one can fit for :math:`T_{cold}, T_{hot}, N_{total} = N_{cold}+ N_{hot}`,
        and optionally :math:`A_v`. One needs at least 5 points to fit two temperatures and column
        densities (slope and intercept :math:`\times 2`), though one could compute (not fit) them with only 4 points.

        Once the fit is done, :class:`~pdrtpy.plot.ExcitationPlot` can be used to view the results.

        :param measurements: Input :math:`^{13}CO` measurements to be fit.
        :type measurements: list of :class:`~pdrtpy.measurement.Measurement`.
        """
        super().__init__(mol.C13O(), measurements)


class CHplusExcitationFit(BaseExcitationFit):
    def __init__(self, measurements: Measurement = None):
        r"""Tool for fitting temperatures, column densities, `A_v`, and ortho-to-para ratio(`OPR`) from an :math:`CH^{+}` excitation diagram. It takes as input a set of :math:`CH^{+}` rovibrational line observations with errors represented as :class:`~pdrtpy.measurement.Measurement`.

        Often, excitation diagrams show evidence of both "hot" and "cold" gas components, where the cold gas dominates the intensity in the low :math:`J` transitions and the hot gas dominates in the high :math:`J` transitions. Given data over several transitions, one can fit for :math:`T_{cold}, T_{hot}, N_{total} = N_{cold}+ N_{hot}`. One needs at least 5 points to fit the temperatures and column densities (slope and intercept :math:`\times 2`), though one could compute (not fit) them with only 4 points.

        Once the fit is done, :class:`~pdrtpy.plot.ExcitationPlot` can be used to view the results.

        :param measurements: Input :math:`CH^{+}` measurements to be fit.
        :type measurements: list of :class:`~pdrtpy.measurement.Measurement`.
        """
        super().__init__(mol.CHplus(), measurements)
