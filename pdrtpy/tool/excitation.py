import cProfile
import io
import math
import pstats
import warnings
from abc import abstractmethod
from copy import deepcopy

import astropy.constants as constants
import astropy.units as u

from astropy.units.quantity import Quantity
import numpy as np
from astropy import log
from astropy.nddata import Cutout2D, StdDevUncertainty
from emcee.pbar import get_progress_bar
from lmfit import Parameters  # , fit_report
from lmfit.model import Model  # , ModelResult

# from scipy.interpolate import interp1d
from typing import Union

from .. import pdrutils as utils
from ..measurement import Measurement
from .. import molecule as mol
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

    def __init__(self, molecule: mol.BaseMolecule, measurements: Union[dict, Measurement] = None):

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
        print(f"{self._canonical_opr=}")
        self._opr = Measurement(data=[self._canonical_opr], uncertainty=None)
        self._residual_functions = {
            1: self._one_component_residual,
            2: self._two_component_residual,
        }
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
        """Initialize measurements dictionary given a list.

        :param m: list of intensity :class:`~pdrtpy.measurement.Measurement`s in units equivalent to :math:`{\\rm erg~cm^{-2}~s^{-1}~sr^{-1}}`
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
            log.warning(f"The molecule {self.molecule.name} does not have a variable OPR.")
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
        """Add an intensity Measurement to internal dictionary used to
        compute the excitation diagram.   This method can also be used
        to safely replace an existing intensity Measurement.

        :param m: A :class:`~pdrtpy.measurement.Measurement` instance containing intensity in units equivalent to :math:`{\\rm erg~cm^{-2}~s^{-1}~sr^{-1}}`
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
        """Safely replace an existing intensity Measurement.  Do not
        change a Measurement in place, use this method.
        Otherwise, the column densities will be inconsistent.

        :param m: A :class:`~pdrtpy.measurement.Measurement` instance containing intensity in units equivalent to :math:`{\\rm erg~cm^{-2}~s^{-1}~sr^{-1}}`
        """
        self.add_measurement(m)

    def set_extinction_model(self, model):
        r"""
        Set the model to be used for fitting visual extinction, $A_v$.  This is typically a model
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
            "profile": False,
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
        self._params.add("m1", value=0, min=-1, max=0)
        self._params.add("n1", value=15, min=10, max=30)
        if self._numcomponents == 2:
            self._params.add("m2", value=0, min=-1, max=0)
            self._params.add("n2", value=15, min=10, max=30)
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
        pp = self._model.make_params()
        # pp.pretty_print()

    def _one_component_model(
        self,
        x,
        m1,
        n1,
        opr,
        av,
        idx=[],
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
        idx = [int(i) for i in idx]
        # model is already in log space
        model = x * m1 + n1
        if fit_opr:
            # model[idx] *= opr/self._canonical_opr
            # print("Adding")
            model[idx] += np.log10(opr / self._canonical_opr)
        if fit_av:
            model = model - 0.4 * extinction_ratio * av * utils.LOGE
        return model

    def _one_component_residual(self, params, x, data, error, idx):
        p = params.valuesdict()
        model = x * p["m1"] + p["n1"]
        if params["opr"].vary:
            model *= p["opr"] / self._canonical_opr
        return (model - data) / error

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
        idx=[],
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
        # why are these coming in as floats?
        idx = [int(i) for i in idx]
        y1 = 10 ** (x * m1 + n1)
        y2 = 10 ** (x * m2 + n2)

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

    def _two_component_residual(self, params, x, data, error, idx):
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
        #
        # @TODO add extinction correction. however this method is not currently used.
        p = params.valuesdict()
        y1 = 10 ** (x * p["m1"] + p["n1"])
        y2 = 10 ** (x * p["m2"] + p["n2"])
        model = np.log10(y1 + y2)
        if params["opr"].vary:
            model += np.log10(p["opr"] / self._canonical_opr)
        return (model - data) / error

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
        The extinction law used when fitting for visual extinction, $A_v$.

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
        # print(f'returning {cl}')
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
        :param unit: The units in which to return the column density. Default: :math:`{\\rm }cm^{-2}`
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
                # print("CD ",cd,"DENOM ",denom)
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
        clip=-1e40 * u.Unit("cm-2"),
    ):
        r"""Compute the average column density over a spatial box.  The box is created using :class:`astropy.nddata.utils.Cutout2D`.

        :param position: The position of the cutout array's center with respect to the data array. The position can be specified either as a `(x, y)` tuple of pixel coordinates.
        :type position: tuple
        :param size: The size of the cutout array along each axis. If size is a scalar number or a scalar :class:`~astropy.units.Quantity`, then a square cutout of size will be created. If `size` has two elements, they should be in `(nx,ny)` order [*this is the opposite of Cutout2D signature*]. Scalar numbers in size are assumed to be in units of pixels.  Default value of None means use all pixels (position is ignored)
        :type size: int, array_like`
        :param norm: if True, normalize the column densities by the
                       statistical weight of the upper state, :math:`g_u`.  For ortho-$H_2$ $g_u = OPR \times (2J+1)$, for para-$H_2$ $g_u=2J+1$. In LTE, $OPR = 3$.
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
            # print(f"weights {weights} avg,sum:{np.average(weights)},{np.sum(weights)}")
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
        """Upper state energies of stored intensities, in K.

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
        """Wavelengths of transitions, in micron (assumed unit using Roueff et al table)

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
        r"""Get the upper state statistical weight $g_u$ for the given transition identifer, and, if the transition is odd-$J$, scale the result by the given ortho-to-para ratio.  If the transition is even-$J$, the LTE value is returned.

        :param id: the measurement identifier
        :type id: str
        :param opr:
        :type opr: float
        :raises KeyError: if id not in existing Measurements
        :rtype: float
        """
        if not self.molecule.opr_can_vary:
            log.warning(f"The molecule {self.molecule,name} does not have a variable OPR.")
            return self._molecule._transition_data.loc[id]["gu"]
        if utils.is_even(self._molecule._transition_data.loc[id]["Ju"]):
            return self._molecule._transition_data.loc[id]["gu"]
        else:
            # print("Ju=%d scaling by [%.2f/%.2f]=%.2f"%(self._molecule._transition_data.loc[id]["Ju"],opr,self._canonical_opr,opr/self._canonical_opr))
            return self._molecule._transition_data.loc[id]["gu"] * opr / self._canonical_opr

    def intensity(self, colden):
        """Given an upper state column density :math:`N_u`, compute the intensity :math:`I`.

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
        """Compute the column density in upper state :math:`N_u`, given an
        intensity :math:`I` and assuming optically thin emission.
        Units of :math:`I` need to be equivalent to
        :math:`{\\rm erg~cm^{-2}~s^{-1}~sr^{-1}}`.

        .. math::
              I &= {A \Delta E~N_u \over 4\pi}

              N_u &= 4\pi {I\over A\Delta E}

        where :math:`A` is the Einstein A coefficient and :math:`\Delta E` is the energy of the transition.

        :param intensity: A :class:`~pdrtpy.measurement.Measurement` instance containing intensity in units equivalent to :math:`{\\rm erg~cm^{-2}~s^{-1}~sr^{-1}}`
        :type intensity: :class:`~pdrtpy.measurement.Measurement`
        :param unit: The units in which to return the column density. Default: :math:`{\\rm }cm^{-2}`
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
        :param unit: The units in which to return the column density. Default: :math:`{\\rm }cm^{-2}`
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
    def _compute_quantities(self, fitmap):
        """Compute the temperatures and column densities for the hot and cold gas components.  This method will set class variables `_temperature` and `_colden`.

        :param params: The fit parameters returned from fit_excitation.
        :type params: :class:`lmfit.Parameters`
        """
        self._temperature = dict()
        # N(J=0) column density = intercept on y axis
        self._j0_colden = dict()
        # total column density = N(J=0)*Z(T) where Z(T) is partition function
        self._total_colden = dict()
        size = fitmap.data.size
        if self._numcomponents == 2:
            # create default arrays in which calculated values will be stored.
            # Use nan as fill value because there may be nans in fitmapdata, in which
            # case nothing need be done to arrays.
            # tc, th = cold and hot temperatures
            # utc, utc = uncertainties in cold and hot temperatures
            # nc, nh = cold and hot column densities
            # unc, unh = uncertainties in cold and hot temperatures
            # opr = ortho to para ratio
            # uopr = uncertainty in OPR
            tc = np.full(shape=size, fill_value=np.nan, dtype=float)
            utc = np.full(shape=size, fill_value=np.nan, dtype=float)
            nc = np.full(shape=size, fill_value=np.nan, dtype=float)
            unc = np.full(shape=size, fill_value=np.nan, dtype=float)
            opr = np.full(shape=size, fill_value=np.nan, dtype=float)
            uopr = np.full(shape=size, fill_value=np.nan, dtype=float)
            av = np.full(shape=size, fill_value=np.nan, dtype=float)
            uav = np.full(shape=size, fill_value=np.nan, dtype=float)

            th = np.full(shape=size, fill_value=np.nan, dtype=float)
            uth = np.full(shape=size, fill_value=np.nan, dtype=float)
            unh = np.full(shape=size, fill_value=np.nan, dtype=float)
            nh = np.full(shape=size, fill_value=np.nan, dtype=float)
            ff = fitmap.data.flatten()
            ffmask = fitmap.mask.flatten()
            for i in range(size):
                if ffmask[i]:
                    continue
                params = ff[i].params
                for p in params:
                    if params[p].stderr is None and params[p].vary:
                        params.pretty_print()
                        raise Exception(
                            "Something went wrong with the fit and it was unable to calculate errors on the fitted"
                            f" parameter {p}. It's likely that a two-temperature model is not appropriate for your"
                            f" data.Check the fit_result report and plot. At pixel {i} with mask {ffmask[i]}"
                        )
                if params["m2"] < params["m1"]:
                    cold = "2"
                    hot = "1"
                else:
                    cold = "1"
                    hot = "2"
                mcold = "m" + cold
                mhot = "m" + hot
                ncold = "n" + cold
                nhot = "n" + hot
                # cold and hot temperatures
                utc[i] = params[mcold].stderr / params[mcold]
                tc[i] = -utils.LOGE / params[mcold]
                uth[i] = params[mhot].stderr / params[mhot]
                th[i] = -utils.LOGE / params[mhot]
                nc[i] = 10 ** params[ncold]
                unc[i] = utils.LN10 * params[ncold].stderr * nc[i]
                nh[i] = 10 ** params[nhot]
                unh[i] = utils.LN10 * params[nhot].stderr * nh[i]
                opr[i] = params["opr"].value
                uopr[i] = params["opr"].stderr
                av[i] = params["av"].value
                uav[i] = params["av"].stderr

            # now reshape them all back to map shape
            tc = tc.reshape(fitmap.data.shape)
            th = th.reshape(fitmap.data.shape)
            utc = utc.reshape(fitmap.data.shape)
            uth = uth.reshape(fitmap.data.shape)
            nc = nc.reshape(fitmap.data.shape)
            nh = nh.reshape(fitmap.data.shape)
            unh = unh.reshape(fitmap.data.shape)
            unc = unc.reshape(fitmap.data.shape)
            opr = opr.reshape(fitmap.data.shape)
            uopr = uopr.reshape(fitmap.data.shape)
            av = av.reshape(fitmap.data.shape)
            uav = uav.reshape(fitmap.data.shape)

            mask = fitmap.mask | np.logical_not(np.isfinite(tc))
            ucc = StdDevUncertainty(np.abs(tc * utc))
            self._temperature["cold"] = Measurement(
                data=tc, unit=self._t_units, uncertainty=ucc, wcs=fitmap.wcs, mask=mask
            )
            mask = fitmap.mask | np.logical_not(np.isfinite(th))
            uch = StdDevUncertainty(np.abs(th * uth))
            self._temperature["hot"] = Measurement(
                data=th, unit=self._t_units, uncertainty=uch, wcs=fitmap.wcs, mask=mask
            )
            # cold and hot total column density
            ucn = StdDevUncertainty(np.abs(unc))
            mask = fitmap.mask | np.logical_not(np.isfinite(nc))
            self._j0_colden["cold"] = Measurement(nc, unit=self._cd_units, uncertainty=ucn, wcs=fitmap.wcs, mask=mask)
            mask = fitmap.mask | np.logical_not(np.isfinite(nh))
            uhn = StdDevUncertainty(np.abs(unh))
            self._j0_colden["hot"] = Measurement(nh, unit=self._cd_units, uncertainty=uhn, wcs=fitmap.wcs, mask=mask)
            #
            self._total_colden["cold"] = self._j0_colden["cold"] * self.molecule.partition_function(self.tcold)
            self._total_colden["hot"] = self._j0_colden["hot"] * self.molecule.partition_function(self.thot)
            mask = fitmap.mask | np.logical_not(np.isfinite(opr))
            self._opr = Measurement(
                opr,
                unit=u.dimensionless_unscaled,
                uncertainty=StdDevUncertainty(uopr),
                wcs=fitmap.wcs,
                mask=mask,
            )
            self._av = Measurement(
                av,
                unit=u.dimensionless_unscaled,
                uncertainty=StdDevUncertainty(uav),
                wcs=fitmap.wcs,
                mask=mask,
            )
        elif self._numcomponents == 1:
            tc = np.full(shape=size, fill_value=np.nan, dtype=float)
            utc = np.full(shape=size, fill_value=np.nan, dtype=float)
            nc = np.full(shape=size, fill_value=np.nan, dtype=float)
            unc = np.full(shape=size, fill_value=np.nan, dtype=float)
            opr = np.full(shape=size, fill_value=np.nan, dtype=float)
            uopr = np.full(shape=size, fill_value=np.nan, dtype=float)
            av = np.full(shape=size, fill_value=np.nan, dtype=float)
            uav = np.full(shape=size, fill_value=np.nan, dtype=float)
            ff = fitmap.data.flatten()
            ffmask = fitmap.mask.flatten()
            for i in range(size):
                if ffmask[i]:
                    continue
                params = ff[i].params
                for p in params:
                    if params[p].stderr is None:
                        params.pretty_print()
                        raise Exception(
                            "Something went wrong with the fit and it was unable to calculate errors on the fitted"
                            f" parameter {p}. It's likely that a two-temperature model is not appropriate for your"
                            f" data. Check the fit_result report and plot. At pixel {i} with mask {ffmask[i]}."
                        )
                mcold = "m1"
                ncold = "n1"
                # cold and hot temperatures
                utc[i] = params[mcold].stderr / params[mcold]
                tc[i] = -utils.LOGE / params[mcold]

                nc[i] = 10 ** params[ncold]
                unc[i] = utils.LN10 * params[ncold].stderr * nc[i]
                opr[i] = params["opr"].value
                uopr[i] = params["opr"].stderr
                av[i] = params["av"].value
                uav[i] = params["av"].stderr

            # now reshape them all back to map shape
            tc = tc.reshape(fitmap.data.shape)
            utc = utc.reshape(fitmap.data.shape)
            nc = nc.reshape(fitmap.data.shape)
            unc = unc.reshape(fitmap.data.shape)
            opr = opr.reshape(fitmap.data.shape)
            uopr = uopr.reshape(fitmap.data.shape)
            av = av.reshape(fitmap.data.shape)
            uav = uav.reshape(fitmap.data.shape)

            mask = fitmap.mask | np.logical_not(np.isfinite(tc))
            ucc = StdDevUncertainty(np.abs(tc * utc))
            self._temperature["cold"] = Measurement(
                data=tc, unit=self._t_units, uncertainty=ucc, wcs=fitmap.wcs, mask=mask
            )
            self._temperature["hot"] = self._temperature["cold"]
            # cold = hot total column density
            ucn = StdDevUncertainty(np.abs(unc))
            mask = fitmap.mask | np.logical_not(np.isfinite(nc))
            self._j0_colden["cold"] = Measurement(nc, unit=self._cd_units, uncertainty=ucn, wcs=fitmap.wcs, mask=mask)
            self._j0_colden["hot"] = self._j0_colden["cold"]
            self._total_colden["cold"] = self._j0_colden["cold"] * self.molecule.partition_function(self.tcold)
            self._total_colden["hot"] = self._total_colden["cold"]
            mask = fitmap.mask | np.logical_not(np.isfinite(opr))
            self._opr = Measurement(
                opr,
                unit=u.dimensionless_unscaled,
                uncertainty=StdDevUncertainty(uopr),
                wcs=fitmap.wcs,
                mask=mask,
            )
            self._av = Measurement(
                av,
                unit=u.dimensionless_unscaled,
                uncertainty=StdDevUncertainty(uav),
                wcs=fitmap.wcs,
                mask=mask,
            )
        else:
            raise Exception(f"Bad numcomponents: {self._numcomponents}")

    def _first_guess(self, x, y):
        r"""The first guess at the fit parameters is done by finding the line between the first two (lowest energy) points to determine $T_{cold}and between the last two (highest energy) points to determine $T_{hot}. The first guess is needed to ensure the final fit converges.  The guess doesn't need to be perfect, just in the ballpark.

        :param x: array of energies, $E/k$
        :type x: numpy array
        :param y: array of normalized column densities $N_u/g_u$
        :type y: numpy array
        """
        slopecold = (y[1] - y[0]) / (x[1] - x[0])
        intcold = y[1] - slopecold * x[1]
        if self._numcomponents == 2:
            slopehot = (y[-1] - y[-2]) / (x[-1] - x[-2])
            inthot = y[-1] - slopehot * x[-1]
        else:
            slopehot = slopecold
            inthot = intcold
        # print("FG ",type(slopecold),type(slopehot),type(intcold),type(inthot))
        return np.array([slopecold, intcold, slopehot, inthot])

    def _fit_excitation(self, position, size, fit_opr=False, fit_av=False, **kwargs):
        """Fit the :math:`log N_u-E` diagram with two excitation temperatures,
           a ``hot`` :math:`T_{ex}` and a ``cold`` :math:`T_{ex}`.  A first
           pass guess is initially made using data partitioning and two
           linear fits.

           If ``position`` and ``size`` are given, the data will be averaged over a spatial box before fitting.  The box is created using :class:`astropy.nddata.utils.Cutout2D`.  If position or size is None, the data are averaged over all pixels.  If the Measurements are single values, these arguments are ignored.

           :param position: The position of the cutout array's center with respect to the data array. The position can be specified either as a `(x, y)` tuple of pixel coordinates or a :class:`~astropy.coordinates.SkyCoord`, which will use the :class:`~astropy.wcs.WCS` of the ::class:`~pdrtpy.measurement.Measurement`s added to this tool. See :class:`~astropy.nddata.utils.Cutout2D`.
           :type position: tuple or :class:`astropy.coordinates.SkyCoord`
           :param size: The size of the cutout array along each axis. If size is a scalar number or a scalar :class:`~astropy.units.Quantity`, then a square cutout of size will be created. If `size` has two elements, they should be in `(ny, nx)` order. Scalar numbers in size are assumed to be in units of pixels. `size` can also be a :class:`~astropy.units.Quantity` object or contain :class:`~astropy.units.Quantity` objects. Such :class:`~astropy.units.Quantity` objects must be in pixel or angular units. For all cases, size will be converted to an integer number of pixels, rounding the the nearest integer. See the mode keyword for additional details on the final cutout size. Default value of None means use all pixels (position is ignored)
           :type size: int, array_like, or :class:`astropy.units.Quantity`
           :param fit_opr: Whether to fit the ortho-to-para ratio or not. If True, the OPR will be varied to determine the best value. If False, the OPR is fixed at the canonical LTE value of 3.
           :type fit_opr: bool
        ,dtype=object   :returns: The fit result which contains slopes, intercepts, the ortho to para ratio (OPR), and fit statistics
           :rtype:  :class:`lmfit.model.ModelResult`
        """
        profile = kwargs.pop("profile", None)
        # fit_av = kwargs.pop("fit_av")
        init_av = kwargs.pop("init_av", 0.0)
        init_opr = kwargs.pop("init_opr", 3.0)
        verbose = kwargs.pop("verbose")
        self._stats = None
        if profile:
            pr = cProfile.Profile()
            pr.enable()
        min_points = self._numcomponents * 2
        if fit_opr:
            min_points += 1
        else:
            self._opr = Measurement(data=[self._canonical_opr], uncertainty=None)
        if fit_av:
            min_points += 1
            wavelengths = list(self.wavelengths(line=True).values()) * u.micron  # assume micron for now
            extinction_ratios = self.extinction_model(wavelengths)  # A_lambda/A_v at the wavelengths of the transitions
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
        _ee = np.array([c for c in energy.values()])
        # @ todo: allow fitting of one-temperature model
        if len(_ee) < min_points:
            raise Exception(
                f"You need at least {min_points:d} data points to determine {self._numcomponents}-temperature model"
            )
        if len(_ee) == min_points:
            warnings.warn(
                f"Number of data points is equal to number of free parameters ({min_points:d}). Fit will be"
                " over-constrained"
            )
        _energy = Measurement(_ee, unit="K")
        _ids = list(energy.keys())
        idx = self._get_ortho_indices(_ids)
        # Get Nu/gu.  Canonical opr will be used.
        if position is None or size is None:
            colden = self.column_densities(norm=True, line=True)
        else:
            colden = self.average_column_density(norm=True, position=position, size=size, line=True)

        # Need to stuff the data into a single vector
        _cd = np.squeeze(np.array([c.data for c in colden.values()]))
        _er = np.squeeze(np.array([c.error for c in colden.values()]))
        _colden = Measurement(_cd, uncertainty=StdDevUncertainty(_er), unit="cm-2")
        fk = utils.firstkey(colden)
        x = _energy.data
        y = np.log10(_colden.data)
        # print("SHAPE Y LEN(SHAPE(Y) ",y.shape,len(y.shape))
        # kwargs_opts = {"guess": self._first_guess(x,y)}
        # kwargs_opts.update(kwargs)
        sigma = utils.LOGE * _colden.error / _colden.data
        slopecold, intcold, slopehot, inthot = self._first_guess(x, y)
        # print(slopecold,slopehot,intcold,inthot)
        tcold = -utils.LOGE / slopecold
        thot = -utils.LOGE / slopehot
        if np.shape(tcold) == ():
            tcold = np.array([tcold])
            thot = np.array([thot])
        saveshape = tcold.shape
        # print("TYPE COLD SIT",type(slopecold),type(intcold),type(tcold))
        # print("SHAPES: colden/sigma/slope/int/temp/cd: ",np.shape(_colden),np.shape(sigma),np.shape(slopecold),np.shape(intcold),np.shape(tcold),np.shape(_cd))
        # print("First guess at excitation temperatures:\n T_cold = %.1f K\n T_hot = %.1f K"%(tcold,thot))
        fmdata = np.empty(tcold.shape, dtype=object).flatten()
        # fm = FitMap(data=fmdata,wcs=colden[fk].wcs,uncertainty=None,unit=None)
        tcold = tcold.flatten()
        thot = thot.flatten()
        slopecold = slopecold.flatten()
        slopehot = slopehot.flatten()
        inthot = inthot.flatten()
        intcold = intcold.flatten()
        # sigma = sigma.flatten()
        # flatten any dimensions past 0
        shp = y.shape
        # print("NS ",shp[0],shp[1:])
        if len(shp) == 1:
            # print("adding new axis")
            y = y[:, np.newaxis]
            shp = y.shape
        yr = y.reshape((shp[0], np.prod(shp[1:])))
        sig = sigma.reshape((shp[0], np.prod(shp[1:])))
        # print("YR, SIG SHAPE",yr.shape,sig.shape)
        count = 0
        # print("LEN(TCOLD)",len(tcold))
        total = len(tcold)
        fm_mask = np.full(shape=tcold.shape, fill_value=False)
        # Suppress the incorrect warning about model parameters
        warnings.simplefilter("ignore", category=UserWarning)
        excount = 0
        badfit = 0
        # update whether opr is allowed to vary or not.
        self._model.set_param_hint("opr", vary=fit_opr)
        self._model.set_param_hint("av", vary=fit_av)
        # use progress bar if more than one pixel
        if total > 1:
            progress = kwargs.pop("progress", True)
        else:
            progress = False
        # print("PARAMS")
        # self._params.pretty_print()
        self._excount = 0
        self._badfit = 0
        with get_progress_bar(progress, total, leave=True, position=0) as pbar:
            for i in range(total):
                if np.isfinite(yr[:, i]).all() and np.isfinite(sig[:, i]).all():
                    # update Parameter hints based on first guess.
                    p = deepcopy(self._params)
                    p["n1"].value = intcold[i]
                    p["m1"].value = slopecold[i]
                    # self._model.set_param_hint("m1", value=slopecold[i], vary=True)
                    # self._model.set_param_hint("n1", value=intcold[i], vary=True)
                    if self._numcomponents == 2:
                        #    self._model.set_param_hint("m2", value=slopehot[i], vary=True)
                        #    self._model.set_param_hint("n2", value=inthot[i], vary=True)
                        p["n2"].value = inthot[i]
                        p["m2"].value = slopehot[i]
                    wts = 1.0 / (sig[:, i] * sig[:, i])
                    try:
                        # print("X=",x)
                        # print("Y=",yr[:i])
                        # print(f"fitting with fit_av={fit_av}")
                        if kwargs["method"] == "emcee":
                            emcee_kwargs = {k: kwargs[k] for k in ("burn", "steps", "nwalkers") if k in kwargs}
                        else:
                            emcee_kwargs = None
                        fmdata[i] = self._model.fit(
                            data=yr[:, i],
                            weights=wts,
                            x=x,
                            params=p,
                            idx=idx,
                            fit_opr=fit_opr,
                            fit_av=fit_av,
                            extinction_ratio=extinction_ratios,
                            method=kwargs["method"],
                            nan_policy=kwargs["nan_policy"],
                            fit_kws=emcee_kwargs,
                        )
                        # if fmdata[i].success and fmdata[i].errorbars:
                        if fmdata[i].success:
                            count = count + 1
                        else:
                            # print(
                            #    f"Bad fit because success {fmdata[i].success} or errorbars"
                            #    f" {fmdata[i].errorbars} was bad"
                            # )
                            fmdata[i] = None
                            fm_mask[i] = True
                            self._badfit = self._badfit + 1
                    except ValueError as v:
                        # print(f"Bad fit because {v}")
                        fmdata[i] = None
                        fm_mask[i] = True
                        self._excount = self._excount + 1
                else:
                    # print("Bad fit because NaNs in data")
                    fmdata[i] = None
                    fm_mask[i] = True
                pbar.update(1)
        # cleanup weird fits
        for ii in range(len(fmdata)):
            badstderr = False
            fmd = fmdata[ii]
            if fmd is None:
                continue
            for p in fmd.params:
                if fmd.params[p].stderr is None and fmd.params[p].vary:
                    # print(f"Fit succeeded at pixel {ii} but stderr for parameter {p} is None. Setting mask")
                    # fmdata[i].success = False
                    fm_mask[ii] = True
                    self._badfit = self._badfit + 1
                    badstderr = True
                    fmdata[ii] = None
            if badstderr:
                count = count - 1
        warnings.resetwarnings()
        fmdata = fmdata.reshape(saveshape)
        fm_mask = fm_mask.reshape(saveshape)
        self._fitresult = FitMap(fmdata, wcs=colden[fk].wcs, mask=fm_mask, name="result")
        # this will raise an exception if the fit was bad (fit errors == None)
        self._compute_quantities(self._fitresult)
        if verbose:
            print(f"fitted {count} of {slopecold.size} pixels")
            print(f"got {excount} exceptions and {badfit} bad fits")
        # if successful, set the used position and size
        self._position = position
        self._size = size
        if profile:
            pr.disable()
            s = io.StringIO()
            sortby = pstats.SortKey.CUMULATIVE
            ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
            ps.print_stats()
            self._stats = s


# ========================== END BASEEXCITATION FIT ===================================================


# ========================== DERIVED CLASSES FOR SPECIFIC MOLECULES ===================================
class H2ExcitationFit(BaseExcitationFit):
    def __init__(self, measurements: Measurement = None):
        r"""Tool for fitting temperatures, column densities, `A_v`, and ortho-to-para ratio(`OPR`) from an :math:`H_2`
        excitation diagram. It takes as input a set of :math:`H_2` rovibrational line observations with errors
        represented as :class:`~pdrtpy.measurement.Measurement`.

        Often, excitation diagrams show evidence of both "hot" and "cold" gas components, where the cold gas
        dominates the intensity in the low `J` transitions and the hot gas dominates in the high `J` transitions.
        Given data over several transitions, one can fit for :math:`T_{cold}, T_{hot}, N_{total} = N_{cold}+ N_{hot}`,
        and optionally `A_v` or `OPR`. One needs at least 5 points to fit two temperatures and column
        densities (slope and intercept :math:`\times 2`), though one could compute (not fit) them with only 4 points.
        To additionally fit `A_v` or `OPR`, one should have 6 points (5 degrees of freedom).

        Once the fit is done, :class:`~pdrtpy.plot.ExcitationPlot` can be used to view the results.

        :param measurements: Input :math:`H_2` measurements to be fit.
        :type measurements: list of :class:`~pdrtpy.measurement.Measurement`.
        """
        super().__init__(mol.H2(), measurements)


class COExcitationFit(BaseExcitationFit):
    def __init__(self, measurements: Measurement = None):
        r"""Tool for fitting temperatures, column densities, `A_v` from an :math:`^{12}CO` excitation diagram. It takes as input a set of :math:`H_2` rovibrational line observations with errors represented as :class:`~pdrtpy.measurement.Measurement`.


        Often, excitation diagrams show evidence of both "hot" and "cold" gas components, where the cold gas
        dominates the intensity in the low `J` transitions and the hot gas dominates in the high `J` transitions.
        Given data over several transitions, one can fit for :math:`T_{cold}, T_{hot}, N_{total} = N_{cold}+ N_{hot}`,
        and optionally `A_v`. One needs at least 5 points to fit two temperatures and column
        densities (slope and intercept :math:`\times 2`), though one could compute (not fit) them with only 4 points.
        To additionally fit `A_v`, one should have 6 points (5 degrees of freedom).

        Once the fit is done, :class:`~pdrtpy.plot.ExcitationPlot` can be used to view the results.

        :param measurements: Input :math:`^{12}CO` measurements to be fit.
        :type measurements: list of :class:`~pdrtpy.measurement.Measurement`.
        """
        super().__init__(mol.CO(), measurements)


class C13OExcitationFit(BaseExcitationFit):
    def __init__(self, measurements: Measurement = None):
        r"""Tool for fitting temperatures, column densities, `A_v` from an :math:`^{13}CO` excitation diagram. It takes as input a set of :math:`H_2` rovibrational line observations with errors represented as :class:`~pdrtpy.measurement.Measurement`.

        Often, excitation diagrams show evidence of both "hot" and "cold" gas components, where the cold gas
        dominates the intensity in the low `J` transitions and the hot gas dominates in the high `J` transitions.
        Given data over several transitions, one can fit for :math:`T_{cold}, T_{hot}, N_{total} = N_{cold}+ N_{hot}`,
        and optionally `A_v` or `OPR`. One needs at least 5 points to fit two temperatures and column
        densities (slope and intercept :math:`\times 2`), though one could compute (not fit) them with only 4 points.
        To additionally fit `A_v` or `OPR`, one should have 6 points (5 degrees of freedom).

        Once the fit is done, :class:`~pdrtpy.plot.ExcitationPlot` can be used to view the results.

        :param measurements: Input :math:`^{13}CO` measurements to be fit.
        :type measurements: list of :class:`~pdrtpy.measurement.Measurement`.
        """
        super().__init__(mol.C13O(), measurements)


class CHplusExcitationFit(BaseExcitationFit):
    def __init__(self, measurements: Measurement = None):
        r"""Tool for fitting temperatures, column densities, `A_v`, and ortho-to-para ratio(`OPR`) from an :math:`CH^{+}` excitation diagram. It takes as input a set of :math:`H_2` rovibrational line observations with errors represented as :class:`~pdrtpy.measurement.Measurement`.

        Often, excitation diagrams show evidence of both "hot" and "cold" gas components, where the cold gas dominates the intensity in the low `J` transitions and the hot gas dominates in the high `J` transitions. Given data over several transitions, one can fit for :math:`T_{cold}, T_{hot}, N_{total} = N_{cold}+ N_{hot}`, and optionally `OPR`. One needs at least 5 points to fit the temperatures and column densities (slope and intercept :math:`\times 2`), though one could compute (not fit) them with only 4 points. To additionally fit `OPR`, one should have 6 points (5 degrees of freedom).

        Once the fit is done, :class:`~pdrtpy.plot.ExcitationPlot` can be used to view the results.

        :param measurements: Input :math:`CH^{+}` measurements to be fit.
        :type measurements: list of :class:`~pdrtpy.measurement.Measurement`.
        """
        super().__init__(mol.CHplus(), measurements)
