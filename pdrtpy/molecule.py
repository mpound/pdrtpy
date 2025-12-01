import warnings
from pathlib import Path
from typing import Union

import astropy.units as u
import numpy as np
from astropy.table import Table  # , QTable
from astropy.units.quantity import Quantity

from . import pdrutils as utils


class BaseMolecule(object):
    def __init__(self, name: str, path: Union[Path, str], opr: float = 1, opr_can_vary: bool = False, **kwargs) -> None:
        """

        The base class for holding molecular transition data.

        Parameters
        ----------
        name : str
            The descriptive name of the molecule, which will be used in displays. Can include LaTeX (without $ signs) e.g. 'H_2', 'CO', '^{13}CO'.
        path : Union[Path, str]
            Path to the table of molecular transition data.
        opr : float
            The canonical ortho-to-para ratio for molecules containing hydrogen.  Default 1.
        opr_can_vary : bool
            Can the ortho-to-para ratio vary in this molecule under different physical conditions? Default:False
        **kwargs: dict
            Additional keywords to pass to Table.read
        """
        self._constants_file = path
        self._canonical_opr = opr
        self._opr_can_vary = opr_can_vary
        self._name = name
        self._transition_data = utils.get_table(path, **kwargs)  # @todo QTable
        self._transition_data.add_index("Line")
        self._transition_data.add_index("Ju")

    def partition_function(self, temperature: Quantity) -> np.ndarray:
        r"""
        Calculate the partition function given an excitation temperature(s)

        The default calculation is

        :math:`Q(T) = \sum g_j~e^{-E_j/kT}`

        where :math:`g_j` is the statistical weight, :math:`E_j` is the transition energy, :math:`T` is the
        excitation temperature, and :math:`k` is Boltzmann's constant.

        Sub-classes should override this method for custom (and likely more accurate) partition function calculation.

        Parameters
        ----------
        temperature :  :class:`astropy.units.quantity.Quantity`
            The excitation temperature(s)

        Returns
        -------
        :class:`~numpy.ndarray`
            The partition function evaluated at the given excitation temperatures

        """

        gu = self._transition_data["gu"]
        eu = self._transition_data["Tu"]
        return np.array([np.sum(gu * np.exp(-eu / t)) for t in temperature])

    @property
    def name(self):
        """
        The descriptive name of the molecule

        Returns
        -------
        str
            Descriptive name of the moelcule

        """
        return self._name

    @property
    def canonical_opr(self) -> float:
        """
        The canonical ortho-to-para ratio of the molecule.  Only used for molecules containing hydrogen (where canonical OPR=3).

        Returns
        -------
        float
            The ortho-to-para ratio

        """
        return self._canonical_opr

    @property
    def transition_data(self) -> Table:
        """
        The table fo transition data, containing columns such as transition energy, statistical weight, Einstein A coefficient, wavelength etc.


        Returns
        -------
        `~astropy.table.Table`
            The table of transition data,
        """
        return self._transition_data

    @property
    def opr_can_vary(self):
        """
        Does the ortho-to-para ratio vary in this molecule under different physical conditions?

        Returns
        -------
        bool
            True if OPR can vary, False otherwise.

        """
        return self._opr_can_vary

    @property
    def line_ids(self) -> list:
        """
        The identifiers (names) of the spectral line transitions in the data.  These should be used to identify your Measurement data for excitation fits.

        Returns
        -------
        list
            The spectral line IDs
        """
        return list(self._transition_data["Line"])

    @property
    def line_wavelengths(self) -> Quantity:
        """
        The wavelengths of the spectral line transitions in the data.  These should be used to identify your Measurement data for excitation fits.

        Returns
        -------
        `~astropy.units.quantity.Quantity`

            The spectral line wavelengths

        """
        return Quantity(self._transition_data["lambda"])


class H2(BaseMolecule):
    def __init__(self, name="H_2", path="RoueffEtAl.tab", opr=3.0, opr_can_vary=True):
        """Molecular hydrogen. This uses the `H_2` line calculations from Roueff et al 2019, A&A, 630, 58, Table 2."""
        super().__init__(name=name, path=path, opr=opr, opr_can_vary=True, format="ascii.ipac")

    def partition_function(self, temperature: Quantity) -> np.ndarray:
        r"""
        Calculate the :math:`H_2` partition function given an excitation temperature(s).   This function uses the
        expression from Herbst et al 1996 http://articles.adsabs.harvard.edu/pdf/1996AJ....111.2403H

        :math:`Q(T) = 0.247 ~T / [1 - exp(6000 / T)]`

        where :math:`T` is the excitation temperature.

        Parameters
        ----------
        temperature  :class:`astropy.units.quantity.Quantity`
            The excitation temperature(s) at which to evaluate the partition function

        Returns
        -------
        Q: :class:`~numpy.ndarray`
            The partition function evaluated at the given excitation temperature(s)
        """
        # See Herbst et al 1996
        # http://articles.adsabs.harvard.edu/pdf/1996AJ....111.2403H
        # Z(T) =  = 0.0247T * [1 - \exp(-6000/T)]^-1

        # This is just being defensive.  I know the temperatures used internally are in K.
        t = np.ma.masked_invalid(
            (temperature.value * u.Unit(temperature.unit)).to("K", equivalencies=u.temperature()).value
        )
        t.mask = np.logical_or(t.mask, np.logical_not(np.isfinite(t)))
        Q = 0.0247 * t / (1.0 - np.exp(-6000.0 / t))
        return Q


class CO(BaseMolecule):  # 12C16O
    def __init__(self, name="^{12}CO", path="12co_transition.tab.gz", opr=1.0, opr_can_vary=False):
        r"""Carbon Monoxide  isotopologue :math:`^{12}C^{16}O`. This uses the molecular Lines and Levels data from the Meudon PDR7 code."""
        super().__init__(name, path, opr, opr_can_vary, format="ascii.ecsv")
        self._partfun_data = utils.get_table("PartFun_12C16O.tab", format="ascii.ecsv")
        self._maxQtemp = np.max(self._partfun_data["T"])

    def partition_function(self, temperature: Quantity) -> np.ndarray:
        """
        Calculate the partition function for CO at the given temperature usin the HITRAN partition function.
        https://hitran.org/data/Q/q26.txt
        The HITRAN function is evaluated at 1K intervals; this function performas a linear interpolation on those data.

        Parameters
        ----------
        temperature  :class:`astropy.units.quantity.Quantity`
            The excitation temperature(s) at which to evaluate the partition function

        Returns
        -------
        Q: :class:`~numpy.ndarray`
            The partition function evaluated at the given excitation temperature(s)
        """
        t = np.ma.masked_invalid(
            (temperature.value * u.Unit(temperature.unit)).to("K", equivalencies=u.temperature()).value
        )
        if np.nanmax(t) > self._maxQtemp:
            warnings.warn(f"Input temperature exceeds maximum partition function temperature: {self._maxQtemp} K")
        return np.interp(t, self._partfun_data["T"], self._partfun_data["Q"])


class C13O(BaseMolecule):  # 13CO16O
    def __init__(self, name="^{13}CO", path="13co_transition.tab", opr=1.0, opr_can_vary=False):
        """Carbon Monoxide isotopologue  :math:`^{13}C^{16}O`. This uses the molecular Lines and Levels data from the Meudon PDR7 code."""
        super().__init__(name, path, opr, opr_can_vary, format="ascii.ecsv")
        self._partfun_data = utils.get_table("PartFun_13C16O.tab", format="ascii.ecsv")
        self._maxQtemp = np.max(self._partfun_data["T"])

    # @todo refactor partition_function since we use same table format for all
    def partition_function(self, temperature: Quantity) -> np.ndarray:
        """Calculate the partition function for 13CO at the given temperature usin the HITRAN partition function.
        https://hitran.org/data/Q/q27.txt
        The HITRAN function is evaluated at 1K intervals; this function performas a linear interpolation on those data.

        Parameters
        ----------
        temperature  :class:`astropy.units.quantity.Quantity`
            The excitation temperature(s) at which to evaluate the partition function

        Returns
        -------
        Q: :class:`~numpy.ndarray`
            The partition function evaluated at the given excitation temperature(s)
        """
        t = np.ma.masked_invalid(
            (temperature.value * u.Unit(temperature.unit)).to("K", equivalencies=u.temperature()).value
        )
        if np.nanmax(t) > self._maxQtemp:
            warnings.warn(f"Input temperature exceeds maximum partition function temperature: {self._maxQtemp} K")
        return np.interp(t, self._partfun_data["T"], self._partfun_data["Q"])


# class CHplus(BaseMolecule):  # CH+
#    def __init__(self, name="CH^+", path="CHp_transition.tab", opr=3.0, opr_can_vary=True):
#        super().__init__(name, path, opr, opr_can_vary)
#       self._partfun_data = utils.get_table("PartFun_CH+.tab", format="ascii.ecsv")
#       self._maxQtemp = np.max(self._partfun_data["T"])
