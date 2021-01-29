from astropy.nddata import Cutout2D
import astropy.units as u
import astropy.constants as constants
import math
import numpy as np
from scipy.optimize import curve_fit

from .toolbase import ToolBase
from .. import pdrutils as utils
from ..measurement import Measurement

class H2Excitation(ToolBase):
    """Tool for fitting temperatures to :math:`H_2` Excitation Diagrams

       **This tool is still under development**
    """
    def __init__(self,measurements=None):

        # must be set before call to init_measurements
        self._intensity_units = "erg cm^-2 s^-1 sr^-1"
        self._cd_units = 'cm^-2'

        if type(measurements) == dict or measurements is None:
            self._measurements = measurements
        else:
            self._init_measurements(measurements)

        # default intensity units
        self._ac = utils.get_table("atomic_constants.tab")
        self._ac.add_index("Line")
        self._ac.add_index("J_u")
        self._column_density = dict()

    @property 
    def intensities(self):
        '''The stored intensities. See :meth:`add_measurement`
         
           :rtype: list of :class:`~pdrtpy.measurement.Measurement`
        '''
        return self._measurements


    def column_densities(self,norm=False,unit=utils._CM2):
        '''The computed upper state column densities of stored intensities

           :param norm: if True, normalize the column densities by the 
                       statistical weight of the upper state, :math:`g_u`.  
                       Default: False
           :type norm: bool

           :returns: dictionary of column densities indexed by Line name
           :rtype: dict
        '''
        # Compute column densities if needed. 
        # Note: this has a gotcha - if user changes an existing intensity 
        # Measurement in place, rather than replaceMeasurement(), the colden 
        # won't get recomputed. But we warned them!
        if not self._column_density or (len(self._column_density) != len(self._measurements)):
            self._compute_column_densities(unit=unit)
        if norm:
            cdnorm = dict()
            for cd in self._column_density:
                # This fails with complaints about units:
                # self._column_density[cd]/self._ac.loc[cd]["g_u"]
                gu = Measurement(self._ac.loc[cd]["g_u"],unit=u.dimensionless_unscaled)
                cdnorm[cd] = self._column_density[cd]/gu
            return cdnorm
        else:
            return self._column_density

    def energies(self,line=False):
        '''Upper state energies of stored intensities, in K. 

           :param line: if True, the dictionary index is the Line name, 
                     otherwise it is the upper state :math:`J` number.  Default: False
           :type line: bool
           :returns: dictionary indexed by upper state :math:`J` number or Line name. 
           :rtype: dict
        '''
        t = dict()
        if line:
            for m in self._measurements:
                t[m] = self._ac.loc[m]["E_upper/k"]
        else:
            for m in self._measurements:
                t[self._ac.loc[m]["J_u"]] = self._ac.loc[m]["E_upper/k"]
        return t

    def _init_measurements(self,m):
        '''Initialize measurements dictionary given a list.

           :param m: list of intensity :class:`~pdrtpy.measurement.Measurement`s in units equivalent to :math:`{\\rm erg~cm^{-2}~s^{-1}~sr^{-1}}`
           :type m: list of :class:`~pdrtpy.measurement.Measurement`
        '''
        self._measurements = dict()
        for mm in m:
            if not utils.check_units(mm.unit,self._intensity_units):
                raise TypeError("Measurement " +mm.id + " units "+mm.unit.to_string()+" are not in intensity units equivalent to "+self._intensity_units)
            self._measurements[mm.id] = mm
        # re-initialize column densities
        self._column_densities = dict()
  
    def add_measurement(self,m):
        '''Add an intensity Measurement to internal dictionary used to 
           compute the excitation diagram.   This method can also be used
           to safely replace an existing intensity Measurement.

           :param m: A :class:`~pdrtpy.measurement.Measurement` instance containing intensity in units equivalent to :math:`{\\rm erg~cm^{-2}~s^{-1}~sr^{-1}}`
        '''
        if not utils.check_units(m.unit,self._intensity_units):
            raise TypeError("Measurement " +m.id + " must be in intensity units equivalent to "+self._intensity_units)

        if self._measurements:
            self._measurements[m.id] = m
            # if there is an existing column density with this ID, remove it
            self._column_densities.pop(m.id,None)
        else:
            self._init_measurements(m)

    def replace_measurement(self,m):
        '''Safely replace an existing intensity Measurement.  Do not 
           change a Measurement in place, use this method. 
           Otherwise, the column densities will be inconsistent.

           :param m: A :class:`~pdrtpy.measurement.Measurement` instance containing intensity in units equivalent to :math:`{\\rm erg~cm^{-2}~s^{-1}~sr^{-1}}`
        '''
        self.add_measurement(self,m)

    def run(self):
        cdavg = self.average_column_density(norm=True)
        energy = self.energies(line=False)
        z=np.array([np.hstack([cdavg[key],energy[key]]) for key in cdavg.keys()])
        x = Measurement(z[:,0],unit="K")
        y = Measurement(z[:1],unit=utils._CM2)

    def colden(self,intensity,unit):
        '''Compute the column density in upper state :math:`N_u`, given an 
           intensity :math:`I` and assuming optically thin emission.  
           Units of :math:`I` need to be equivalent to 
           :math:`{\\rm erg~cm^{-2}~s^{-1}~sr^{-1}}`.

           .. math::
                 I &= {A \Delta E~N_u \over 4\pi}

                 N_u &= 4\pi {I\over A\Delta E}

           where :math:`A` is the Einstein A coefficient and :math:`\Delta E` is the energy of the transition.

           :param intensity: A :class:`~pdrtpy.measurement.Measurement` instance containing intensity in units equivalent to :math:`{\\rm erg~cm^{-2}~s^{-1}~sr^{-1}}`
           :type intensity: :class:`~pdrtpy.measurement.Measurement`
           :param unit: The units in which to return the column density. Default: cm-2
           :type unit: str or :class:`astropy.unit.Unit`
           :returns: a :class:`~pdrtpy.measurement.Measurement` of the column density.
           :rtype: :class:`~pdrtpy.measurement.Measurement` 
        '''
        dE = self._ac.loc[intensity.id]["dE/k"]*constants.k_B.cgs*self._ac["dE/k"].unit
        A = self._ac.loc[intensity.id]["A"]*self._ac["A"].unit
        v = 4.0*math.pi*u.sr/(A*dE)
        val = Measurement(data=v.value,unit=v.unit)
        N_upper = intensity * val
        return N_upper.convert_unit_to(unit)

    def _compute_column_densities(self,unit):
        '''Compute all column densities for stored intensity measurements
           :param unit: The units in which to return the column density. Default: cm-2
           :type unit: str or :class:`astropy.unit.Unit`
           :returns: a :class:`~pdrtpy.measurement.Measurement` of the column density.
        '''
        for m in self._measurements:
            self._column_density[m] = self.colden(self._measurements[m],unit)


    def average_column_density(self,position,size,norm=True,unit=utils._CM2,line=False):
        r'''Compute the average column density over a spatial box.  The box is created using :class:`astropy.nddata.utils.Cutout2D`.

        :param position: The position of the cutout array's center with respect to the data array. The position can be specified either as a `(x, y)` tuple of pixel coordinates or a :class:`~astropy.coordinates.SkyCoord`, which will use the :class:`~astropy.wcs.WCS` of the ::class:`~pdrtpy.measurement.Measurement`s added to this tool. See :class:`~astropy.nddata.utils.Cutout2D`.
        :type x: tuple or :class:`astropy.coordinates.SkyCoord`
        :param size: The size of the cutout array along each axis. If size is a scalar number or a scalar :class:`~astropy.units.Quantity`, then a square cutout of size will be created. If `size` has two elements, they should be in `(ny, nx)` order. Scalar numbers in size are assumed to be in units of pixels. `size` can also be a :class:`~astropy.units.Quantity` object or contain :class:`~astropy.units.Quantity` objects. Such :class:`~astropy.units.Quantity` objects must be in pixel or angular units. For all cases, size will be converted to an integer number of pixels, rounding the the nearest integer. See the mode keyword for additional details on the final cutout size.
        :type size: int, array_like, or :class:`astropy.units.Quantity`
        :param norm: if True, normalize the column densities by the 
                       statistical weight of the upper state, :math:`g_u`.  
        :type norm: bool
        :param unit: The units in which to return the column density. Default: math:`{\rm cm}^{-2}` 
        :type unit: str or :class:`astropy.unit.Unit`
        :param line: if True, the returned dictionary index is the Line name, otherwise it is the upper state :math:`J` number.  
        :type line: bool
        :returns: dictionary of column density values, with keys as :math:`J` number or Line name
        :rtype:  dict
        '''

        cdnorm = self.column_densities(norm=norm,unit=unit)
        cdavg = dict()
        for cd in cdnorm:
            ca = cdnorm[cd]
            cutout = Cutout2D(ca.data,position,size,ca.wcs,mode='partial',fill_value=np.nan)
            w= Cutout2D(ca.uncertainty.array,position,size,ca.wcs,mode='partial',fill_value=np.nan)
            cddata = np.ma.masked_array(cutout.data,np.isnan(cutout.data))
            weights = np.ma.masked_array(w.data,np.isnan(w.data))
            if line:
                index = cd
            else:
                index = self._ac.loc[cd]["J_u"]
            cdavg[index] = np.average(cddata,weights=weights)
            if norm:
                cdavg[index] = cdavg[index] /self._ac.loc[cd]["g_u"]
                #d = cdnorm[cd].data[y:y+ysize,x:x+xsize]
                #data =  np.ma.masked_array(d,np.isnan(d))
                #print("dividing %s by %.1f"%(cd,self._ac.loc[cd]["g_u"]))
            #else:
            #    cdavg[index] = np.average(a=cdnorm[cd].data[y:y+ysize,x:x+xsize],weights=weights)
             
        return cdavg

    def plot_intensities(self,**kwargs):
        pass
    def plot_column_densities(self,**kwargs):
        pass

    def excitation_diagram(self,**kwargs):
        pass

    def fit_excitation(self,energy,colden,**kwargs):
        """Fit the :math:`log N_u-E` diagram with two excitation temperatures,
        a ``warm`` :math:`T_{ex}` and a ``cold`` :math:`T_{ex}`.  A first
        pass guess is initially made using data partitioning and two
        linear fits.
        :param energy: Eu/k
        :type energy: :class:`~pdrtpy.measurement.Measurement` containing list of values.
        :param colden: list of log(Nu/gu)
        :type colden: :class:`~pdrtpy.measurement.Measurement` containing list of values and errors

        :returns: The fit parameters as in :mod:`scipy.optimize.curve_fit`
        :rtype: list
        """
        x = energy.data
        y = np.log10(colden.data)
        sigma = np.log10(colden.error)
        fit_param, pcov = curve_fit(self._two_lines, x, y,sigma=sigma)
        m1, n1, m2, n2 = fit_param
        le = -np.log10(math.e)
        tcold=le/m1
        thot=le/m2
        print("First guess at excitation temperatures: T_cold = %.1f, T_hot = %.1f"%(tcold,thot))
        #if m1 != m2:
        #    x_intersect = (n2 - n1) / (m1 - m2)
        #    print(x_intersect)
        #else:
        #    print("did not find two linear components")

        # Now do second pass fit to full curve using first pass as initial guess
        fit_par2,pcov2 = curve_fit(x_lin,x,y,p0=fit_param,sigma=sigma)
        ma1, na1, ma2, na2 = fit_par2
        tcolda=le/ma1
        thota=le/ma2
        print("Fitted excitation temperatures: T_cold = %.1f, T_hot = %.1f"%(tcolda,thota))
        r = y - x_lin(x, *fit_par2)
        print("Residuals: ",np.sum(np.square(r)))
        return [fit_param, pcov, fit_par2, pcov2]

    def _two_lines(self,x, m1, n1, m2, n2):
        '''This function is used to partition a fit to data using two lines and 
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
        ''' 
        return np.max([m1 * x + n1, m2 * x + n2], axis = 0)

    def _one_line(x,m1,n1):
        '''Return a line.

           :param x: array of x values
           :type x: :class:`numpy.ndarray` 
           :param m1: slope of first line
           :type m1: float
           :param n1: intercept of first line
           :type n1: float
        '''
        return m1*x+n1

    #def x_exp(x,m1,n1,m2,m2):
    #    return n1*np.exp(x*m1)+n2*np.exp(x*m2)
#
    def x_lin(x, m1, n1, m2, n2):
        zz1 = 10**(x*m1+n1)
        zz2 = 10**(x*m2+n2)
        retval = np.log10(zz1+zz2)
        return retval
