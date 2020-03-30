
from astropy.table import Table
import astropy.units as u
import astropy.constants as constants
import math
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

from .tool import ToolBase
from .pdrutils import get_table, check_units, firstkey
from .measurement import Measurement

class H2Excitation(ToolBase):
    def __init__(self,measurements=None):

        # must be set before call to init_measurements
        self._intensity_units = "erg cm^-2 s^-1 sr^-1"
        self._cd_units = 'cm^-2'

        if type(measurements) == dict or measurements is None:
            self._measurements = measurements
        else:
            self._init_measurements(measurements)

        # default intensity units
        self._ac = get_table("atomic_constants.tab")
        self._ac.add_index("Line")
        self._ac.add_index("J_u")
        self._column_density = dict()

    #@property 
    def intensities(self):
        '''Return stored intensities. See `addMeasurement`'''
        return self._measurements


    def column_densities(self,norm=False):
        '''Return computed upper state column densities of stored intensities
           Parameters:
                norm - if True, normalize the column densities by the 
                       statistical weight of the upper state, *g_u*.  
                       Default: False
           Returns:
                dictionary of column densities indexed by Line name
        '''
        # Compute column densities if needed. 
        # Note: this has a gotcha - if user changes an existing intensity 
        # Measurement in place, rather than replaceMeasurement(), the colden 
        # won't get recomputed. But we warned them!
        if not self._column_density or (len(self._column_density) != len(self._measurements)):
            self._compute_column_densities()
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
        '''Return upper state energies of stored intensities, in K. 

           Parameters:
              line - if True, the dictionary index is the Line name, 
                     otherwise it is the upper state J number.  Default: False
           Returns:
              dictionary indexed by upper state J level or Line name. 
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

           Parameters:
                m - list of intensity Measurements in units
                  equivalent to (erg cm^-2 s^-1 sr^-1)
        '''
        self._measurements = dict()
        for mm in m:
            if not check_units(mm.unit,self._intensity_units):
                raise TypeError("Measurement " +mm.id + " must be in intensity units equivalent to "+self._intensity_units)
            self._measurements[mm.id] = mm
        # re-initialize column densities
        self._column_densities = dict()
  
    def addMeasurement(self,m):
        '''Add an intensity Measurement to internal dictionary used to 
           compute the excitation diagram.   This method can also be used
           to safely replace an existing intensity Measurement.

           Parameters:
              m - a Measurement instance containing intensity in units 
                  equivalent to (erg cm^-2 s^-1 sr^-1)
        '''
        if not check_units(m.unit,self._intensity_units):
            raise TypeError("Measurement " +m.id + " must be in intensity units equivalent to "+self._intensity_units)

        if self._measurements:
            self._measurements[m.id] = m
            # if there is an existing column density with this ID, remove it
            self._column_densities.pop(m.id,None)
        else:
            self._init_measurements(m)

    def replaceMeasurement(self,m):
        '''Safely replace an existing intensity Measurement.  Do not 
           change a Measurement in place, use this method. 
           Otherwise, the column densities will be inconsistent.

               Parameters:
                  m - a Measurement instance containing intensity in units 
                      equivalent to (erg cm^-2 s^-1 sr^-1)
        '''
        self.addMeasurement(self,m)


    def colden(self,intensity):
        '''Compute the column density in upper state N_upper, given an 
           intensity I and assuming optically thin emission.  
           Units of I need to be equivalent to (erg cm^-2 s^-1 sr^-1)
                 I = A * dE * N_upper/(4 pi)
                 N_upper = 4*pi*I/(A*dE)
            where A is the Einstein A coefficient and dE is the energy of the transition
            Parameters:
                intensity - a Measurement of the intensity
            Returns:
                a Measurement of the column density.
        '''
        dE = self._ac.loc[intensity.id]["dE/k"]*constants.k_B.cgs*self._ac["dE/k"].unit
        A = self._ac.loc[intensity.id]["A"]*self._ac["A"].unit
        val = Measurement(4.0*math.pi*u.sr/(A*dE))
        #print(val)
        ##print(val.value)
        #print(intensity.unit)
        #print(intensity)
        N_upper = intensity * val
        if not check_units(N_upper.unit,self._cd_units):
          print("##### Warning: colden did not come out in correct units ("+N_upper.unit.to_string()+")")
        return N_upper

    def _compute_column_densities(self):
        '''Compute all column densities for stored intensity measurements'''
        for m in self._measurements:
            self._column_density[m] = self.colden(self._measurements[m])

    def average_column_density(self,norm,x,y,xsize,ysize,line):
        cdnorm = self.column_densities(norm=norm)
        cdunit = cdnorm[firstkey(cdnorm)].unit
        cdavg = dict()
        for cd in cdnorm:
            w = cdnorm[cd].uncertainty.array[y:y+ysize,x:x+xsize]
            weights = np.ma.masked_array(w,np.isnan(w))
            #print("weights[%s]: %s"%(cd,weights))
            if line:
                index = cd
            else:
                index = self._ac.loc[cd]["J_u"]
            if norm:
                d = cdnorm[cd].data[y:y+ysize,x:x+xsize]
                data =  np.ma.masked_array(d,np.isnan(d))
                cdavg[index] = np.average(data,weights=weights)/self._ac.loc[cd]["g_u"]
                #print("dividing %s by %.1f"%(cd,self._ac.loc[cd]["g_u"]))
            else:
                cdavg[index] = np.average(a=cdnorm[cd].data[y:y+ysize,x:x+xsize],weights=weights)
             
        return cdavg

    def plot_intensities(self,**kwargs):
        pass
    def plot_column_densities(self,**kwargs):
        pass

    def excitation_diagram(self,**kwargs):
        pass

    def fit_excitation(self,**kwargs):
        # do first pass guess using data partitioning and two linear fits
        fit_param, pcov = curve_fit(two_lin, x, y,sigma=sigma)
        tcold=-np.log10(math.e)/m1
        thot=-np.log10(math.e)/m2
        print("First guess at excitation temperatures: T_cold = %.1f, T_hot = %.1f"%(tcold,thot))
        #if m1 != m2:
        #    x_intersect = (n2 - n1) / (m1 - m2)
        #    print(x_intersect)
        #else:
        #    print("did not find two linear components")

        # Now do second pass fit to full curve using first pass as initial guess
        fit_par2,pcov2 = curve_fit(x_lin,x,y,p0=fit_param,sigma=sigma)
        ma1, na1, ma2, na2 = fit_par2
        tcolda=-np.log10(math.e)/ma1
        thota=-np.log10(math.e)/ma2
        print("Fitted excitation temperatures: T_cold = %.1f, T_hot = %.1f"%(tcolda,thota))
        r = y - x_lin(x, *fit_par2)
        print("Residuals: ",np.sum(np.square(r)))
        return [fit_param, pcov, fit_par2, pcov2]

    def _two_lines(x, m1, n1, m2, n2):
        '''This function is used to partition a fit to data using two lines and 
           an inflection point.  Second slope is steeper because slopes are 
           negative in excitation diagram.

            Parameters:
               x - array of x values
               m1 - slope of first line
               n1 - intercept of first line
               m2 - slope of second line
               n2 - intercept of second line

            See https://stackoverflow.com/questions/48674558/how-to-implement-automatic-model-determination-and-two-state-model-fitting-in-py
        ''' 
        return np.max([m1 * x + n1, m2 * x + n2], axis = 0)

    def _one_line(x,m1,n1):
        '''Return a line.

          Parameters:
               x - array of x values
               m1 - slope of line
               n1 - intercept of line
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
