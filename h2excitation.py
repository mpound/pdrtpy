
from astropy.table import Table
import astropy.units as u
import astropy.constants as constants
from measurement import Measurement
import math
import matplotlib.pyplot as plt
import numpy as np

from tool import Tool
import pdrutils as utils

class H2Excitation(Tool):
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
            if not utils.check_units(mm.unit,self._intensity_units):
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
        if not utils.check_units(m.unit,self._intensity_units):
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
        if not utils.check_units(N_upper.unit,self._cd_units):
          print("##### Warning: colden did not come out in correct units ("+N_upper.unit.to_string()+")")
        return N_upper

    def _compute_column_densities(self):
        '''Compute all column densities for stored intensity measurements'''
        for m in self._measurements:
            self._column_density[m] = self.colden(self._measurements[m])

    def average_column_density(self,norm,x,y,xsize,ysize,line):
        cdnorm = self.column_densities(norm=norm)
        cdunit = cdnorm[utils.firstkey(cdnorm)].unit
        cdavg = dict()
        for cd in cdnorm:
            weights = cdnorm[cd].uncertainty.array[y:y+ysize,x:x+xsize]
            if line:
                index = cd
            else:
                index = self._ac.loc[cd]["J_u"]
            cdavg[index] = np.average(a=cdnorm[cd].data[y:y+ysize,x:x+xsize],weights=weights)
        return cdavg

    def plot_intensities(self,**kwargs):
        pass
    def plot_column_densities(self,**kwargs):
        pass

    def excitation_diagram(self,**kwargs):
        pass

    def fit_excitation(self,**kwargs):
        pass

