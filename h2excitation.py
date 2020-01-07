
from astropy.table import Table
import astropy.units as u
import astropy.constants as constants
from measurement import Measurement
import math
import matplotlib.pyplot as plt

from tool import Tool
import pdrutils as utils

class H2Excitation(Tool):
    def __init__(self,measurements=None):
        if measurements is not None:
            if type(measurements) == dict:
                self._measurements = measurements
            else:
                self._init_measurements(measurements)
        else:
            self._measurements = None

        self._intensity_units = "erg cm^-2 s^-1 sr^-1"
        self._ac = utils.get_table("atomic_constants.tab")
        self._ac.add_index("Line")

    def _init_measurements(self,m):
        self._measurements = dict()
        for mm in m:
            self._measurements[mm.id] = mm

    def addMeasurement(self,m):
        '''Add an intensity Measurement to internal dictionary used to 
           compute the excitation diagram

           Parameters:
              m - a Measurement instance containing intensity in units 
                  equivalent to (erg cm^-2 s^-1 sr^-1)
        '''
        if self._measurements:
            self._measurements[m.id] = m
        if not utils.check_units(self._intensity_units):
            raise TypeError("Measurement must be in intensity units equivalent to "+self._intensity_units)

    def colden(self,intensity):
        '''Compute column density in upper state N_upper, given an 
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
        val = 4.0*math.pi*u.sr/(A*dE)
        #print(val)
        ##print(val.value)
        #print(intensity.unit)
        #print(intensity)
        N_upper = val*intensity
        if not utils.check_units(N_upper.unit,'cm^-2'):
          print("##### Warning: colden did not come out in correct units ("+N_upper.unit+")")
        return N_upper

    def column_densities(self):
        ne = dict()
        for i in self._intensitys:
            n = self.colden(i)
