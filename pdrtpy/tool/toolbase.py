import pdrtpy.pdrutils as utils

class ToolBase(object):
    """ Base class object for PDR Toolbox tools.  This class implements a simple 
        interface with a run method.  Tools will generally do some set up
        such as reading in observational data before run() can be invoked.
    """
    def __init__(self):
        self._measurements = None
        self._measurementnaxis = -1 
        
    def run(self):
        """Runs the tool. Each subclass Tool must implement its own run() method.
        """
        pass
    
    def _set_measurementnaxis(self):
        if self._measurements is None: return
        fk = utils.firstkey(self._measurements)
        self._measurementnaxis = len(self._measurements[fk].shape)
        
    @property
    def has_maps(self):
        '''Are the Measurements used map-based?. (i.e., have 2 spatial axes)
        
        :returns: True, if the observational inputs are spatial maps, False otherwise
 
        :rtype: bool
        '''
        
        return self._measurementnaxis > 1
    @property  
    def has_vectors(self):
        '''Are the Measurements used a Nx1 vector, e.g. read in from a table with :meth:`~pdrtpy.Measurement.from_table`.
        
        :returns: True, if the observational inputs are a vector, False otherwise
        :rtype: bool
        '''
        return self._measurementnaxis == 1
    
    @property
    def has_scalar(self):
        '''Are the Measurements used scalars.
        
        :returns: True, if the observational inputs are scalars, False otherwise
        :rtype: bool
        '''
        return self._measurementnaxis == 0