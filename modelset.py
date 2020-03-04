import pdrutils as utils
import numpy as np

#@ToDo:
#   model version should be a property
#   addModelSet() - for custom model sets. See model convention white paper
class ModelSet(object):
    """Class for PDR Model Sets. Model Sets will interface with a  
       directory containing the model FITS files.
       Parameters  
          name - string identifier
          z    - metallicity in solar units.  
    """
       
    def __init__(self,name,z):
        self._all_models = utils.get_table("all_models.tab")
        self._all_models.add_index("name")
        if name not in self._all_models["name"]:
            raise ValueError("Unrecognized model %s. Choices are: %s"%(name,self._possible_models))
        self._name = name

        zzz = self._all_models.loc[name]["z"]
        self._row = np.where(zzz==z)
        if self._row.size == 0:
            raise ValueError("Z=%2.1f not found in %s. Choices are: %s"%(z,name,zzz))
        self._metallicity = z
        self._row = self._row[0][0]
        # maybe I should just keep tabrow around and not make all these member variables.
        tabrow = self._all_models[self._row]
        self._version = tabrow["version"]
        self._filename = tabrow["filename"]
        self._table = utils.get_table(path=tabrow["path"],filename=tabrow["filename"])

    @property
    def description(self):
        """Return the description of this model"""
        return self._all_models.loc[self._name][self._row]["description"]+", Z=%2.1f"%self.z

    @property
    def name(self):
        """Return the name of this model"""
        return self._name

    @property
    def version (self):
        """Return the name of this model"""
        return self._version

    @property
    def z(self):
        """Return the metallicity of this model"""
        return self.metallicity

    @property
    def metallicity(self):
        """Return the metallicity of this model"""
        return self._metallicity

    #@property
    #def isSolarMetallicity(self):
    #    return self._metallicity == 1

    @property
    def table(self):
        """Return an astropy Table containing details of the models"""
        return self._table
    

#class ModelSet(ModelSetBase)
#    """PDR Model Sets Model Sets interface with a  
#       directory containing the model FITS files.
#       Parameters  
#          name - string identifier
#          z    - metallicity in solar units.  Default: 1 (==solar)
#    """
#    def __init__(self,name,z=1):
#        super(name,z)
#        self._table = utils.get_table(self.all_models[name])
#
#    def has_metallicity(self,z):
#        """Check if this model set contains a given metallicity 
#           Parameters:
#              z  - metallicity in solar units.  
#           Returns: 
#           True if the model set has at least one model of this metallicity
#        """
#        return np.any(np.isin(z,self._table["z"]),axis=0)
#
