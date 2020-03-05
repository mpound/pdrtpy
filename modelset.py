import pdrutils as utils
import numpy as np

#@ToDo:
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

        zzz = self._all_models.loc[name]["z"]
        self._row = np.where(zzz==z)
        if self._row[0].size == 0:
            raise ValueError("Z=%2.1f not found in %s. Choices are: %s"%(z,name,zzz))
        self._row = self._row[0][0]
        self._tabrow = self._all_models[self._row]
        self._table = utils.get_table(path=self._tabrow["path"],filename=self._tabrow["filename"])

    @property
    def description(self):
        """Return the description of this model"""
        return self._tabrow["description"]+", Z=%2.1f"%self.z

    @property
    def name(self):
        """Return the name of this model"""
        return self._tabrow["name"]

    @property
    def version (self):
        """Return the name of this model"""
        return self._tabrow["version"]

    @property
    def z(self):
        """Return the metallicity of this model"""
        return self.metallicity

    @property
    def metallicity(self):
        """Return the metallicity of this model"""
        return self._tabrow["z"]

    #@property
    #def isSolarMetallicity(self):
    #    return self._metallicity == 1

    @property
    def table(self):
        """Return an astropy Table containing details of the models"""
        return self._table

    @property
    def supported_lines(self):
        '''Return a `set` of lines and continuum that are covered by this ModelSet'''
        #TODO see below
        return set(np.append(self._table["numerator"].data,self._table["denominator"].data))

    @property
    def supported_ratios(self):
       '''Return the emission ratios that are covered by this model'''
       #TODO: keep as Table column or as something else. Should be consistent with return type of supported_lines
       return self.table["label"]

    # ============= Static Methods =============
    @staticmethod
    def list():
        """List the names and descriptions of available models"""
        t = utils.get_table("all_models.tab")
        t.remove_column("path")
        t.remove_column("filename")
        t.pprint_all(align="<")

    @staticmethod
    def WolfireKaufman():
    #ToDo later upgrade to variable z when 2020 models are available
        """Easy way to get the latest Wolfire-Kaufman models, z=1"""
        return ModelSet("wk2006",z=1)

    @staticmethod
    def KosmaTau():
        """Easy way to get the latest KOSMA TAU models, z=1"""
        return ModelSet("kosmatau",z=1)
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
