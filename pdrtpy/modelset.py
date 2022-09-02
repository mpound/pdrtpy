"""Manage pre-computed PDR models"""

import itertools
import collections
from copy import deepcopy
import numpy as np
from astropy.table import Table, Column, unique, vstack
import astropy.units as u
from .pdrutils import get_table,model_dir, _OBS_UNIT_
from .measurement import Measurement

class ModelSet(object):
    """Class for computed PDR Model Sets. :class:`ModelSet` provides interface with a directory containing the model FITS files and the ability to query details about

    :param name: identifying name, e.g., 'wk2006'
    :type name: str
    :param z:  metallicity in solar units.
    :type z: float
    :param medium:  medium type, e.g. 'constant density', 'clumpy', 'non-clumpy'
    :type medium: str
    :param mass: maximum clump mass (for KosmaTau models).  Default:None (appropriate for Wolfire/Kaufman models)
    :type float:
    :raises ValueError: If model set not recognized/found.
    """
    #@ToDo replace with kwargs?
    def __init__(self,name,z,medium="constant density",mass=None):
        self._all_models = get_table("all_models.tab")
        self._all_models.add_index("name")
        possible = dict()
        if name not in self._all_models["name"]:
            raise ValueError(f'Unrecognized model {name:s}. Choices  are: {list(self._all_models["name"])}')
        if np.all(self._all_models.loc[name]["mass"].mask):
            matching_rows = np.where((self._all_models["z"]==z) &
                                 (self._all_models["medium"]==medium))
            possible["mass"] = None
        else:
            matching_rows = np.where((self._all_models["z"]==z) &
                     (self._all_models["medium"]==medium) &
                     (self._all_models["mass"] == mass))
            possible["mass"] = self._all_models.loc[name]["mass"]
        for key in ["z", "medium"]:
            possible[key]=  self._all_models.loc[name][key]
        # ugh, possible[] resulting from above can be a Python native or a Column.
        # If only one row matches it will be a native, otherwise it will be a Column,
        # so we have to check if it is a Column or not, so that we can successfully
        # import numberscreate a numpy array.
        for i in possible:
            if possible[i] is None:
                continue
            if isinstance(possible[i],Column):
                # convert Column to np.array
                possible[i] = sorted(set(np.array(possible[i])))
            else:
                # convert native to np.array
                possible[i] = sorted(set(np.array([possible[i]])))

        #print("possible:",possible)
        if mass is None and possible['mass'] is not None:
            raise ValueError(f'mass value is required for model {name:s}. Allowed values are {possible["mass"]}')
        if matching_rows[0].size == 0:
            msg = f"Requested ModelSet not found in {name:s}. Check your input values.  Allowed z are {possible['z']}.  Allowed medium are {possible['medium']}."
            if possible['mass'] is not None:
                msg = msg + f" Allowed mass are {possible['mass']}."
            raise ValueError(msg)

        self._tabrow = self._all_models[matching_rows].loc[name]
        self._table = get_table(path=self._tabrow["path"],filename=self._tabrow["filename"])
        self._table.add_index("ratio")
        self._set_identifiers()
        self._set_ratios()
        self._default_unit = dict()
        self._default_unit["ratio"] = u.dimensionless_unscaled
        self._default_unit["intensity"] = _OBS_UNIT_
        self._user_added_models = dict()

    @property
    def description(self):
        """The description of this model

        :rtype: str
        """
        return self._tabrow["description"]+", Z=%2.1f"%self.z

    @property
    def code(self):
        """The PDR code that created this ModelSet, e.g. 'KOSMA-tau'

        :rtype: str
        """
        return self._tabrow["PDR code"]

    @property
    def name(self):
        """The name of this ModelSet

        :rtype: str
        """
        return self._tabrow["name"]

    @property
    def version(self):
        """The version of this ModelSet

        :rtype: str
        """
        return self._tabrow["version"]

    @property
    def medium(self):
        """The medium type of this model, e.g. 'constant density', 'clumpy', 'non-clumpy'

        :rtype: str
        """
        return self._tabrow["medium"]

    @property
    def z(self):
        """The metallicity of this ModelSet

        :rtype: float
        """
        return self.metallicity

    @property
    def metallicity(self):
        """The metallicity of this ModelSet

        :rtype: float
        """
        return self._tabrow["z"]

    @property
    def table(self):
        """The table containing details of the models in this ModelSet.

        :rtype: :class:`astropy.table.Table`
        """
        return self._table

    @property
    def identifiers(self):
        """Table of lines and continuum that are included in ratio models of this ModelSet. Only lines and continuum that are part of ratios are included in this list.   For a separate list of line and continuum intensity models see :meth:`~pdrtpy.modelset.ModelSet.supported_intensities`.

        :rtype: :class:`astropy.table.Table`
        """
        return self._identifiers

    @property
    def supported_intensities(self):
        """Table of lines and continuum that are covered by this ModelSet and have models separate from
        the any ratio model they might be in.

        :rtype: :class:`astropy.table.Table`
        """
        return self._supported_lines

    @property
    def supported_ratios(self):
        """The emission ratios that are covered by this ModelSet

        :rtype: :class:`astropy.table.Table`
        """
        return self._supported_ratios

    @property
    def user_added_models(self):
        """Show which models have been added to this ModelSet by the user

        :rtype: list
        """
        return list(self._user_added_models.keys())

    def ratiocount(self,m):
        """The number of valid ratios in this ModelSet, given a list of observation (:class:`~pdrtpy.measurement.Measurement`) identifiers.

        :param m: list of string :class:`~pdrtpy.measurement.Measurement` IDs, e.g. ["CII_158","OI_145","FIR"]
        :type m: list
        :returns: The number of model ratios found for the given list of measurement IDs
        :rtype: int
        """
        return len(self._get_ratio_elements(m))

    def find_pairs(self,m):
        """Find the valid model ratios labels in this ModelSet for a given list of measurement IDs

        :param m: list of string `Measurement` IDs, e.g. ["CII_158","OI_145","FIR"]
        :type m: list
        :returns: An iterator of model ratios labels for the given list of measurement IDs
        :rtype: iterator
        """
        if not isinstance(m, collections.abc.Iterable) or isinstance(m, (str, bytes)) :
            raise Exception("m must be an array of strings")

        for q in itertools.product(m,m):
            if q[0] == "FIR" and (q[1] == "OI_145" or q[1] == "OI_63") and "CII_158" in m:
                s = q[1] + "+CII_158/" + q[0]
            else:
                s = q[0]+"/"+q[1]
            if s in self.table["ratio"]:
                yield s

    def find_files(self,m,ext="fits"):
        """Find the valid model ratios files in this ModelSet for a given list of measurement IDs.  See :meth:`~pdrtpy.measurement.Measurement.id`

        :param m: list of string :class:`~pdrtpy.measurement.Measurement` IDs, e.g. ["CII_158","OI_145","FIR"]
        :type m: list
        :param ext: file extension. Default: "fits"
        :type ext: str
        :returns: An iterator of model ratio files for the given list of Measurement IDs
        :rtype: iterator
        """
        if not isinstance(m, collections.abc.Iterable) or isinstance(m, (str, bytes)):
            raise Exception("m must be an array of strings")
        for q in itertools.product(m,m):
            # must deal with OI+CII/FIR models. Note we must check for FIR first, since
            # if you check q has OI,CII and m has FIR order you'll miss OI/CII.
            if q[0] == "FIR" and (q[1] == "OI_145" or q[1] == "OI_63") and "CII_158" in m:
                #print("SPECIAL doing %s %s"%(q[0],q[1]))
                s = q[1] + "+CII_158/" + q[0]
            else:
                #print("doing %s %s"%(q[0],q[1]))
                s = q[0]+"/"+q[1]
            if s in self.table["ratio"]:
                fullpath = self._tabrow["path"]+self.table.loc[s]["filename"]+"."+ext
                tup = (s,fullpath)
                yield tup

    def model_ratios(self,m):
        '''Return the model ratios that match the input Measurement ID list.  You must provide at least 2 Measurements IDs

        :param m: list of string :class:`~pdrtpy.measurement.Measurement` IDs, e.g., ["CII_158","OI_145","FIR"]
        :type m: list
        :returns: list of string identifiers of ratios IDs, e.g., ['OI_145/CII_158', 'OI_145+CII_158/FIR']
        :rtype: list
        '''
        ratios = list()
        if len(m) < 2:
            raise Exception("m most contain at least two strings")
        for p in self.find_files(m):
            ratios.append(p[0])
        return ratios

    def model_intensities(self,m):
        '''Return the model intensities in this ModelSet that match the input Measurement ID list.
        This method will return the intersection of the input list and the list of supported lines.

        :param m: list of string :class:`~pdrtpy.measurement.Measurement` IDs, e.g., ["CII_158","OI_145","CS_21"]
        :type m: list
        :returns: list of string identifiers of ratios IDs, e.g., ['CII_158','OI_145']
        :rtype: list
        '''
        # get intersection of input list and supported lines
        return list(set(m) & set(self._supported_lines["intensity label"]))

    def get_model(self,identifier,unit=None,ext="fits"):
        '''Get a specific model by its identifier

        :param identifier: a :class:`~pdrtpy.measurement.Measurement` ID. It can be an intensity or a ratio, e.g., "CII_158","CI_609/FIR".
        :type identifier: str
        :returns: The model matching the identifier
        :rtype: :class:`~pdrtpy.measurement.Measurement`
        :raises: KeyError if identifier not found in this ModelSet
        '''

        if identifier in self._user_added_models:
            return self._user_added_models[identifier]

        if identifier not in self.table["ratio"]:
            raise KeyError(f"{identifier} is not in this ModelSet")

        d = model_dir()
        _thefile = d+self._tabrow["path"]+self.table.loc[identifier]["filename"]+"."+ext
        _title = self._table.loc[identifier]['title']
        if unit is None or unit == "":
            # make a guess at the unit
            if '/' in identifier:
                unit = self._default_unit["ratio"]
                modeltype = "ratio"
            else:
                unit = self._default_unit['intensity']
                modeltype = "intensity"
        else:
            if unit == u.dimensionless_unscaled:
                modeltype = "ratio"
            else:
                modeltype = "intensity"
        _model = Measurement.read(_thefile,title=_title,unit=unit,identifier=identifier)
        #if _model.unit=="":
        #    _model.unit = u.Unit("adu")#self._default_unit["ratio"]
        _wcs = _model.wcs
        _model.header["MODELTYP"] = modeltype
        _model.modeltype = modeltype
        #@todo this is messy.  clean up by doing if wcs.. first?
        if self.is_wk2006 or self.name == "smc":
        # fix WK2006 model headerslisthd
            if _wcs.wcs.cunit[0] == "":
                _model.header["CUNIT1"] = "cm^-3"
                _wcs.wcs.cunit[0] = u.Unit("cm^-3")
            else:
                _model.header["CUNIT1"] = str(_wcs.wcs.cunit[0])
            if _wcs.wcs.cunit[1] == "":
                _model.header["CUNIT2"] = "Habing"
                # Raises UnitScaleError:
                # "The FITS unit format is not able to represent scales that are not powers of 10.  Multiply your data by 1.600000e-03."
                # This causes all sorts of downstream problems.  Workaround in LineRatioFit.read_models().
                #_model.wcs.wcs.cunit[1] = habing_unit
        elif self.code == "KOSMA-tau":
        # fix KosmaTau model headers
            if _wcs.wcs.cunit[0] == "":
                _model.header["CUNIT1"] = "cm^-3"
                _wcs.wcs.cunit[0] = u.Unit("cm^-3")
            else:
                _model.header["CUNIT1"] = str(_wcs.wcs.cunit[0])
            if _wcs.wcs.cunit[1] == "":
                _model.header["CUNIT2"] = "Draine"
            else:
                _model.header["CUNIT2"] = str(_wcs.wcs.cunit[1])
        else:
            # copy wcs cunit to header. used later.
            _model.header["CUNIT1"] = str(_wcs.wcs.cunit[0])
            _model.header["CUNIT2"] = str(_wcs.wcs.cunit[1])

        return _model

    def get_models(self,identifiers,model_type="ratio",ext="fits"):
        '''get the models from thie ModelSet that match the input list of identifiers

        :param identifiers: list of string :class:`~pdrtpy.measurement.Measurement` IDs, e.g., ["CII_158","OI_145","CS_21"]
        :type identifiers: list
        :param model_type: indicates which type of model is requested one of 'ratio' or 'intensity'
        :type model_type: str
        :returns: The matching models as a list of :class:`~pdrtpy.measurement.Measurement`.
        :rtype: list
        :raises: KeyError if identifiers not found in this ModelSet
        '''

        #if identifier not in self._identifiers["ID"]:
        #    raise Exception("There is no model in ModelSet %s with the identifier %s"%(self.name,identifier))
        if model_type != "intensity" and model_type != "ratio" and model_type != "both":
            raise ValueError("Unrecognized model_type: must be one of 'intensity', 'ratio', or 'both'")
        models=dict()
        a = list()
        self._table.remove_indices('ratio')
        self._table.add_index('ratio')
        if model_type == "intensity" or model_type == "both":
            a.extend(self.model_intensities(identifiers))
        if model_type == "ratio" or model_type == "both":
            a.extend(self.model_ratios(identifiers))

        if model_type == "intensity" or model_type == "ratio":
            _unit = self._default_unit[model_type]
        else:
            _unit = None
        for k in a:
            models[k] = self.get_model(k,unit=_unit,ext=ext)

        return models

    def add_model(self,identifier,model,title,overwrite=False):
        r"""Add your own model to this ModelSet.

        :param identifier: a :class:`~pdrtpy.measurement.Measurement` ID. It can be an intensity or a ratio, e.g., "CII_158","CI_609/FIR".
        :type identifier: str
        :param model:  the model to add.  If a string, this must be the fully-qualified path of a FITS file.  If a :class:`~pdrtpy.measurement.Measurement` it must have the same CTYPEs and CUNITs as the models in the ModelSet(?).
        :type model: str or :class:`~pdrtpy.measurement.Measurement`
        :param title: A formatted string (e.g., LaTeX) describing this observation that can be used for plotting. Python r-strings are accepted, e.g., r'$^{13}$CO(3-2)'  would give :math:`^{13}{\rm CO(3-2)}`.
    :type title: str
        :param overwrite:  Whether to overwrite the model if the identifier already exists in the ModelSet or has been previously added.  Default: False
        :type overwrite: bool
        """
        if identifier not in self.table["ratio"] and identifier not in self._user_added_models:
            self._really_add_model(identifier,model,title)
        elif identifier in self._user_added_models and not overwrite:
            raise Exception(f"{identifier} was previously added to this ModelSet. If you wish to overwrite it, use overwrite=True")
        elif identifier in self.table["ratio"] and not overwrite:
            raise Exception(f"{identifier} is already in the {self.name} ModelSet. If you wish to overwrite it, use overwrite=True")
        else:
            #print(f"Overwriting {identifier}.")
            self._really_add_model(identifier,model,title)


    def _really_add_model(self,identifier,model,title):
        print("Adding user model %s"%identifier)
        if type(model) is str:
            m = Measurement.read(model,identifier=identifier)
        else:
            m = model
        self._user_added_models[identifier] = m
        if "/" in identifier: # it's a ratio
            if identifier in self._supported_ratios["ratio label"]:
                # ack, there should be a way just to replace title but I can't get Table.loc to work.
                index = np.where(self._supported_ratios["ratio label"] == identifier)[0][0]
                self._supported_ratios.remove_row(index)
            self._supported_ratios.add_row([title,identifier])
        else:
            if identifier in self._supported_lines["intensity label"]:
                index = np.where(self._supported_lines["intensity label"] == identifier)[0][0]
                self._supported_lines.remove_row(index)
            self._supported_lines.add_row([title,identifier])

    def _find_ratio_elements(self,m):
        # TODO handle case of OI+CII/FIR so it is not special cased in lineratiofit.py
        """Find the valid model numerator,denominator pairs in this ModelSet for a given list of measurement IDs. See :meth:`~pdrtpy.measurement.Measurement.id`

        :param m: list of string :class:`~pdrtpy.measurement.Measurement` IDs, e.g. ["CII_158","OI_145","FIR"]
        :type m: list
        :returns: iterator -- An dictionary iterator where key is numerator and value is denominator
        """
        if not isinstance(m, collections.abc.Iterable) or isinstance(m, (str, bytes)) :
            raise Exception("m must be an array of strings")

        for q in itertools.product(m,m):
            s = q[0]+"/"+q[1]
            z = dict()
            if s in self.table["ratio"]:
                z={"numerator":self.table.loc[s]["numerator"],
                   "denominator":self.table.loc[s]["denominator"]}
                yield z

    def _get_ratio_elements(self,m):
        """Get the valid model numerator,denominator pairs in this ModelSet for a given list of measurement IDs. See :meth:`~pdrtpy.measurement.Measurement.id`

        :param m: list of string :class:`~pdrtpy.measurement.Measurement` IDs, e.g. ["CII_158","OI_145","FIR"]
        :type m: list
        :returns: dict -- A dictionary where key is numerator and value is denominator
        """
        if not isinstance(m, collections.abc.Iterable) or isinstance(m, (str, bytes)) :
            raise Exception("m must be an array of strings")
        k = list()
        for q in itertools.product(m,m):
            s = q[0]+"/"+q[1]
            if s in self.table["ratio"]:
                z={"numerator":self.table.loc[s]["numerator"],
                   "denominator":self.table.loc[s]["denominator"]}
                k.append(z)
        self._get_oi_cii_fir(m,k)
        return k

    def _get_oi_cii_fir(self,m,k):
        """Utility method for determining ratio elements, to handle special
           case of ([O I] 63 micron + [C II] 158 micron)/I_FIR.

        :param m: list of string :class:`~pdrtpy.measurement.Measurement` IDs, e.g. ["CII_158","OI_145","FIR"]
        :type m: list
        :param k: list of existing ratios to append to
        :type k: list
        """
        if "CII_158" in m and "FIR" in m:
            if "OI_63" in m:
                num = "OI_63+CII_158"
                den = "FIR"
                #lab="OI_63+CII_158/FIR"
                z = {"numerator":num,"denominator":den}
                k.append(z)
            if "OI_145" in m:
                num = "OI_145+CII_158"
                den = "FIR"
                #lab="OI_145+CII_158/FIR"
                z = {"numerator":num,"denominator":den}
                k.append(z)

    def _set_ratios(self):
        """make a useful table of ratios covered by this model"""
        self._supported_ratios = Table( [ self.table["title"], self.table["denominator"], self.table["ratio"] ],copy=True)
        matching_rows = np.where(self._supported_ratios["denominator"]=="1")[0]
        self._supported_lines = Table(self._supported_ratios[matching_rows],copy=True)
        self._supported_lines.remove_column("denominator")
        self._supported_ratios.remove_rows(matching_rows)
        self._supported_ratios['title'].unit = None
        self._supported_ratios['ratio'].unit = None
        self._supported_ratios.remove_column("denominator")
        self._supported_ratios.rename_column("ratio","ratio label")
        self._supported_lines.rename_column("ratio","intensity label")

    def _set_identifiers(self):
        """make a useful table of identifiers of lines covered by ratios in this ModelSet"""
        # remove the single line intensity models from the list.
        matching_rows = np.where((self._table['denominator'] != "1"))[0]
        n=deepcopy(self._table['numerator'][matching_rows])
        n.name = 'ID'
        d=deepcopy(self._table['denominator'][matching_rows])
        d.name='ID'

        t1 = Table([self._table['title'][matching_rows],n],copy=True)
        # discard the summed fluxes as user would input them individually
        for id in ['OI_145+CII_158','OI_63+CII_158']:
            a = np.where(t1['ID']==id)[0]
            for z in a:
                t1.remove_row(z)
        # now remove denominator from title (everything from / onwards)
        for i in range(len(t1['title'])):
            if '/' in t1['title'][i]:
                t1['title'][i] = t1['title'][i][0:t1['title'][i].index('/')]

        t2 = Table([self._table['title'][matching_rows],d],copy=True)
        # remove numerator from title (everything before and including /)
        for i in range(len(t2['title'])):
            if '/' in t2['title'][i]:
                t2['title'][i] = t2['title'][i][t2['title'][i].index('/')+1:]
        t = vstack([t1,t2])
        t = unique(t,keys=['ID'],keep='first',silent=True)
        t['title'].unit = None
        t['ID'].unit = None
        t.rename_column('title','canonical name')
        self._identifiers = t

    @property
    def is_wk2006(self):
        """method to indicate this is a wk2006 model, to deal with quirks
           of that modelset

           :returns: True if it is.
        """
        return self.name == "wk2006"


    # ============= Static Methods =============
    @staticmethod
    def list():
        """Print the names and descriptions of available ModelSets (not just this one) """
        ModelSet.all_sets().pprint_all(align="<")

    @staticmethod
    def all_sets():
        """Return a table of the names and descriptions of available ModelSets (not just this one)

        :rtype: :class:`~astropy.table.Table`
        """
        t = get_table("all_models.tab")
        t.remove_column("path")
        t.remove_column("filename")
        return t
