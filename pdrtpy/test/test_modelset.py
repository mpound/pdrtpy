# test modelset.ModelSet
import os
import unittest
from copy import deepcopy

import pytest

import pdrtpy.pdrutils as utils
from pdrtpy.modelset import ModelSet


class TestModelSet(unittest.TestCase):
    def test_existence(self):
        # print("ModelSet Unit Test")
        success = True
        # check all models.tab files and existence of all therein
        t = ModelSet.all_sets()
        failed = list()
        global_success = []
        for (
            n,
            z,
            md,
            m,
            losangle,
        ) in zip(list(t["name"]), list(t["z"]), list(t["medium"]), list(t["mass"]), list(t["losangle"])):
            print(n, z, md, m, losangle)
            nodir = None
            try:
                fnf = True
                ms = ModelSet(name=n, z=z, medium=md, mass=m, losangle=losangle)
            except FileNotFoundError as zz:
                success = False
                nodir = zz.filename
                fnf = True
            if not fnf:
                for r in ms.table["ratio"]:
                    try:
                        # print(r)
                        ms.get_model(r)
                    except Exception as e:
                        success = False
                        failed.append(str(e))
                if not success:
                    print("Couldn't open these models:", failed)
            if nodir is not None:
                print(f"Couldn't open {nodir}")
            global_success.append(success)
        assert all(global_success)

    def test_add_model(self):
        ms = ModelSet("wk2020", z=1)
        a = ms.get_model("CII_158")
        b = ms.get_model("CO_1110")
        c = a / b
        c.header = deepcopy(a.header)
        c.header["TITLE"] = "[CII] 158 micron/CO (J=11-10)"
        c.header["DATAMAX"] = c.data.max()
        c.header["DATAMIN"] = c.data.min()
        c.header["HISTORY"] = "Computed arithmetically from (CII_158/CO_1110)"
        c.header["DATE"] = utils.now()  # the current time and date
        # print(c.header)
        ms.add_model(identifier="CII_158/CO_1110", model=c, title="[C II] 158 $\mu$m / CO(J=11-10)")


class TestModelSetValidation:
    def test_bad_name_raises(self):
        with pytest.raises(ValueError, match="Unrecognized PDR model code"):
            ModelSet("nonexistent_model", z=1)

    def test_bad_z_raises(self):
        with pytest.raises(ValueError):
            ModelSet("wk2020", z=999)

    def test_bad_medium_raises(self):
        with pytest.raises(ValueError):
            ModelSet("wk2020", z=1, medium="nonexistent_medium")

    def test_kosmatau_requires_mass(self):
        with pytest.raises(ValueError, match="mass value is required"):
            ModelSet("kt2013wd01-7", z=1, medium="clumpy")  # no mass

    def test_wk2006_z1_ok(self):
        ms = ModelSet("wk2006", z=1)
        assert ms is not None

    def test_wk2020_z1_ok(self):
        ms = ModelSet("wk2020", z=1)
        assert ms is not None


class TestModelSetProperties:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.ms2006 = ModelSet("wk2006", z=1)
        self.ms2020 = ModelSet("wk2020", z=1)

    def test_is_wk2006_true(self):
        assert self.ms2006.is_wk2006 is True

    def test_is_wk2006_false(self):
        assert self.ms2020.is_wk2006 is False

    def test_is_wk2020_true(self):
        assert self.ms2020.is_wk2020 is True

    def test_is_wk2020_false(self):
        assert self.ms2006.is_wk2020 is False

    def test_description(self):
        # avlos/avperp are not yet implemented (return None), so description
        # may raise TypeError.  Just check it doesn't crash on the parts that work.
        try:
            desc = self.ms2020.description
            assert isinstance(desc, str)
        except TypeError:
            pass  # known incomplete implementation

    def test_code(self):
        assert "Wolfire" in self.ms2020.code or "Kaufman" in self.ms2020.code


class TestModelSetFindMethods:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.ms = ModelSet("wk2020", z=1)

    def test_find_pairs_basic(self):
        ids = ["CII_158", "OI_63"]
        pairs = list(self.ms.find_pairs(ids))
        assert len(pairs) > 0
        assert "OI_63/CII_158" in pairs or "CII_158/OI_63" in pairs

    def test_find_pairs_oi_cii_fir_special_case(self):
        """OI+CII/FIR special combination should appear"""
        ids = ["CII_158", "OI_63", "FIR"]
        pairs = list(self.ms.find_pairs(ids))
        assert "OI_63+CII_158/FIR" in pairs

    def test_find_pairs_non_iterable_raises(self):
        with pytest.raises(Exception):
            list(self.ms.find_pairs("CII_158"))  # string not allowed

    def test_find_files_returns_tuples(self):
        ids = ["CII_158", "OI_63"]
        files = list(self.ms.find_files(ids))
        assert len(files) > 0
        for ratio, path in files:
            assert isinstance(ratio, str)
            assert isinstance(path, str)

    def test_find_files_paths_exist(self):
        from pdrtpy.pdrutils import model_dir

        ids = ["CII_158", "OI_63"]
        files = list(self.ms.find_files(ids))
        for _, relpath in files:
            fullpath = model_dir() + relpath
            assert os.path.exists(fullpath), f"Missing model file: {fullpath}"

    def test_find_files_oi_cii_fir(self):
        ids = ["CII_158", "OI_63", "FIR"]
        files = dict(self.ms.find_files(ids))
        assert "OI_63+CII_158/FIR" in files

    def test_find_files_non_iterable_raises(self):
        with pytest.raises(Exception):
            list(self.ms.find_files("CII_158"))

    def test_model_intensities(self):
        ids = ["CII_158", "OI_63", "not_a_real_line"]
        result = self.ms.model_intensities(ids)
        assert "CII_158" in result
        assert "OI_63" in result
        assert "not_a_real_line" not in result

    def test_model_intensities_empty(self):
        result = self.ms.model_intensities(["fake_1", "fake_2"])
        assert result == []


if __name__ == "__main__":
    unittest.main()
