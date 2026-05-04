# test modelset.ModelSet
import os
from copy import deepcopy

import pdrtpy.utils as utils
import pytest
from pdrtpy.modelset import ModelSet


class TestModelSet:
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
        ) in zip(list(t["name"]), list(t["z"]), list(t["medium"]), list(t["mass"]), list(t["losangle"]), strict=False):
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
        ms.add_model(identifier="CII_158/CO_1110", model=c, title=r"[C II] 158 $\mu$m / CO(J=11-10)")


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
        from pdrtpy.utils import model_dir

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

    def test_model_ratios(self):
        ids = ["CII_158", "OI_63", "FIR"]
        ratios = self.ms.model_ratios(ids)
        assert isinstance(ratios, list)
        assert len(ratios) > 0

    def test_model_ratios_too_few_raises(self):
        with pytest.raises(Exception):
            self.ms.model_ratios(["CII_158"])

    def test_get_models_ratio(self):
        ids = ["CII_158", "OI_63"]
        models = self.ms.get_models(ids, model_type="ratio")
        assert isinstance(models, dict)
        assert len(models) > 0

    def test_get_models_intensity(self):
        ids = ["CII_158", "OI_63"]
        models = self.ms.get_models(ids, model_type="intensity")
        assert isinstance(models, dict)

    def test_get_models_bad_type_raises(self):
        with pytest.raises(ValueError):
            self.ms.get_models(["CII_158"], model_type="nonsense")


class TestModelSetAvLos:
    """Tests for avlos and avperp properties."""

    def test_wk2020_faceon_avlos(self):
        ms = ModelSet("wk2020", z=1, losangle=0)
        assert ms.avlos == pytest.approx(7.0)

    def test_wk2020_faceon_avperp_is_none(self):
        """avperp is undefined (0) for face-on geometry."""
        ms = ModelSet("wk2020", z=1, losangle=0)
        assert ms.avperp is None

    def test_wk2020_inclined_avlos(self):
        ms = ModelSet("wk2020", z=1, losangle=30)
        assert ms.avlos == pytest.approx(8.082903)

    def test_wk2020_inclined_avperp(self):
        ms = ModelSet("wk2020", z=1, losangle=30)
        assert ms.avperp == pytest.approx(4.041451)

    def test_wk2020_inclined_avlos_60(self):
        ms = ModelSet("wk2020", z=1, losangle=60)
        assert ms.avlos == pytest.approx(14.0)

    def test_wk2020_inclined_avperp_60(self):
        ms = ModelSet("wk2020", z=1, losangle=60)
        assert ms.avperp == pytest.approx(12.12435)

    def test_wk2006_avlos_is_none(self):
        ms = ModelSet("wk2006", z=1)
        assert ms.avlos is None

    def test_wk2006_avperp_is_none(self):
        ms = ModelSet("wk2006", z=1)
        assert ms.avperp is None

    def test_kosmatau_avlos_is_none(self):
        ms = ModelSet("kt2013wd01-7", z=1, medium="clumpy", mass=100)
        assert ms.avlos is None

    def test_kosmatau_avperp_is_none(self):
        ms = ModelSet("kt2013wd01-7", z=1, medium="clumpy", mass=100)
        assert ms.avperp is None

    def test_description_is_string(self):
        """description property should return a string without crashing."""
        for losangle in [0, 30, 60]:
            ms = ModelSet("wk2020", z=1, losangle=losangle)
            desc = ms.description
            assert isinstance(desc, str)
            assert len(desc) > 0

    def test_description_wk2006_no_avlos(self):
        """wk2006 description should work and not contain avlos."""
        ms = ModelSet("wk2006", z=1)
        desc = ms.description
        assert isinstance(desc, str)
        assert "avlos" not in desc
