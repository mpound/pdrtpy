# test modelset.ModelSet
import unittest
from pdrtpy.modelset import ModelSet
from copy import deepcopy
import pdrtpy.pdrutils as utils


class TestModelSet(unittest.TestCase):
    def test_existence(self):
        #print("ModelSet Unit Test")
        success = True
        # check all models.tab files and existence of all therein
        t = ModelSet.all_sets()
        failed = list()
        global_success = []
        for n, z, md, m, losangle, avperp in zip(
            list(t["name"]), list(t["z"]), list(t["medium"]), list(t["mass"]), list(t["losangle"]), list(t["avperp"])
        ):
            #print(n, z, md, m, losangle, avperp)
            nodir = None
            try:
                fnf = True
                ms = ModelSet(name=n, z=z, medium=md, mass=m, losangle=losangle, avperp=avperp, debug=True)
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

if __name__ == "__main__":
    unittest.main()
