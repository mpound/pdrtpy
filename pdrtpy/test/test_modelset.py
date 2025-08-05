# test modelset.ModelSet
import unittest

from pdrtpy.modelset import ModelSet


class TestModelSet(unittest.TestCase):
    def test_existence(self):
        print("ModelSet Unit Test")
        success = True
        # check all models.tab files and existence of all therein
        t = ModelSet.all_sets(debug=True)
        failed = list()
        global_success = []
        for n, z, md, m, losangle, avperp in zip(
            list(t["name"]), list(t["z"]), list(t["medium"]), list(t["mass"]), list(t["losangle"]), list(t["avperp"])
        ):
            # print(n, z, md, m)
            nodir = None
            try:
                fnf = True
                ms = ModelSet(name=n, z=z, medium=md, mass=m, inc=losangle, avperp=avperp, debug=True)
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


if __name__ == "__main__":
    unittest.main()
