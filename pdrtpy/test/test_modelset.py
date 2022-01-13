# test modelset.ModelSet
import unittest
from pdrtpy.modelset import ModelSet

class TestModelSet(unittest.TestCase):
    def test_existence(self):
        print("ModelSet Unit Test")
        success = True
        # check all models.tab files and existence of all therein
        t = ModelSet.all_sets()
        failed = list()
        for n,z,md,m in zip(list(t["name"]),list(t["z"]),list(t["medium"]),list(t["mass"])):
            print(n,z,md,m)
            ms = ModelSet(name=n,z=z,medium=md,mass=m)
            for r in ms.table["ratio"]:
                try:
                    ms.get_model(r)
                except Exception as e:
                    success = False
                    failed.append(str(e))
            if not success:
                print("Couldn't open these models:",failed)
            self.assertTrue(success)

if __name__ == '__main__':
    unittest.main()
