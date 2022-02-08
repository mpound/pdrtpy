from pdrtpy.modelset import ModelSet
from pdrtpy.plot.modelplot import ModelPlot
import numpy.ma as ma
import os

class Page():

    def make_page(self):
        success = True
        # check all models.tab files and existence of all therein
        t = ModelSet.all_sets()
        failed = list()
        for n,z,md,m in zip(list(t["name"]),list(t["z"]),list(t["medium"]),list(t["mass"])):
            print(n,z,md,m)
            if n == "wk2020" and md == "constant density":
                ms = ModelSet(name=n,z=z,medium=md,mass=m)
                mp = ModelPlot(ms)
                # stop complaining about too many figures
                mp._plt.rcParams.update({'figure.max_open_warning': 0})
                print(f'Making page for {n,z,md,m}')
                if m is None or m != '--' or ma.is_masked(m):
                    dir = f'{n}_{z}_{md}'
                else:
                    dir = f'{n}_{z}_{md}_{m}'
                dir = dir.replace(' ','_')
                os.mkdir(f'/tmp/mpound/{dir}')
                for r in ms.table["ratio"]:
                    try:
                        modelid = ms.table.loc[r]["filename"]
                        print(f"doing {r} = {modelid}.png")
                        mp.plot(r,yaxis_unit="Habing",label=True,
                                norm="log",cmap='plasma')
                        mp.savefig(f'/tmp/mpound/{dir}/{modelid}.png')
                        # this does not stop complaints about too many figures
                        mp._plt.close(mp.figure) 
                    except Exception as e:
                        success = False
                        failed.append(str(e))
                if not success:
                    print("Couldn't open these models:",failed)

if __name__ == '__main__':
    p = Page()
    p.make_page()
