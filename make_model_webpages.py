from pdrtpy.modelset import ModelSet
from pdrtpy.plot.modelplot import ModelPlot
import numpy.ma as ma
import os

from jinja2 import Template

class Page():

    def make_page(self):
        success = True
        # check all models.tab files and existence of all therein
        t = ModelSet.all_sets()
        failed = list()
        for n,z,md,m in zip(list(t["name"]),list(t["z"]),list(t["medium"]),list(t["mass"])):
            print(n,z,md,m)
            mdict = dict()
            ms = ModelSet(name=n,z=z,medium=md,mass=m)
            mp = ModelPlot(ms)
            # stop complaining about too many figures
            mp._plt.rcParams.update({'figure.max_open_warning': 0})
            print(f'Making page for {n,z,md,m}')
            if m is None or ma.is_masked(m):
                dir = f'{n}_{z}_{md}'
            else:
                dir = f'{n}_{z}_{md}_{m}'
            dir = dir.replace(' ','_')
            os.mkdir(f'/tmp/mpound/{dir}')
            index = open(f'/tmp/mpound/{dir}/index.html','w')
            index.write(f'<html><head> <meta charset="utf-8">\n <meta name="viewport" content="width=device-width, initial-scale=1, shrink-to-fit=no">\n <meta name="description" content="Tools to analyze observations of photodissociation regions">\n <meta name="author" content="Marc W. Pound">\n <title>PhotoDissociation Region Toolbox {dir}</title>\n <!-- Font Awesome icons (free version)-->\n <script src="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/5.13.0/js/all.min.js" crossorigin="anonymous"></script>\n <!-- Font Awesome accessibility options -->\n <script src="https://use.fontawesome.com/824d9b17ca.js"></script>\n <link href="http://dustem.astro.umd.edu/freelancer/css/styles.css" rel="stylesheet">\n <!-- from https://startbootstrap.com/themes/freelancer/-->\n <link rel="stylesheet" href="http://dustem.astro.umd.edu/freelancer/css/heading.css">\n <link rel="stylesheet" href="http://dustem.astro.umd.edu/freelancer/css/body.css">\n \n <!-- PDRT specific CSS -->\n <link href="http://dustem.astro.umd.edu/css/pdrt.css" rel="stylesheet">\n </head><body><br>')
            index.write('<table class="table mytable table-striped table-striped table-bordered" bgcolor="white" >\n<tr>')

            i = 0
            numcols = 4
            for r in ms.table["ratio"]:
                if i !=0 and i%numcols == 0:
                    index.write("</tr>\n<tr>")
                try:
                    model=ms.get_model(r)
                    modelfile = ms.table.loc[r]["filename"]
                    if "/" in model._title:
                        model._title += " Intensity Ratio"
                    else:
                        if "FIR" not in model._title and "Surface" not in  model._title and "A_V" not in model._title:
                            model._title += " Intensity"
                    model._title = model._title.replace("$\mu$","&micro;").replace("$_{FIR}$","<sub>FIR</sub>").replace("$_2$","<sub>2</sub>").replace("$A_V$","A<sub>V</sub>")
                                            #.replace("$T_S$","T<sub>S</sub>")
                                            #.replace("$^{13}$","<sup>13</sup>")
                    #print(f"doing {r} = {modelfile}.png title={model._title}")
                    if "$" in model._title:
                        print(f"############ OOPS missed some latex {model._title}")
                    fig_out = f'{dir}/{modelfile}.png'
                    fig_html = f'{dir}/{modelfile}.html'
                    f_html = f'{modelfile}.html'
                    index.write(f'<td><a href="{f_html}">{model._title}</a></td>')
                    
                    mdict[r] = fig_html
                    i = i+1
                    if False:
                        if model.header["CTYPE1"] == "T_e":
                            # Iron line ratios are function of electron temperature and electron density
                            # not H2 density and radiation field.
                            mp.plot(r,label=True,
                                    norm="log",cmap='plasma')
                        else:
                            mp.plot(r,yaxis_unit="Habing",label=True,
                                    norm="log",cmap='plasma')
                        mp.savefig(f'/tmp/mpound/{fig_out}')
                        # This is supposed to stop complaints about 
                        # too many figures, but actually does not!
                        mp._plt.close(mp.figure) 
                except Exception as e:
                    success = False
                    failed.append(f'{r} {modelfile} : {str(e)}\n')
            if not success:
                print("Couldn't open these models:",failed)
            index.write('</tr></table></body></html>')
            index.close()

    def make_aux_page(self):
        pass

if __name__ == '__main__':
    p = Page()
    p.make_page()
