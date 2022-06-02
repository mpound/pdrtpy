#!/usr/bin/env python
from pdrtpy.modelset import ModelSet
from pdrtpy.plot.modelplot import ModelPlot
import numpy.ma as ma
import os
import jinja2

class Page():

    def make_page(self):
        debug = True
        explain = dict()
        explain["wk2006"] = 'This ModelSet uses the models of <a class="mya" href="http://adsabs.harvard.edu/cgi-bin/nph-bib_query?bibcode=1999ApJ...527..795K" >Kaufman et al. 1999</a> and <a class="mya" href="https://ui.adsabs.harvard.edu/abs/2006ApJ...644..283K/abstract" >Kaufman et al. 2006 </a> are used with <a class="mya" href="/models.html#parameters">these parameters.</a> More details are in the FITS headers.'
        explain["wk2020"] = 'The models in this ModelSet are based on <a class="mya" href="https://ui.adsabs.harvard.edu/abs/2006ApJ...644..283K/abstract" >Kaufman et al. 2006 </a>, <a class="mya" href="https://ui.adsabs.harvard.edu/abs/2010ApJ...716.1191W/abstract">Wolfire et al. 2010</a>, and <a class="mya" href="https://ui.adsabs.harvard.edu/abs/2016ApJ...826..183N/abstract">Neufeld &amp; Wolfire 2016</a> are used with <a class="mya" href="/models.html#2020">these parameters.</a> More details are in the FITS headers.'
        explain["kt2013"] = 'The models in this ModelSet were created with the <a class="mya" href="https://astro.uni-koeln.de/stutzki/research/kosma-tau">KOSMA-tau</a> PDR code. More details are in the FITS headers.'
        model_title = dict()
        model_title["wk2006"] = "Wolfire/Kaufman 2006"
        model_title["wk2020"] = "Wolfire/Kaufman 2020"
        model_title["kt2013"] = "KOSMA-tau 2013"

        tarball = dict()
        tarball["wk2006"] = "/models/wk2006_models.tgz"
        tarball["wk2020"] = "/models/wk2020_models.tgz"
        tarball["kt2020"] = "/models/kt2013_models.tgz"

        success = True
        # check all models.tab files and existence of all therein
        t = ModelSet.all_sets()
        failed = list()
        env=jinja2.Environment(loader=jinja2.FileSystemLoader("."))
        template = env.get_template('jinjatemplate.html')
        base_dir = "/home/mpound/pdrtoolbox-web/models"
        for n,z,md,m in zip(list(t["name"]),list(t["z"]),list(t["medium"]),list(t["mass"])):
            print(n,z,md,m)
            mdict = dict()
            if debug and n != "wk2006": continue
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
            os.mkdir(f'{base_dir}/{dir}')
            index = open(f'{base_dir}/{dir}/index.html','w')
            index.write(f'<html><head> <meta charset="utf-8">\n <meta name="viewport" content="width=device-width, initial-scale=1, shrink-to-fit=no">\n <meta name="description" content="Tools to analyze observations of photodissociation regions">\n <meta name="author" content="Marc W. Pound">\n <title>PhotoDissociation Region Toolbox {dir}</title>\n <!-- Font Awesome icons (free version)-->\n <script src="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/5.13.0/js/all.min.js" crossorigin="anonymous"></script>\n <!-- Font Awesome accessibility options -->\n <script src="https://use.fontawesome.com/824d9b17ca.js"></script>\n <link href="http://dustem.astro.umd.edu/freelancer/css/styles.css" rel="stylesheet">\n <!-- from https://startbootstrap.com/themes/freelancer/-->\n <link rel="stylesheet" href="http://dustem.astro.umd.edu/freelancer/css/heading.css">\n <link rel="stylesheet" href="http://dustem.astro.umd.edu/freelancer/css/body.css">\n \n <!-- PDRT specific CSS -->\n <link href="http://dustem.astro.umd.edu/css/pdrt.css" rel="stylesheet">\n </head>\<body id="page-top"\n><br>')

            navbar = '<nav class="navbar navbar-expand-lg bg-secondary fixed-top" id="mainNav">\n\
                    <div class="container"><a class="navbar-brand js-scroll-trigger" href="#page-top">PhotoDissociation Region Toolbox</a>\n\
                        <button class="navbar-toggler navbar-toggler-right font-weight-bold bg-primary text-white rounded" type="button" data-toggle="collapse" data-target="#navbarResponsive" aria-controls="navbarResponsive" aria-expanded="false" aria-label="Toggle navigation">Menu<i class="fas fa-bars"></i></button>\n\
                        <div class="collapse navbar-collapse" id="navbarResponsive">\n\
                            <ul class="navbar-nav ml-auto">\n\
                                    <li class="nav-item mx-0 mx-lg-1"><a class="nav-link py-3 px-0 px-lg-3 rounded js-scroll-trigger" href="/index.html">HOME</a>\n\
                                    <li class="nav-item mx-0 mx-lg-1"><a class="nav-link py-3 px-0 px-lg-3 rounded js-scroll-trigger" href="/tools.html">TOOLS</a>\n\
                                    </li>\n\
                                    <li class="nav-item mx-0 mx-lg-1"><a class="nav-link py-3 px-0 px-lg-3 rounded js-scroll-trigger" href="/models.html">MODELS</a>\n\
                                    <li class="nav-item mx-0 mx-lg-1"><a class="nav-link py-3 px-0 px-lg-3 rounded js-scroll-trigger" href="/docs.html">DOCUMENTS</a>\n\
                                </li>\n\
                            </ul>\n\
                        </div>\n\
                    </div>\n\
                </nav>\n'
            
            index.write(navbar)
            preamble = f'<header class="mymasthead bg-primary text-white text-center"> <div class="container d-flex flex-column"> <!-- Masthead Subheading--> <h1 class="pre-wrap mymasthead-heading font-weight-bold mb-0">{model_title[n]} Z={z} {md} Models</h1> <p class="myp"> Below are links to the individual FITS files and plots in the <i>{n}</i> <a class="mya" href="https://pdrtpy.readthedocs.io/en/latest/pdrtpy.modelset.html">ModelSet</a>. We also provide a <a class="mya" href="{tarball[n]}">gzipped tarball</a> of all the <i>{n}</i> FITS files.'
            index.write(preamble)
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
                    model._title = model._title.replace("$\mu$","&micro;").replace("$_{FIR}$","<sub>FIR</sub>").replace("$_2$","<sub>2</sub>").replace("$A_V$","A<sub>V</sub>").replace("$^{13}$","<sup>13</sup>").replace("$A_V=0.01$","A<sub>V</sub> = 0.01")
                                            #.replace("$T_S$","T<sub>S</sub>")
                    #print(f"doing {r} = {modelfile}.png title={model._title}")
                    if "$" in model._title:
                        print(f"############ OOPS missed some latex {model._title}")
                    fig_out = f'{dir}/{modelfile}.png'
                    fig_html = f'{dir}/{modelfile}.html'
                    f_html = f'{modelfile}.html'
                    index.write(f'<td><a href="{f_html}">{model._title}</a></td>')
                    mdict[r] = fig_html
                    i = i+1
                    if model.wcs.wcs.ctype[0] == "T_e":
                        # Iron line ratios are function of electron temperature and electron density
                        # not H2 density and radiation field.
                        mp.plot(r,label=True,legend=False,
                                norm="log",cmap='plasma')
                    else:
                        mp.plot(r,yaxis_unit="Habing",label=True, legend=False,
                                norm="log",cmap='plasma')
                    mp.savefig(f'{base_dir}/{fig_out}')
                    # This is supposed to stop complaints about 
                    # too many figures, but actually does not!
                    mp._plt.close(mp.figure) 
                    fh = open(f'{base_dir}//{fig_html}','w')
                    output=template.render(model=model,
                                           fitsfilename=f'{modelfile}.fits',
                                           model_explain=explain[n],
                                           modelfile=modelfile)
                    fh.write(output)
                    #print(f'{base_dir}/{fig_html}')
                   # print(output)
                   # print("===========================================")
                    fh.close()

                except FileNotFoundError:
                    raise
                except Exception as e:
                    raise
                    success = False
                    failed.append(f'{r} {modelfile} : {str(e)}\n')
            if not success:
                print("Couldn't open these models:",failed)
            index.write('</tr></table>')
            endpage = '       </p> </div> </header> <footer class="footer text-center"> <div class="container"> <div class="row"> <!-- Footer Social Icons--> <div class="col-lg-4 mb-5 mb-lg-0"> <a class="btn btn-outline-light btn-social mx-1" href="https://github.com/mpound/pdrtpy"> <i class="fab fa-fw fa-github" title="Visit us on Github" alt="Visit us on Github"></i></a> <a class="btn btn-outline-light btn-social mx-1" href="https://pdrtpy.readthedocs.io" title="Visit us on ReadTheDocs" alt="Visit us on ReadTheDocs"> <i class="fa fa-fw fa-book"></i></a> <a class="btn btn-outline-light btn-social mx-1" href="https://www.umd.edu/web-accessibility" title="UMD Web Accessibility" alt="UMD Web Accessibility"> <i class="fa fa-fw fa-universal-access"></i></a> <a href="https://ascl.net/1102.022"><img src="https://img.shields.io/badge/ascl-1102.022-blue.svg?colorB=262255" alt="ascl:1102.022" /></a> </div> <!-- Footer Location--> <div class="col-lg-4 mb-5 mb-lg-0"> <p class="pre-wrap lead mb-0">Astronomy Department<br>University of Maryland<br>College Park, MD 20740</p> </div> <!-- Footer LOGOS--> <div class="col-lg-1 mb-5 mb-lg-0"> <a class="btn btn-outline-light btn-social mx-1" href="https://www.umd.edu" title="University of Maryland"> <img class="p-3" src="/images/New_UMD_Globe_small.png" width="100px" alt="UMD logo"> </a> </div> <div class="col-lg-1 mb-5 mb-lg-0"> <a class="btn btn-outline-light btn-social mx-1 p-3" href="http://www.nasa.gov" title="NASA"> <img class="p-3" src="/images/nasa-logo-web-rgb.png" width="120px" alt="NASA logo"> </a> </div> </div> </div> </footer> <!-- Copyright Section--> <div class="copyright py-4 text-center text-white"> <div class="container"><small class="pre-wrap">Copyright &copy; 2Marx Productions 2020 </small></div> </div> <!-- Javascript includes --> <div id="scripting"> <!-- Scroll to Top Button (Only visible on small and extra-small screen sizes)--> <div class="scroll-to-top d-lg-none position-fixed"><a class="js-scroll-trigger d-block text-center text-white rounded" href="#page-top"><i class="fa fa-chevron-up"></i></a></div> <!-- Bootstrap core JS--> <script src="https://cdnjs.cloudflare.com/ajax/libs/jquery/3.4.1/jquery.min.js"></script> <script src="https://stackpath.bootstrapcdn.com/bootstrap/4.4.1/js/bootstrap.bundle.min.js"></script> <!-- Third party plugin JS--> <script src="https://cdnjs.cloudflare.com/ajax/libs/jquery-easing/1.4.1/jquery.easing.min.js"></script> <!-- LaTeX math in HTML --> <script async="async" type="text/javascript" src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.5/latest.js?config=TeX-AMS-MML_HTMLorMML"></script> <!-- Core theme JS from https://startbootstrap.com/themes/freelancer/--> <script src="freelancer/js/scripts.js"></script> </div> </body> </html>'
            index.write(endpage)
            index.close()

    def make_aux_page(self):
        pass

if __name__ == '__main__':
    p = Page()
    p.make_page()
