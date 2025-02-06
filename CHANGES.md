## Change Log

### Release 2.4.2

#### _Models_

- LMC metallicity surface temperature models added

- Guidance for interpretation of [O I] 1.316 micron line added https://dustem.astro.umd.edu/oidiagnostic/index.html

#### _H2 Excitation Tool_


- fitting of visual extinction Av supported (experimental, API likely to change)
 
- bug in `explore()` method fixed, where clicking brought up the wrong excitation plot

#### _Issues_

- issues 55, 58, 80, 84, 90, 93, 94, 95, 96,99, 101, 117, 157, 158, 163 closed

#### _Documentation_

- updated to new style, with easier searching and navigation

#### _Other_

- drop support for Python 3.9

- completely updated build system (setup.py &rarr; pyproject.toml) and continuous integration workflow

- new release workflow


### Release 2.3.1
#### _Models_

- added new Kosma-tau kt2020 models which use parameters that closely match wk2020 models

- updated HII region diagnostic models

#### _Issues_

- issue #78 reported, fixed, and closed

#### _Documentation_

- updated

#### _Notebooks_ ###

- existing example notebooks updated, new notebook from Dec 6 JWST webinar added


### Release 2.3.0
#### _H2 Excitation Tool_

- allow single temperature fit

- allow plotting of multiple vibrational levels

#### _Plotting_

- new ModelPlot.isoplot method to make isocontour plots of individual model parameters

- refactoring of plotting capability in plot-based tools

#### _Documentation_

- updated

#### _Notebooks_ ###

- existing example notebooks updated, most notably to show new PDR Toolbox functionality

#### _Issues_

- issues 22,59,60,65 closed

- various other bugs discovered or reported and fixed

#### _Website_

- fixed labelling of ionized line plots

### Release 2.2.9

#### _Models_

- allow user added models

#### _Plotting_

- consistent handling of aspect ratios and colorbars

#### _Documentation_

- updated

#### _Issues_

- issues 55, 58, 59 closed

- various other bugs discovered and fixed

#### _Notebooks_ ###

- existing example notebooks updated

### _Testing_

- expanded test suite using GitHub workflow

- PEP8 compliance with flake

- added examples directory with test code from AJ paper, which are now auto-run as part of test suite

### Release 2.2.6
#### _Models_

- adds CI/13CO models

- typo fix in  CII\_158/OI\_145  model tag

### Release 2.2.5
#### _Models_

- adds 13CO models and updates iron line ionized gas models

### Release 2.2.3
#### _H2 Excitation Fitting_

- H2 Excitation fitting for maps completed.

    - each unmasked pixel in a map fitted

    - user can see maps of column density, hot and cold temperature, ortho-to-para ratios

    - "explore" function to see fit at various points in a map

#### _Density and Radiation Field Fitting_

- Now uses LMFIT package for fitting.

- Map fitting updated to allow interpolated solutions between model grid points

- MCMC method available (best for single pixel fitting) via emcee package

#### _Models_

- updated and expanded Wolfire/Kaufman 2020 models covering many more spectral lines

- large set of KOSMA-Tau models now available


#### _Plotting_

- consolidated some plotting functions

- added text() method to add text to any plot

#### _Documentation_

- updated

#### _Issues_

- issues 28, 34, 38 closed

- various other bugs discovered and fixed

#### _Notebooks_ ###

- existing example notebooks updated

### Release 2.1.1

#### _H2 Excitation Fitting_

- H2 Excitation fitting tool first version completed.

  - Completely rewritten fitting method to use LMFIT package

  - Allow fitting of ortho-to-para ratio

  - Use Measurements for all fitted quantities

  - Compute total column densities using partition function

- H2 Excitation fitting example notebook completed.

#### _Measurements_

  - Measurements can now be read from tables

  - Better formatting for print using `__format__' and f-strings

#### _Models_
 - added to Wolfire-Kaufman 2006 z=1 set:
     * CO(6-5) / CO(3-2)
     * [C II] 158&mu;m / FIR
     * [C II] 158&mu;m / CO(6-5)
     * [C II] 158&mu;m / CO(3-2)
     * [Fe II] 1.60&mu;m / [Fe II] 1.64&mu;m
     * [Fe II] 1.64&mu;m / [Fe I\] 5.43&mu;m

#### _Plotting_

- H2 Excitation plotter completed

- Phase space plots can now plot multiple points

#### _Documentation_

- updated

#### _Issues_

- issues 2, 12, 14, 28 closed

### Release 2.0.7

#### _Models_

- Intensity files added to list of available models (the models were always there but not easily accessible to users)

- Model access made easier with `ModelSet.get_models` method

#### _Measurements_

- add `title` in constructor which can be passed along to plots

- squeeze single pixel axes on `read`

- propogate masks in arithmetic operations

#### _Plotting_

- New `ModelPlot` class

    - plots model files without the need of `LineRatioFit`

    - adds phase space plots

    - plotting from `LineRatioPlot` is now is delegated to `ModelPlot`

- Fixes and consistency in plot labels

- additional keywords such as `legend` added to plot methods

- add `usetex` option to PlotBase and derived classes

#### _Tools_

- add mask option `LineRatioFit.run()` with median absolute deviation as default. This results in better radiation field and density maps in low S/N regions

- some refactoring

- better handling of non-FITS units in wk2006 models

#### _Documentation_

- improved descriptions

- typos

- new style

#### __Issues__

- issues 4, 8, 10, 15, 17, 19 closed
