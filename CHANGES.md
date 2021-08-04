## Change Log

### Release 2.1

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
     * CO(6-5)/CO(3-2)
     * \[C II\] 158&mu;m/FIR
     * \[C II\] 158&mu;m/CO(6-5)
     * \[C II\] 158&mu;m/CO(3-2)
     * \[Fe II\] 1.60&mu;m/\[Fe II\] 1.64&mu;m
     * \[Fe II\] 1.64&mu;m/\[Fe II\] 5.43&mu;m
     
#### _Plotting_     

- H2 Excitation plotter completed 

- Phase space plots can now plot multiple points

#### _Documentation_

- updated

#### _Issues_

- issues 12, 14, 28 closed

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
