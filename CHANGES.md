## Change Log

### 2.0.7

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
