## Change Log

### Unreleased

### 2.0.7b

#### Models
- Intensity files added to list of available models (the models were always there but not easily accessible to users)

- Model access made easier with `ModelSet.get_models` method

#### Measurements
- add `title` in constructor which can be passed along to plots

- squeeze single pixel axes on `read`

- propogate masks in arithmetic operations

#### Plotting
- New `ModelPlot` class 
    - plots model files without the need of `LineRatioFit`
    - adds phase space plots
    - plotting from `LineRatioPlot` is now is delegated to `ModelPlot`

- Fixes and consistency in plot labels

- additional keywords such as `legend` added to plot methods

- add `usetex` option to PlotBase and derived classes

#### Tools
- add mask option `LineRatioFit.run()` with median absolute deviation as default. This results in better radiation field and density maps in low S/N regions

- some refactoring

- better handling of non-FITS units in wk2006 models

#### Documentation
- improved descriptions

- typos

- new style
