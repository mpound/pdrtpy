## Roadmap of Desired Functionality

These are the big-picture things we want to add before the money runs out!


#### _Radiation Field and Density Fitting_

:heavy_check_mark: use lmfit minimization to allow off-grid minimum $\chi^2$

:heavy_check_mark: full support for map-based pixel-by-pixel fitting

- regularization technique to reduce unphysical spatial variation

#### _H2 Excitation Fitting_

:heavy_check_mark: add full Roueff et al Table of H2 line parameters

- helpful methods to compute temperature and column density for users with insufficient data for full fitting

:heavy_check_mark: full support for map-based pixel-by-pixel fitting

- allow use of median absolute deviation to mask the map data

- regularization technique to reduce unphysical spatial variation

- add Bayesian fitting via lmfit/emcee

#### _Models_

 :heavy_check_mark: Add Kosma-Tau models

 :heavy_check_mark: Add full set of Wolfire-Kaufman 2020-1 models

 :heavy_check_mark: Allow user-added models

 - Add functionality to compare models from different ModelSets

#### _Plotting_

- H2 plotter

  :heavy_check_mark: plot maps of $T_{cold}, T_{hot}, N_{cold}, N_{hot}, N_{total}$, and $OPR$

  :heavy_check_mark: for maps, have mouse-click show pop-up excitation diagram for that pixel

#### _Testing_

- We seriously need more test code and an automated test suite

#### _Documentation_

- Always can use updating

- More notebook examples

#### _Issues_

- See Issue list in github.  True bugs have highest priority
