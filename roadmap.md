## Roadmap of Desired Functionality

These are the big-picture things we want to add before the money runs out!


#### _Radiation Field and Density Fitting_

&#10004; use lmfit minimization to allow off-grid minimum $\chi^2$

&#10004; full support for map-based pixel-by-pixel fitting

&#10004; performance enhancement for maps

- base code refactoring

- regularization technique to reduce unphysical spatial variation

#### _H2 Excitation Fitting_

&#10004; add full Roueff et al Table of H2 line parameters

- helpful methods to compute temperature and column density for users with insufficient data for full fitting

&#10004; full support for map-based pixel-by-pixel fitting

- allow use of median absolute deviation to mask the map data

- base code refactoring

- performance enhancement for maps

- Zanesse+2025 method for fitting A_V

- regularization technique to reduce unphysical spatial variation

- add Bayesian fitting via lmfit/emcee

#### _Models_

&#10004; Add Kosma-Tau models

&#10004; Add full set of Wolfire-Kaufman 2020-1 models

&#10004; Allow user-added models

&#10004; Alternate viewing angle models

 - metallicities

 - Add functionality to compare models from different ModelSets

#### _Plotting_

**H2 plotter:**

 &#10004; plot maps of $T_{cold}, T_{hot}, N_{cold}, N_{hot}, N_{total}$, and $OPR$

 &#10004; for maps, have mouse-click show pop-up excitation diagram for that pixel


#### _Testing_

 &#10004; We seriously need more test code and an automated test suite


#### _Documentation_

- Always can use updating

- More notebook examples

#### _Issues_

- See Issue list in github.  True bugs have highest priority
