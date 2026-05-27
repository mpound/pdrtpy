# Goal

Determine root cause in `ExcitationPlot.explore` that raises Exception in Cell 38 of the notebook
pdrtpy-nb/notebooks/PDRT_Example_H2_Excitation.ipynb

# Context

If `%matplotlib ipynb` is invoked in the notebook, calls to ``ExcitationPlot.explore`` can fail with the matplotlib error "ValueError: The passed figure is not managed by pyplot".   If `%matplotlib ipynb` is not invoked, the notebook runs, but the interactive nature of `ExcitationPlot.explore` is unavailable, so we need to keep the interactivity.

## Important

- The work must be done on a branch from master
- Claude should plan first and discuss the plan with Marc before implementing changes
