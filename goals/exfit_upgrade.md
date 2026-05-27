## Goal

Determine best order in which to do two major changes to ExcitationFit

### Details

There are two major improvements to be made to ExcitationFit:

1. Data partitioning for a two-component fit is done performed in the method `_two_lines`.  This is fast but can fail for if there if there is even a single very bad data point.  The improvement would be to replace the fitting with true piece-wise regression to find the best partition/inflection point, which would also allow adding more than two components to the fit in the future.

2.  Improve the performance of fitting with ``ExcitationFit.run`` and
the methods it calls.  In particular, we need to speed up fitting for
multi-pixel images, similar to what was done for ``LineRatioFit``.


### Plan

- Please analyze the current code and suggest which order these improvements should be made and why.

- For piecewise regression fitting suggest at least two Python packages for the job and give pros and cons of each.

- Ask Marc questions if you want more information.

## Important

- The work must be done on a branch from master

- Claude should plan first and discuss the plan with Marc before implementing any changes
