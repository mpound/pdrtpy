Plotting Tools: Display models and data
=======================================

The :mod:`~pdrtpy.plot` module provides mechanisms for plotting models, observations, and model fits.

The :class:`~pdrtpy.plot.modelplot.ModelPlot` class can be used plotting models and observations without any :math:`\chi^2` fitting.
An example notebook for using `ModelPlot` is
`PDRT_Example_ModelPlotting.ipynb <https://github.com/mpound/pdrtpy-nb/blob/master/notebooks/PDRT_Example_ModelPlotting.ipynb>`_  .

Some classes are paired with analysis tools in the :mod:`~pdrtpy.tool` module.  :class:`~pdrtpy.plot.lineratioplot.LineRatioPlot` which is used to plot the results of :class:`~pdrtpy.tool.lineratiofit.LineRatioFit`, and :class:`~pdrtpy.plot.excitationplot.H2ExcitationPlot` that is used in :class:`~pdrtpy.tool.excitation.H2Excitation`.  All plot classes are derived from :class:`~pdrtpy.plot.plotbase.PlotBase`.

.. include:: plotkeywords.rst

--------------

.. automodule:: pdrtpy.plot
   :members:
   :undoc-members:
   :show-inheritance:


PlotBase
--------

.. automodule:: pdrtpy.plot.plotbase
   :members:
   :undoc-members:
   :show-inheritance:

ExcitationPlot
----------------

.. automodule:: pdrtpy.plot.excitationplot
   :members:
   :undoc-members:
   :show-inheritance:

LineRatioPlot
-------------

.. automodule:: pdrtpy.plot.lineratioplot
   :members:
   :undoc-members:
   :show-inheritance:

ModelPlot
-------------

.. automodule:: pdrtpy.plot.modelplot
   :members:
   :undoc-members:
   :show-inheritance:
