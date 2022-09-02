Analysis Tools: Fit models to data
==================================

The :mod:`~pdrtpy.tool` module contains the analysis tools in the `PDR Toolbox <http://dustem.astro.umd.edu>`_.  All tools are derived from :class:`~pdrtpy.tool.toolbase.ToolBase`.

For examples how to use `LineRatioFit`, see the notebooks 
`PDRT_Example_Find_n_G0_Single_Pixel.ipynb <https://github.com/mpound/pdrtpy-nb/blob/master/notebooks/PDRT_Example_Find_n_G0_Single_Pixel.ipynb>`_ 
and 
`PDRT_Example_Make_n_G0_maps.ipynb <https://github.com/mpound/pdrtpy-nb/blob/master/notebooks/PDRT_Example_Make_n_G0_maps.ipynb>`_.

For an example how to use `H2ExcitationFit` and `ExcitationPlot` see the notebook `PDRT_Example_H2_Excitation.ipynb <https://github.com/mpound/pdrtpy-nb/blob/master/notebooks/PDRT_Example_H2_Excitation.ipynb>`_.

--------------

.. automodule:: pdrtpy.tool
   :members:
   :undoc-members:
   :show-inheritance:



ToolBase
--------

The base class of all tools.  Tools have a :meth:`run` method both of which subclasses must define.

.. automodule:: pdrtpy.tool.toolbase
   :members:
   :undoc-members:
   :show-inheritance:


Excitation Diagram Fitting
--------------------------

:class:`~pdrtpy.tool.h2excitation.H2ExcitationFit` is a tool for fitting temperature, column density, and ortho-to-para ratio in :math:`H_2` excitation diagrams.  A two temperature model is assumed, and the fit will find :math:`T_{hot}, T_{cold}, N_{hot}(H_2), N_{cold}(H_2),` and optionally `OPR`.  The base class :class:`~pdrtpy.tool.h2excitation.ExcitationFit` can be used to create a tool to fit a different molecule.

.. automodule:: pdrtpy.tool.h2excitation
   :members:
   :undoc-members:
   :show-inheritance:

LineRatioFit
------------

:class:`~pdrtpy.tool.lineratiofit.LineRatioFit` is a tool for determining photodissociation region external radiation field and hydrogen nucleus density (commonly given as :math:`G_0` and :math:`n`) from measured spectral line intensity ratios.  

.. automodule:: pdrtpy.tool.lineratiofit
   :members:
   :undoc-members:
   :show-inheritance:

FitMap
------
When fitting either single pixels or spatial maps, the fit results are stored per pixel in an :class:`~astropy.nddata.NDData` object that contains :class:`~lmfit.model.ModelResult` 
objects for :class:`~pdrtpy.tool.h2excitation.H2ExcitationFit` or :class:`~lmfit.minimizer.MinimizerResult` objects for :class:`~pdrtpy.tool.lineratiofit.LineRatioFit`.  The user can thus examine in detail the fit at any pixel.

.. automodule:: pdrtpy.tool.fitmap
   :members:
   :undoc-members:
   :show-inheritance:
