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

Tool for determining photodissociation region external radiation field and particle density (commonly given as :math:`G_0` and :math:`n`) from measured spectral line intensity ratios.  

.. automodule:: pdrtpy.tool.lineratiofit
   :members:
   :undoc-members:
   :show-inheritance:

