Analysis Tools: Fit models to data
==================================

The :mod:`~pdrtpy.tool` module contains the analysis tools in the `PDR Toolbox <http://dustem.astro.umd.edu>`_.  All tools are derived from :class:`~pdrtpy.tool.toolbase.ToolBase`.

For examples how to use `LineRatioFit`, see the notebooks 
`PDRT_Example_Find_n_G0_Single_Pixel.ipynb <https://github.com/mpound/pdrtpy-nb/blob/master/notebooks/PDRT_Example_Find_n_G0_Single_Pixel.ipynb>`_ 
and 
`PDRT_Example_Make_n_G0_maps.ipynb <https://github.com/mpound/pdrtpy-nb/blob/master/notebooks/PDRT_Example_Make_n_G0_maps.ipynb>`_.

--------------

.. automodule:: pdrtpy.tool
   :members:
   :undoc-members:
   :show-inheritance:



ToolBase
--------

The base class of all tools.  Tools have a built-in plotter and a :meth:`run` method both of which subclasses must define.

.. automodule:: pdrtpy.tool.toolbase
   :members:
   :undoc-members:
   :show-inheritance:


H2Excitation
------------

Tool for fitting temperatures in :math:`H_2` excitation diagrams.  A two temperature model is assumed, :math:`T_{warm}` and :math:`T_{cold}`.

.. automodule:: pdrtpy.tool.h2excitation
   :members:
   :undoc-members:
   :show-inheritance:

LineRatioFit
------------

Tool for determining photodissociation region external radiation field and density (commonly known as :math:`G_0` and :math:`n`) from measured spectral line intensity ratios.  

.. automodule:: pdrtpy.tool.lineratiofit
   :members:
   :undoc-members:
   :show-inheritance:

