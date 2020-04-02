
.. automodule:: pdrtpy.tool
   :members:
   :undoc-members:
   :show-inheritance:

The :mod:`~pdrtpy.tool` module contains the tools in the `PDR Toolbox <http://dustem.astro.umd.edu>`_.  All tools are derived from :class:`~pdrtpy.tool.toolbase.ToolBase`.

ToolBase
--------

The base class of all tools.  Tools have a built-in plotter and a :meth:`run` method both of which subclasses must define.

.. automodule:: pdrtpy.tool.toolbase
   :members:
   :undoc-members:
   :show-inheritance:


H2Excitation
------------

Tool for fitting :math:`H_2` excitation diagrams.

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

