Spatial-Domain Regularization: Smoothing Map Fits Without Blurring Real Edges
==============================================================================

The :mod:`~pdrtpy.regularization` module implements spatial-domain regularization for
map-based fits. It is tool-agnostic: it operates only on plain 2-D parameter maps, a
shared validity mask, and caller-supplied objective/gradient/proximal callables, so it
can be reused by any fitting tool that determines an independent value at every spatial
pixel — currently :class:`~pdrtpy.tool.lineratiofit.LineRatioFit` via
:meth:`~pdrtpy.tool.lineratiofit.LineRatioFit.regularize`.

Neighboring pixels in a map fit can sometimes land on two solutions with nearly
identical :math:`\chi^2` but very different physical values — a degenerate valley in
the fit's phase space, not noise. This is unphysical: real spatial structure (e.g. a
PDR ionization front) spans many contiguous pixels, not one. Spatial regularization
adds a penalty term that discourages such pixel-to-pixel disagreement without erasing
genuine multi-pixel structure.

Two penalty types are provided:

- **Total Variation** (:class:`~pdrtpy.regularization.total_variation.TotalVariationRegularizer`,
  the recommended default) penalizes the *absolute* difference between neighboring
  pixels. Because this cost does not grow faster for a sharp jump than for the same
  jump spread over many pixels, it suppresses isolated single-pixel oscillation while
  leaving genuine, spatially-coherent edges largely intact. See ``docs/chambolle.md``
  for a plain-language walkthrough of the dual-projection algorithm used internally.
- **Tikhonov** (:class:`~pdrtpy.regularization.tikhonov.TikhonovRegularizer`) penalizes
  the *squared* difference between neighbors. It is retained for comparison but not
  recommended: because a large jump costs disproportionately more than the same change
  spread out, it tends to blur real multi-pixel edges along with pixel-to-pixel noise.

Both are solved via a shared proximal-gradient (:func:`~pdrtpy.regularization.base.fista`)
driver — the Fast Iterative Shrinkage-Thresholding Algorithm, with backtracking line
search. See ``docs/ista.md`` for a plain-language introduction to the shrinkage/
thresholding idea this generalizes.

See ``goals/regularization_design_options.md`` in the repository for the full design
rationale, including why Total Variation rather than Tikhonov was chosen as the
recommended penalty.

--------------

.. automodule:: pdrtpy.regularization
   :members:
   :undoc-members:
   :show-inheritance:

Neighbor Graphs and the FISTA Solver
-------------------------------------

.. automodule:: pdrtpy.regularization.base
   :members:
   :undoc-members:
   :show-inheritance:

Total Variation Regularization
-------------------------------

.. automodule:: pdrtpy.regularization.total_variation
   :members:
   :undoc-members:
   :show-inheritance:

Tikhonov Regularization
------------------------

.. automodule:: pdrtpy.regularization.tikhonov
   :members:
   :undoc-members:
   :show-inheritance:
