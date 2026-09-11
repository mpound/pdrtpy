"""Spatial-domain regularization for map-based fits.

Tool-agnostic: operates on plain 2-D parameter maps and a validity mask, so
it is usable by :class:`~pdrtpy.tool.lineratiofit.LineRatioFit` today and,
eventually, :class:`~pdrtpy.tool.excitation.ExcitationFit`.
See ``goals/regularization_design_options.md`` for the design rationale.
"""

from .base import Regularizer, fista, neighbor_graph
from .tikhonov import TikhonovRegularizer
from .total_variation import TotalVariationRegularizer

__all__ = [
    "Regularizer",
    "neighbor_graph",
    "fista",
    "TikhonovRegularizer",
    "TotalVariationRegularizer",
]
