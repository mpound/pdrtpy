"""
Compatibility shim — imports everything from pdrtpy.utils.

All symbols are still accessible here for external code that uses
``import pdrtpy.pdrutils as utils`` or ``from pdrtpy.pdrutils import X``.
Internal package code uses ``pdrtpy.utils`` directly.
"""

from pdrtpy.utils import *  # noqa: F403
