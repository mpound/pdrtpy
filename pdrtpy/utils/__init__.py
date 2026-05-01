"""
pdrtpy.utils — utility subpackage for PDR Toolbox.

Submodules:
    units   — radiation field units, constants, and unit conversions
    paths   — path and I/O helpers
    fits    — FITS keyword utilities
    wcs     — WCS and image array utilities
    helpers — general helpers
"""

from pdrtpy.utils.fits import addkey, comment, dataminmax, firstkey, history, setkey, signature
from pdrtpy.utils.helpers import (
    _has_H2,
    _has_substring,
    _trim_all_to_H2,
    _trim_to_H2,
    is_even,
    is_image,
    is_odd,
    is_ratio,
    warn,
)
from pdrtpy.utils.paths import (
    _tablename,
    get_table,
    get_testdata,
    model_dir,
    now,
    root_dir,
    root_path,
    table_dir,
    testdata_dir,
)
from pdrtpy.utils.units import (
    LOGE,
    LN10,
    _CM,
    _CM2,
    _K,
    _KKMS,
    _OBS_UNIT_,
    _RFS_UNIT_,
    _rad_title,
    check_units,
    convert_if_necessary,
    convert_integrated_intensity,
    draine_unit,
    float_formatter,
    get_rad,
    habing_unit,
    is_rad,
    mathis_unit,
    toDraine,
    toHabing,
    toMathis,
    to,
    tocgs,
)
from pdrtpy.utils.wcs import (
    dropaxis,
    fliplabel,
    get_xy_from_wcs,
    has_single_axis,
    mask_union,
    rescale_axis_units,
    squeeze,
)

__all__ = [
    # units
    "LOGE",
    "LN10",
    "_CM",
    "_CM2",
    "_K",
    "_KKMS",
    "_OBS_UNIT_",
    "_RFS_UNIT_",
    "_rad_title",
    "check_units",
    "convert_if_necessary",
    "convert_integrated_intensity",
    "draine_unit",
    "float_formatter",
    "get_rad",
    "habing_unit",
    "is_rad",
    "mathis_unit",
    "toDraine",
    "toHabing",
    "toMathis",
    "to",
    "tocgs",
    # paths
    "_tablename",
    "get_table",
    "get_testdata",
    "model_dir",
    "now",
    "root_dir",
    "root_path",
    "table_dir",
    "testdata_dir",
    # fits
    "addkey",
    "comment",
    "dataminmax",
    "firstkey",
    "history",
    "setkey",
    "signature",
    # wcs
    "dropaxis",
    "fliplabel",
    "get_xy_from_wcs",
    "has_single_axis",
    "mask_union",
    "rescale_axis_units",
    "squeeze",
    # helpers
    "_has_H2",
    "_has_substring",
    "_trim_all_to_H2",
    "_trim_to_H2",
    "is_even",
    "is_image",
    "is_odd",
    "is_ratio",
    "warn",
]
