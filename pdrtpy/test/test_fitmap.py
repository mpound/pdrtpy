"""Tests for FitMap class"""
import numpy as np
import pytest
from astropy.wcs import WCS

from pdrtpy.tool.fitmap import FitMap


@pytest.fixture
def simple_wcs():
    """A minimal 2D celestial WCS"""
    w = WCS(naxis=2)
    w.wcs.crpix = [5.0, 5.0]
    w.wcs.cdelt = [0.01, 0.01]
    w.wcs.crval = [10.0, -70.0]
    w.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    w._naxis = [10, 10]
    return w


@pytest.fixture
def fitmap_with_wcs(simple_wcs):
    data = np.empty((10, 10), dtype=object)
    data.fill("result")
    return FitMap(data, wcs=simple_wcs, name="test_map")


@pytest.fixture
def fitmap_no_wcs():
    data = np.array([1.0, 2.0, 3.0])
    return FitMap(data, name="no_wcs")


class TestFitMapBasics:
    def test_name(self, fitmap_with_wcs):
        assert fitmap_with_wcs.name == "test_map"

    def test_name_none(self, fitmap_no_wcs):
        assert fitmap_no_wcs.name == "no_wcs"

    def test_getitem(self, fitmap_with_wcs):
        assert fitmap_with_wcs[0, 0] == "result"

    def test_getitem_1d(self, fitmap_no_wcs):
        assert fitmap_no_wcs[0] == 1.0
        assert fitmap_no_wcs[2] == 3.0

    def test_scalar_data_wrapped(self):
        """Scalar data should be wrapped in a 1-element array"""
        fm = FitMap(np.float64(5.0), name="scalar")
        assert fm[0] == 5.0

    def test_debug_flag(self, capsys, simple_wcs):
        """debug=True should not raise, just print"""
        data = np.array([1.0, 2.0])
        # debug just prints, should not raise
        FitMap(data, name="debug_test", debug=False)


class TestFitMapCoordinates:
    def test_get_world(self, fitmap_with_wcs):
        world = fitmap_with_wcs.get_world(4, 4)
        assert len(world) == 2
        # world coords near crval at crpix-1 offset
        assert isinstance(world[0], float)
        assert isinstance(world[1], float)

    def test_get_world_center(self, fitmap_with_wcs):
        """Pixel at crpix-1 should give world coords near crval"""
        # crpix is 1-indexed in FITS but 0-indexed here
        world = fitmap_with_wcs.get_world(4, 4)
        # crval = [10.0, -70.0], crpix = [5, 5], so pixel (4,4) = crval
        assert abs(world[0] - 10.0) < 0.01
        assert abs(world[1] - (-70.0)) < 0.01

    def test_get_pixel(self, fitmap_with_wcs):
        """Round-trip: pixel -> world -> pixel should return original pixel"""
        world = fitmap_with_wcs.get_world(4, 4)
        pixel = fitmap_with_wcs.get_pixel(world[0], world[1])
        assert pixel[0] == 4
        assert pixel[1] == 4

    def test_get_pixel_returns_ints(self, fitmap_with_wcs):
        pixel = fitmap_with_wcs.get_pixel(10.0, -70.0)
        assert isinstance(pixel[0], (int, np.integer))
        assert isinstance(pixel[1], (int, np.integer))

    def test_get_skycoord(self, fitmap_with_wcs):
        """get_skycoord should return a SkyCoord"""
        from astropy.coordinates import SkyCoord

        coord = fitmap_with_wcs.get_skycoord(4, 4)
        assert coord is not None

    def test_get_world_no_wcs_raises(self, fitmap_no_wcs):
        with pytest.raises(Exception, match="No wcs"):
            fitmap_no_wcs.get_world(0, 0)

    def test_get_pixel_no_wcs_raises(self, fitmap_no_wcs):
        with pytest.raises(Exception, match="No wcs"):
            fitmap_no_wcs.get_pixel(0.0, 0.0)

    def test_get_pixel_from_coord_no_wcs_raises(self, fitmap_no_wcs):
        with pytest.raises(Exception, match="No wcs"):
            fitmap_no_wcs.get_pixel_from_coord((10.0, -70.0))
