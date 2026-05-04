import astropy.units as u
import numpy as np
import pytest
from pdrtpy.molecule import C13O, CO, H2


class TestH2:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.mol = H2()

    def test_name(self):
        assert self.mol.name == "H_2"

    def test_canonical_opr(self):
        assert self.mol.canonical_opr == 3.0

    def test_opr_can_vary(self):
        assert self.mol.opr_can_vary is True

    def test_line_ids_is_list(self):
        ids = self.mol.line_ids
        assert isinstance(ids, list)
        assert len(ids) > 0

    def test_line_wavelengths(self):
        wl = self.mol.line_wavelengths
        assert len(wl) > 0
        assert wl.unit is not None

    def test_partition_function_returns_array(self):
        T = [100, 500, 1000] * u.K
        Q = self.mol.partition_function(T)
        assert len(Q) == 3
        assert np.all(np.isfinite(Q))

    def test_partition_function_increases_with_temperature(self):
        T = [100, 500, 1000, 2000] * u.K
        Q = self.mol.partition_function(T)
        assert np.all(np.diff(Q) > 0)

    def test_partition_function_known_value(self):
        # At T=500K, Herbst formula: 0.0247 * 500 / (1 - exp(-6000/500))
        t = 500.0
        expected = 0.0247 * t / (1.0 - np.exp(-6000.0 / t))
        T = [t] * u.K
        Q = self.mol.partition_function(T)
        assert Q[0] == pytest.approx(expected, rel=1e-6)

    def test_transition_data_has_index(self):
        assert "Line" in self.mol.transition_data.colnames


class TestCO:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.mol = CO()

    def test_name(self):
        assert "CO" in self.mol.name

    def test_canonical_opr(self):
        assert self.mol.canonical_opr == 1.0

    def test_opr_can_vary(self):
        assert self.mol.opr_can_vary is False

    def test_line_ids_is_list(self):
        assert isinstance(self.mol.line_ids, list)
        assert len(self.mol.line_ids) > 0

    def test_partition_function_returns_array(self):
        T = [10, 50, 100] * u.K
        Q = self.mol.partition_function(T)
        assert len(Q) == 3
        assert np.all(np.isfinite(Q))

    def test_partition_function_increases_with_temperature(self):
        T = [10, 50, 100, 200] * u.K
        Q = self.mol.partition_function(T)
        assert np.all(np.diff(Q) > 0)

    def test_partition_function_exceeds_max_warns(self):
        T = [1e7] * u.K
        with pytest.warns(UserWarning, match="maximum partition function temperature"):
            self.mol.partition_function(T)


class TestC13O:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.mol = C13O()

    def test_name(self):
        assert "CO" in self.mol.name

    def test_partition_function_returns_array(self):
        T = [10, 50, 100] * u.K
        Q = self.mol.partition_function(T)
        assert len(Q) == 3
        assert np.all(np.isfinite(Q))

    def test_partition_function_increases_with_temperature(self):
        T = [10, 50, 100, 200] * u.K
        Q = self.mol.partition_function(T)
        assert np.all(np.diff(Q) > 0)

    def test_partition_function_exceeds_max_warns(self):
        T = [1e7] * u.K
        with pytest.warns(UserWarning, match="maximum partition function temperature"):
            self.mol.partition_function(T)


class TestCOC13OConsistency:
    """CO and C13O use the same interpolation logic — verify structural consistency."""

    def test_both_have_partition_function(self):
        for cls in [CO, C13O]:
            mol = cls()
            T = [50, 100] * u.K
            Q = mol.partition_function(T)
            assert len(Q) == 2

    def test_co_and_c13o_partition_functions_differ(self):
        """CO and 13CO have different partition function tables."""
        co = CO()
        c13o = C13O()
        T = [100, 200] * u.K
        Q_co = co.partition_function(T)
        Q_c13o = c13o.partition_function(T)
        assert not np.allclose(Q_co, Q_c13o)
