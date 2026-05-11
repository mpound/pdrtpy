import astropy.units as u
import numpy as np
import pytest
from pdrtpy.molecule import C13O, C13O18, CO, CO18, H2, CHplus

# ---------------------------------------------------------------------------
# Parametrized tests that apply to all table-based molecules (CO family + CH+)
# ---------------------------------------------------------------------------

TABLE_MOLECULES = [CO, C13O, CO18, C13O18, CHplus]


@pytest.mark.parametrize("cls", TABLE_MOLECULES)
def test_partition_function_returns_finite_array(cls):
    mol = cls()
    T = [10, 50, 100] * u.K
    Q = mol.partition_function(T)
    assert len(Q) == 3
    assert np.all(np.isfinite(Q))


@pytest.mark.parametrize("cls", TABLE_MOLECULES)
def test_partition_function_increases_with_temperature(cls):
    mol = cls()
    T = [10, 50, 100, 200] * u.K
    Q = mol.partition_function(T)
    assert np.all(np.diff(Q) > 0)


@pytest.mark.parametrize("cls", TABLE_MOLECULES)
def test_partition_function_exceeds_max_warns(cls):
    mol = cls()
    T = [1e7] * u.K
    with pytest.warns(UserWarning, match="maximum partition function temperature"):
        mol.partition_function(T)


@pytest.mark.parametrize("cls", TABLE_MOLECULES)
def test_line_ids_is_list(cls):
    mol = cls()
    ids = mol.line_ids
    assert isinstance(ids, list)
    assert len(ids) > 0


# ---------------------------------------------------------------------------
# TestH2 — uses an analytic formula, not the table path
# ---------------------------------------------------------------------------


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
        t = 500.0
        expected = 0.0247 * t / (1.0 - np.exp(-6000.0 / t))
        Q = self.mol.partition_function([t] * u.K)
        assert Q[0] == pytest.approx(expected, rel=1e-6)

    def test_transition_data_has_index(self):
        assert "Line" in self.mol.transition_data.colnames


# ---------------------------------------------------------------------------
# TestCO — spot-checks unique to 12CO
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# TestC13O
# ---------------------------------------------------------------------------


class TestC13O:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.mol = C13O()

    def test_name(self):
        assert "CO" in self.mol.name

    def test_canonical_opr(self):
        assert self.mol.canonical_opr == 1.0

    def test_opr_can_vary(self):
        assert self.mol.opr_can_vary is False


# ---------------------------------------------------------------------------
# TestCO18
# ---------------------------------------------------------------------------


class TestCO18:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.mol = CO18()

    def test_name(self):
        assert "18" in self.mol.name

    def test_canonical_opr(self):
        assert self.mol.canonical_opr == 1.0

    def test_opr_can_vary(self):
        assert self.mol.opr_can_vary is False


# ---------------------------------------------------------------------------
# TestC13O18
# ---------------------------------------------------------------------------


class TestC13O18:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.mol = C13O18()

    def test_name(self):
        assert "13" in self.mol.name
        assert "18" in self.mol.name

    def test_canonical_opr(self):
        assert self.mol.canonical_opr == 1.0

    def test_opr_can_vary(self):
        assert self.mol.opr_can_vary is False


# ---------------------------------------------------------------------------
# TestCHplus
# ---------------------------------------------------------------------------


class TestCHplus:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.mol = CHplus()

    def test_name(self):
        assert "CH" in self.mol.name

    def test_canonical_opr(self):
        assert self.mol.canonical_opr == 3.0

    def test_opr_can_vary(self):
        assert self.mol.opr_can_vary is True


# ---------------------------------------------------------------------------
# Cross-molecule consistency checks
# ---------------------------------------------------------------------------


class TestIsotopologueConsistency:
    """Verify that isotopologues use different partition function tables."""

    @pytest.mark.parametrize(
        "cls_a,cls_b",
        [
            (CO, C13O),
            (CO, CO18),
            (CO, C13O18),
            (C13O, CO18),
            (C13O, C13O18),
            (CO18, C13O18),
        ],
    )
    def test_partition_functions_differ(self, cls_a, cls_b):
        T = [100, 200] * u.K
        Q_a = cls_a().partition_function(T)
        Q_b = cls_b().partition_function(T)
        assert not np.allclose(Q_a, Q_b)
