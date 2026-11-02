"""Orakel (DEMO-PLAYBOOK §12, ORACLE-PLAYBOOK §5): die Projektionen gegen einen anderen Rechenweg.

- cvxpy mit Clarabel löst `min ‖x − v‖² mit x ∈ K` für alle fünf Kegel; verglichen wird der **Abstand** (der Punkt selbst ist am flachen Optimum nur auf etwa 10⁻⁵ bestimmt, der Abstand auf 10⁻⁹)
- die Moreau-Probe als Zertifikat braucht keinen zweiten Löser; sie wird an den Punkten der Referenz gegengeprüft (die Referenz besteht sie, bis auf die Toleranz)
- Eigenschaften, die ein Fehler in beiden Rechenwegen nicht überlebt (Idempotenz, Nichtexpansivität, Spiegelformel bei selbstdualen Kegeln)

Fehlt cvxpy im Test-venv, überspringen sich diese Tests (pytest.importorskip), sie raten nie."""

import numpy as np
import pytest

import kpj_projection as P


def _cvxpy():
    cp = pytest.importorskip("cvxpy")
    if "CLARABEL" not in cp.installed_solvers():
        pytest.skip("Clarabel fehlt")
    return cp, dict(solver=cp.CLARABEL, tol_gap_abs=1e-10, tol_gap_rel=1e-10, tol_feas=1e-10, max_iter=500)


def _reference(cp, kw, kind, par, v):
    if kind == "l":
        x = cp.Variable(len(v))
        cons = [x >= 0]
    elif kind == "q":
        x = cp.Variable(len(v))
        cons = [cp.SOC(x[0], x[1:])]
    elif kind == "s":
        X = cp.Variable((2, 2), symmetric=True)
        cp.Problem(cp.Minimize(cp.sum_squares(X - P.smat_stack(v[None, :], 2)[0])), [X >> 0]).solve(**kw)
        return P.svec_stack(np.array(X.value)[None, :, :])[0]
    elif kind == "e":
        x = cp.Variable(3)
        cons = [cp.constraints.ExpCone(x[0], x[1], x[2])]
    else:
        x = cp.Variable(3)
        cons = [cp.constraints.PowCone3D(x[0], x[1], x[2], par)]
    cp.Problem(cp.Minimize(cp.sum_squares(x - v)), cons).solve(**kw)
    return np.array(x.value).reshape(-1)


@pytest.mark.filterwarnings("ignore:Solution may be inaccurate")
@pytest.mark.parametrize("kind,par", [("l", None), ("q", None), ("s", 2), ("e", None), ("p", 0.3), ("p", 0.5), ("p", 0.75)])
def test_the_distance_to_the_cone_equals_the_cvxpy_distance(kind, par):
    cp, kw = _cvxpy()
    rng = np.random.default_rng(10)
    V = np.concatenate([2.0 * rng.normal(size=(60, 3)), 0.3 * rng.normal(size=(20, 3)), 8.0 * rng.normal(size=(20, 3))])
    out = P.project(kind, par, V)
    ref = np.array([_reference(cp, kw, kind, par, v) for v in V])
    excess = np.linalg.norm(V - out, axis=1) - np.linalg.norm(V - ref, axis=1)
    assert excess.max() < 1e-7, (kind, par)                                       # unser Punkt ist nie weiter entfernt als die Referenz (bis auf deren Genauigkeit)
    assert excess.min() > -1e-3                                                   # und auch nicht „zu nah“ (das hieße: außerhalb des Kegels)
    assert P.in_cone(kind, par, out, 1e-7).all()


@pytest.mark.filterwarnings("ignore:Solution may be inaccurate")
def test_the_exponential_cone_with_extreme_ratios_equals_the_cvxpy_distance():
    cp, kw = _cvxpy()
    rng = np.random.default_rng(11)
    for lo, zs in ((-60.0, 0.1), (20.0, 1e6)):
        V = np.stack([lo * np.exp(0.5 * rng.normal(size=60)), np.exp(0.2 * rng.normal(size=60)), zs * np.exp(rng.normal(size=60))], axis=1)
        out = P.proj_exp(V)
        ref = np.array([_reference(cp, kw, "e", None, v) for v in V])
        assert (np.linalg.norm(V - out, axis=1) - np.linalg.norm(V - ref, axis=1)).max() < 1e-7, lo


@pytest.mark.filterwarnings("ignore:Solution may be inaccurate")
def test_the_power_cone_at_extreme_exponents_equals_the_cvxpy_distance_even_if_the_certificate_is_tight():
    """Bei a = 0,1 und 0,9 (Spitze am Rand, Werte über viele Größenordnungen) stimmt der Abstand auf 10⁻⁷; die Moreau-Probe fällt dort an einzelnen Punkten knapp durch (siehe README)."""
    cp, kw = _cvxpy()
    rng = np.random.default_rng(12)
    V = np.concatenate([rng.normal(size=(60, 3)), np.exp(2 * rng.normal(size=(40, 3))) * rng.choice([-1, 1], size=(40, 3))])
    for a in (0.1, 0.9):
        out = P.proj_pow(V, a)
        ref = np.array([_reference(cp, kw, "p", a, v) for v in V])
        assert (np.linalg.norm(V - out, axis=1) - np.linalg.norm(V - ref, axis=1)).max() < 1e-7, a
        assert P.certificate("p", a, V, out)["ok"].mean() > 0.95


@pytest.mark.filterwarnings("ignore:Solution may be inaccurate")
def test_the_reference_passes_the_moreau_probe_too():
    cp, kw = _cvxpy()
    rng = np.random.default_rng(13)
    V = 2.0 * rng.normal(size=(40, 3))
    ref = np.array([_reference(cp, kw, "e", None, v) for v in V])
    assert P.certificate("e", None, V, ref, tol=1e-4)["ok"].mean() > 0.95
