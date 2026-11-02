"""Abwechselnde Projektionen (kpj_pocs): Handbeispiel, Eigenschaften, die vier Kugel-Programme."""

import numpy as np
import pytest

import kegel_kern as K
import kpj_barrier as B
import kpj_evaluation as ev
import kpj_forms as F
import kpj_pocs as PO
import kpj_projection as P
from kpj_scenario import make_cloud


def test_the_lorentz_hand_program_halves_the_distance_each_step(tiny_cones):
    """min t mit (t; 3, 4) ∈ Lorentz: Start t = 0, y = (t; 3, 4), P(y) = ((5 + t)/2)(1; 3/5, 4/5): t_{k+1} = (5 + t_k)/2, also t_k = 5 − 5 · 2^(−k), und ‖y − P(y)‖ = (5 − t) / √2: der Abstand halbiert sich."""
    prob, opt = tiny_cones["soc"]
    r = PO.alternate(prob, gamma=6.0, max_iter=60, tol=1e-9, x_start=np.array([0.0]))
    assert r.converged and opt == 5.0
    assert r.history[0] == pytest.approx(5.0 / np.sqrt(2.0)) and r.history[1] == pytest.approx(2.5 / np.sqrt(2.0)) and r.history[2] == pytest.approx(1.25 / np.sqrt(2.0))
    assert all(b == pytest.approx(a / 2.0, rel=1e-9) for a, b in zip(r.history, r.history[1:6]))
    assert r.value == pytest.approx(5.0, abs=1e-6)


def test_the_bound_gamma_clamps_the_objective(tiny_cones):
    """Mit γ unter dem Optimum (γ = 4 < 5) gibt es keinen Durchschnitt: der Abstand bleibt positiv, nicht gelöst."""
    prob, _ = tiny_cones["soc"]
    r = PO.alternate(prob, gamma=4.0, max_iter=200, tol=1e-9, x_start=np.array([10.0]))
    assert not r.converged and r.distance > 0.1 and r.value <= 4.0 + 1e-12


def test_the_orthant_program_is_solved_in_one_step(tiny_cones):
    prob, _ = tiny_cones["lp"]
    r = PO.alternate(prob, gamma=1.5, max_iter=10, tol=1e-12)
    assert r.converged and r.iterations == 1 and r.value == pytest.approx(1.5)


def test_the_psd_hand_program_converges_to_the_optimum(tiny_cones):
    prob, opt = tiny_cones["sdp"]
    r = PO.alternate(prob, gamma=1.2, max_iter=500, tol=1e-9, x_start=np.array([0.0]))
    assert r.converged and opt - 1e-6 <= r.value <= 1.2


def test_the_history_distance_is_non_increasing_for_the_ball_programs():
    """Der Abstand ‖y − s‖ der beiden Mengen fällt nicht (Fejér-Monotonie der abwechselnden Projektionen)."""
    pts = make_cloud(8, 4.0, 45, 35).points
    for prob in (F.build_soc_ball(pts), F.build_psd_ball(pts), F.build_lp_ball(pts, 3.0)):
        rstar = (K.solve_barrier(prob, tol=1e-11).value if hasattr(prob, "cones") else B.solve(prob, tol=1e-11).value)
        r = PO.alternate(prob, rstar * 1.03, max_iter=300, tol=1e-9, x_start=ev.start_point(prob, pts, rstar))
        assert all(b <= a * (1 + 1e-9) + 1e-12 for a, b in zip(r.history, r.history[1:]))


def test_the_result_satisfies_the_bound_and_is_nearly_in_the_cone():
    pts = make_cloud(8, 4.0, 45, 35).points
    prob = F.build_soc_ball(pts)
    rstar = K.solve_barrier(prob, tol=1e-12).value
    for eps in (0.1, 0.01):
        r = PO.alternate(prob, rstar * (1 + eps), max_iter=4000, tol=1e-7, x_start=ev.start_point(prob, pts, rstar))
        assert r.converged and r.value <= rstar * (1 + eps) + 1e-9 and r.value >= rstar - 1e-6        # der Wert liegt zwischen R* und der Schranke


def test_spec_of_groups_equal_cones_and_keeps_programs_groups():
    pts = make_cloud(6, 2.0, 10, 1).points
    assert PO.spec_of(F.build_soc_ball(pts)) == [("q", 3, 6)] and PO.spec_of(F.build_psd_ball(pts)) == [("s", 3, 6)]
    assert PO.spec_of(F.build_lp_ball(pts, 3.0)) == [("p", 1.0 / 3.0, 12), ("l", None, 6)] and PO.spec_of(F.build_lse_ball(pts, 2.0))[0] == ("e", None, 24)


def test_project_spec_projects_block_by_block():
    spec = [("l", None, 2), ("q", 3, 1), ("s", 2, 1), ("e", None, 1), ("p", 0.5, 1)]
    s = np.array([-1.0, 2.0, 1.0, 3.0, 4.0, 2.0, 0.0, -1.0, 0.0, 1.0, 2.0, 2.0, 8.0, 4.1])
    out = PO.project_spec(spec, s)
    assert out[:2].tolist() == [0.0, 2.0] and out[2:5] == pytest.approx([3.0, 1.8, 2.4]) and out[5:8] == pytest.approx([2.0, 0.0, 0.0])
    assert out[8:11].tolist() == [0.0, 1.0, 2.0] and len(out) == len(s)
    assert P.pow_member(out[11:14][None, :], 0.5, 1e-9)[0] and not P.pow_member(s[11:14][None, :], 0.5)[0]       # der letzte Block liegt nach der Projektion im Potenzkegel, davor nicht
