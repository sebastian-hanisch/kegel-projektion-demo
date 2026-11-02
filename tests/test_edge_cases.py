"""Randfälle, an denen Projektionen erfahrungsgemäß brechen (DEMO-PLAYBOOK §12)."""

import numpy as np
import pytest

import kpj_constants as C
import kpj_evaluation as ev
import kpj_projection as P


@pytest.mark.parametrize("kind,par", [("l", None), ("q", None), ("s", 2), ("e", None), ("p", 0.3)])
def test_the_zero_vector_projects_to_zero_and_is_in_both_cones(kind, par):
    V = np.zeros((1, 3))
    assert np.abs(P.project(kind, par, V)).max() == 0.0
    assert P.in_cone(kind, par, V)[0] and P.in_polar(kind, par, V)[0]


@pytest.mark.parametrize("kind,par", [("l", None), ("q", None), ("e", None), ("p", 0.5)])
def test_points_on_the_boundary_stay_put(kind, par):
    boundary = {"l": [0.0, 1.0, 2.0], "q": [5.0, 3.0, 4.0], "e": [1.0, 1.0, float(np.e)], "p": [2.0, 8.0, 4.0]}[kind]
    V = np.array([boundary])
    assert np.abs(P.project(kind, par, V)[0] - V[0]).max() < 1e-9 and P.certificate(kind, par, V, P.project(kind, par, V), 1e-7)["ok"][0]


def test_lorentz_of_size_two_and_psd_of_size_one():
    out = P.proj_lorentz(np.array([[1.0, 3.0], [3.0, 1.0], [-3.0, 1.0]]))
    assert out[0] == pytest.approx([2.0, 2.0]) and out[1].tolist() == [3.0, 1.0] and out[2].tolist() == [0.0, 0.0]            # t = 1, v̄ = 3: ((3 + 1)/2)(1, 1) = (2, 2)
    assert P.proj_psd(np.array([[-2.0], [3.0]]), 1).reshape(-1).tolist() == [0.0, 3.0]


def test_extreme_ratios_of_the_exponential_cone_are_still_certified():
    rng = np.random.default_rng(0)
    for lo in (-60.0, 20.0):
        V = np.stack([lo * np.exp(0.5 * rng.normal(size=200)), np.exp(0.2 * rng.normal(size=200)), 10.0 ** rng.uniform(-4, 6, 200)], axis=1)
        assert P.certificate("e", None, V, P.proj_exp(V))["ok"].mean() >= 0.99, lo


def test_small_and_large_scales_scale_the_projection():
    rng = np.random.default_rng(1)
    V = 2.0 * rng.normal(size=(100, 3))
    for kind, par in (("e", None), ("p", 0.3), ("q", None)):
        base = P.project(kind, par, V)
        for lam in (1e-8, 1e8):
            assert np.abs(P.project(kind, par, lam * V) - lam * base).max() < 1e-6 * lam * (1 + np.abs(base).max()), (kind, lam)


def test_the_view_covers_every_slider_corner_for_every_cone():
    for kind in P.KINDS:
        for corner in (C.V_MIN, 0.0, C.V_MAX):
            s = ev.Settings(kind=kind, alpha=C.ALPHA_CHOICES[-1], vx=corner, vy=C.V_MAX, vz=C.V_MIN)
            view = ev.projection_view(s)
            assert view["case"] in {"im Kegel", "im Polarkegel", "auf dem Rand"} and np.isfinite(view["distance"]) and view["distance"] <= np.linalg.norm(view["v"]) + 1e-9
            assert ev.verdict_projection(s, view)[0] in {"success", "warning"}


def test_a_point_is_classified_into_exactly_one_case():
    s = ev.Settings(kind="q", vx=1.0, vy=3.0, vz=4.0)
    assert ev.projection_view(ev.Settings(kind="q", vx=5.0, vy=3.0, vz=4.0))["case"] == "im Kegel"
    assert ev.projection_view(ev.Settings(kind="q", vx=-5.0, vy=3.0, vz=4.0))["case"] == "im Polarkegel"
    assert ev.projection_view(s)["case"] == "auf dem Rand"
    assert ev.projection_view(ev.Settings(kind="l", vx=0.0, vy=0.0, vz=0.0))["case"] == "im Kegel"                  # die Null liegt im Kegel (und im Polarkegel); Kegel gewinnt


def test_the_table_has_one_row_per_cone_and_the_selected_alpha():
    rows = ev.projection_table(ev.Settings(alpha=0.5))
    assert [r["kind"] for r in rows] == list(P.KINDS) and rows[-1]["name"] == "Potenzkegel, a = 0,5" and all(np.isfinite(r["distance"]) for r in rows)


def test_the_experiments_run_at_the_smallest_and_largest_cloud_settings():
    for n in (8, 40):
        for stretch, angle in ((1.2, 0), (6.0, 90)):
            rows = ev.pocs_rows(ev.Settings(n=n, stretch=stretch, angle=angle, seed=1), eps_list=(0.1,))
            assert len(rows) == 4 and all(len(r["results"]) == 1 and r["results"][0]["iterations"] >= 1 for r in rows)
