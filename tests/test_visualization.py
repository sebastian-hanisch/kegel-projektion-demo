"""Abbildungen: Achsen gesperrt, Spuren stimmen mit den Zahlen überein, Diagramme bauen an den Rändern."""

import numpy as np
import pytest

import kpj_constants as C
import kpj_evaluation as ev
import kpj_projection as P
import kpj_visualization as V


def _locked(fig):
    axes = [v for k, v in fig.layout.to_plotly_json().items() if k.startswith(("xaxis", "yaxis"))]
    return bool(axes) and all(ax.get("fixedrange") is True for ax in axes)


@pytest.mark.parametrize("kind", P.KINDS)
def test_the_cone_chart_has_surface_points_and_the_residual(kind):
    s = ev.Settings(kind=kind)
    view = ev.projection_view(s)
    fig = V.build_cone3d(kind, s.alpha, view["v"], view["p"])
    surfaces = [t for t in fig.data if t.type == "surface"]
    assert len(surfaces) == {"l": 3, "q": 1, "s": 1, "e": 1, "p": 2}[kind]
    points = {t.name: t for t in fig.data if t.type == "scatter3d"}
    assert list(points["v"].x) == [view["v"][0]] and list(points["P(v)"].z) == [view["p"][2]] and "Rest v − P(v)" in points
    assert fig.layout.scene.aspectmode == "cube" and fig.layout.scene.xaxis.range[1] >= abs(view["v"]).max()


def test_the_surface_points_lie_on_the_boundary_of_the_cone():
    for kind, alpha in (("q", 0.3), ("s", 0.3), ("e", 0.3), ("p", 0.3), ("p", 0.5)):
        for X, Y, Z in V._surface(kind, alpha, 5.0):
            Pts = np.stack([X.reshape(-1), Y.reshape(-1), Z.reshape(-1)], axis=1)
            par = ev.par_of(kind, alpha)
            assert P.in_cone(kind, par, Pts, 1e-6).all(), (kind, alpha)
            assert np.abs(Pts).max() <= 5.0 + 1e-9


def test_moreau_chart_colors_self_dual_cones_differently():
    rows = ev.moreau_rows(count=200)
    fig = V.build_moreau(rows)
    colors = list(fig.data[0].marker.color)
    assert _locked(fig) and colors[:3] == [C.COLOR_OK] * 3 and colors[3:] == [C.COLOR_BAD] * 3 and fig.layout.yaxis.type == "log" and all(y > 0 for y in fig.data[0].y)


def test_cost_chart_has_the_measured_curve_and_two_reference_slopes():
    res = ev.cost_rows(sizes=(4, 8, 16, 32))
    fig = V.build_cost(res)
    assert _locked(fig) and [t.name for t in fig.data][:3] == ["gemessen", "k³", "k²"] and fig.layout.xaxis.type == "log" and fig.data[3].type == "bar" and len(fig.data[3].x) == 1 + 2 + 2


def test_pocs_charts_have_one_line_per_cone_and_the_selected_history():
    rows = ev.pocs_rows(ev.Settings(n=8), eps_list=(0.1, 0.03))
    fig = V.build_pocs(rows)
    assert _locked(fig) and len([t for t in fig.data if t.mode == "lines+markers"]) == 4 and fig.layout.yaxis.type == "log" and list(fig.data[0].x) == ["10 %", "3 %"]
    rate = V.build_rate(rows, 0.03)
    assert _locked(rate) and len(rate.data) == 4 and all(len(t.x) == len(t.y) for t in rate.data) and all(y > 0 for t in rate.data for y in t.y)


def test_expcheck_chart_has_two_bar_series():
    rows = ev.exp_check_rows(count=100)
    fig = V.build_expcheck(rows)
    assert _locked(fig) and [t.name for t in fig.data] == ["Gitter + Maximum (erste Fassung)", "Wurzel von ⟨v, u × n⟩ = 0"] and list(fig.data[1].y) == [0, 0, 0, 0]
