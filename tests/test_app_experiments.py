"""Pfade der App, die der Rauchtest nicht erreicht: jeder Kegel, die fünf Experimente im Seiteninhalt, PDF, Eckfälle aller Regler zugleich (DEMO-PLAYBOOK §1d)."""

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

import kpj_constants as C
import kpj_projection as P

APP = str(Path(__file__).resolve().parent.parent / "app.py")


def _app(**state) -> AppTest:
    at = AppTest.from_file(APP, default_timeout=300)
    at.run()
    for key, value in state.items():
        widget = at.sidebar.selectbox if key == "kind_select" else at.sidebar.number_input if key == "seed_input" else at.sidebar.select_slider if key in ("alpha_select", "eps_select") else at.sidebar.slider
        widget(key=key).set_value(value)
    if state:
        at.run()
    assert not at.exception, [str(e) for e in at.exception]
    return at


def _texts(at):
    return " ".join(str(e.value) for e in list(at.success) + list(at.info) + list(at.warning) + list(at.error) + list(at.markdown))


def test_default_page_shows_the_metrics_and_all_experiment_verdicts():
    at = _app()
    values = {m.label: m.value for m in at.metric}
    assert values["Lage von v"] == "auf dem Rand" and values["Moreau-Probe"] == "besteht" and values["Projektion P(v)"] == "(0,023; 0,043; 0,074)" and values["Abstand ‖v − P(v)‖"] == "1,2215"
    text = _texts(at)
    for part in ("Der Punkt v = (1,000; 0,500; −0,500) liegt auf dem Rand des **Exponentialkegel**", "Der Punkt v = (1,000; 0,500; −0,500) liegt", "An je 2000 Zufallspunkten", "Lorentz-Kegel: ",
                 "Bei Schlupf 3 %", "Die Moreau-Probe prüft eine Projektion ohne Referenzlöser"):
        assert part in text, part


def test_the_five_experiments_the_limits_table_and_the_formulation_are_on_the_page():
    at = _app()
    headings = [s.value for s in at.subheader]
    assert [h.split(":")[0] for h in headings if h.startswith("🔬")] == [f"🔬 Experiment {i}" for i in range(1, 6)] and "🚧 Wo die Annahmen enden" in headings
    assert [e.label for e in at.expander] == ["So funktioniert es", "📐 Mathematische Formulierung"]
    assert len(at.dataframe) >= 5 and not at.error


def test_pdf_button_produces_a_download():
    at = _app()
    at.button(key="pdf_make").click().run()
    assert not at.exception
    assert at.session_state["pdf_made"][1][:4] == b"%PDF"


@pytest.mark.parametrize("kind", P.KINDS)
def test_every_cone_renders_without_error(kind):
    at = _app(kind_select=kind)
    assert not at.error
    assert any("Der Punkt v = " in s.value for s in list(at.success) + list(at.warning))


@pytest.mark.parametrize("alpha", C.ALPHA_CHOICES)
def test_every_exponent_of_the_power_cone_renders(alpha):
    at = _app(kind_select="p", alpha_select=alpha)
    assert not at.error and "Potenzkegel" in _texts(at)


@pytest.mark.parametrize("eps", C.EPS_CHOICES)
def test_every_slack_renders(eps):
    at = _app(eps_select=eps)
    assert not at.error and not at.exception


@pytest.mark.parametrize("edge", ["min", "max"])
def test_all_controls_at_their_minimum_or_maximum_together(edge):
    pick = (lambda lo, hi: lo) if edge == "min" else (lambda lo, hi: hi)
    state = {"vx_slider": pick(C.V_MIN, C.V_MAX), "vy_slider": pick(C.V_MIN, C.V_MAX), "vz_slider": pick(C.V_MIN, C.V_MAX), "n_slider": pick(C.N_MIN, C.N_MAX),
             "stretch_slider": pick(C.STRETCH_MIN, C.STRETCH_MAX), "angle_slider": pick(C.ANGLE_MIN, C.ANGLE_MAX), "eps_select": pick(C.EPS_CHOICES[0], C.EPS_CHOICES[-1])}
    at = _app(**state)
    assert not at.exception and not at.error


def test_the_point_corners_work_in_every_cone():
    for kind in P.KINDS:
        at = _app(kind_select=kind, vx_slider=C.V_MAX, vy_slider=C.V_MIN, vz_slider=C.V_MAX)
        assert not at.error, kind
