"""Presets: Vollständigkeit, gültige Werte, Standard-Preset = Standardeinstellungen; Permalink verwirft ungültige Werte; die Zahlen der Hilfetexte."""

from pathlib import Path

import numpy as np
import pytest
from streamlit.testing.v1 import AppTest

import kpj_constants as C
import kpj_evaluation as ev
import kpj_presets as PR
import kpj_projection as P

APP = str(Path(__file__).resolve().parent.parent / "app.py")
DEFAULT = "Standardfall: Exponentialkegel, Punkt außen"


def test_every_preset_has_help_and_all_keys():
    assert set(C.PRESETS) == set(C.PRESET_HELP)
    for name, preset in C.PRESETS.items():
        assert set(preset) == set(PR.PRESET_KEYS), name
        assert C.PRESET_HELP[name]


def test_preset_values_are_inside_the_control_bounds():
    for preset in C.PRESETS.values():
        assert preset["kind"] in P.KINDS and preset["alpha"] in C.ALPHA_CHOICES and preset["eps"] in C.EPS_CHOICES
        assert all(C.V_MIN <= preset[k] <= C.V_MAX for k in ("vx", "vy", "vz"))
        assert C.N_MIN <= preset["n"] <= C.N_MAX and C.STRETCH_MIN <= preset["stretch"] <= C.STRETCH_MAX and C.ANGLE_MIN <= preset["angle"] <= C.ANGLE_MAX and 0 <= preset["seed"] <= C.SEED_MAX


def test_default_preset_equals_the_default_settings():
    assert ev.Settings(**C.PRESETS[DEFAULT]) == ev.Settings()


def test_setting_specs_cover_every_preset_key():
    assert set(PR.PRESET_KEYS.values()) == set(PR.SETTING_SPECS)


def _view(name):
    return ev.projection_view(ev.Settings(**C.PRESETS[name]))


def test_the_help_texts_name_what_the_presets_produce():
    v = _view(DEFAULT)
    assert v["case"] == "auf dem Rand" and v["certificate_ok"] and "auf dem Rand" in C.PRESET_HELP[DEFAULT]
    assert _view("Punkt im Kegel: nichts zu tun")["case"] == "im Kegel" and np.abs(_view("Punkt im Kegel: nichts zu tun")["r"]).max() == 0.0
    polar = _view("Punkt im Polarkegel: die Projektion ist 0")
    assert polar["case"] == "im Polarkegel" and np.abs(polar["p"]).max() == 0.0 and np.allclose(polar["r"], polar["v"])
    lor = _view("Lorentz-Kegel: die geschlossene Formel")
    assert lor["p"] == pytest.approx([3.0, 1.8, 2.4]) and "(3; 1,8; 2,4)" in C.PRESET_HELP["Lorentz-Kegel: die geschlossene Formel"]
    psd = _view("PSD-Kegel: Eigenwerte abschneiden")
    assert psd["p"] == pytest.approx([2.0, 0.0, 0.0]) and "diag(2, 0)" in C.PRESET_HELP["PSD-Kegel: Eigenwerte abschneiden"]
    pw = _view("Potenzkegel a = 0,3: nicht selbstdual")
    assert not pw["self_dual"] and pw["mirror_deviation"] > 0.3 and pw["certificate_ok"]


def test_permalink_takes_valid_values_and_drops_invalid_ones():
    at = AppTest.from_file(APP, default_timeout=300)
    at.query_params["kind"] = "q"
    at.query_params["n"] = "999"            # über der Grenze: wird begrenzt
    at.query_params["vx"] = "nan"           # keine endliche Zahl: bleibt Standard
    at.query_params["vy"] = "abc"           # keine Zahl: bleibt Standard
    at.query_params["vz"] = "99"            # über der Grenze: wird begrenzt
    at.query_params["alpha"] = "0.9"        # kein erlaubter Wert: bleibt Standard
    at.query_params["eps"] = "0.01"
    at.run()
    assert not at.exception
    assert at.sidebar.selectbox(key="kind_select").value == "q"
    assert at.sidebar.slider(key="n_slider").value == C.N_MAX
    assert at.sidebar.slider(key="vx_slider").value == C.DEFAULT_V[0] and at.sidebar.slider(key="vy_slider").value == C.DEFAULT_V[1] and at.sidebar.slider(key="vz_slider").value == C.V_MAX
    assert at.session_state["alpha_select"] == C.DEFAULT_ALPHA and at.session_state["eps_select"] == 0.01
    bad = AppTest.from_file(APP, default_timeout=300)
    bad.query_params["kind"] = "dreieck"
    bad.run()
    assert not bad.exception and bad.sidebar.selectbox(key="kind_select").value == C.DEFAULT_KIND
