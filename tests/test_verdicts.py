"""Verdikte an von Hand gesetzten Werten (DEMO-PLAYBOOK §12): jede Zahl im Text und jede Schwelle der Art steht hier."""

import numpy as np

import kpj_evaluation as ev


def test_formatting_helpers():
    assert ev.de(2.3456, 3) == "2,346" and ev.de(-1.0, 0) == "−1" and ev.sci(1.7e-10) == "1,7·10⁻¹⁰" and ev.sci(2.5e5) == "2,5·10⁵" and ev.sci(float("inf")) == "unendlich"
    assert ev.vec_text([1.0, -0.5, 2.25]) == "(1,000; −0,500; 2,250)" and ev.eps_text(0.03) == "3 %" and ev.eps_text(0.003) == "0,3 %" and ev.eps_text(0.1) == "10 %"
    assert ev.alpha_text(0.3) == "0,3" and ev.alpha_text(0.5) == "0,5" and ev.alpha_text(0.25) == "0,25" and ev.steps_text(1) == "1 Schritt" and ev.steps_text(216) == "216 Schritte"
    assert ev.cone_label("p", 0.3) == "Potenzkegel, a = 0,3" and ev.cone_label("q", 0.3) == "Lorentz-Kegel" and ev.par_of("s", 0.3) == 2 and ev.par_of("p", 0.3) == 0.3 and ev.par_of("l", 0.3) is None


def _view(**kw):
    base = {"v": np.array([1.0, 0.5, -0.5]), "p": np.array([0.023, 0.043, 0.074]), "r": np.array([0.977, 0.457, -0.574]), "distance": 1.2215, "inner": -6.9e-18, "case": "auf dem Rand",
            "certificate_ok": True, "in_cone": True, "in_polar": True, "orthogonal": True, "mirror_deviation": 0.457, "extra": "Randpunkt mit Verhältnis x/y = 0,5325", "self_dual": False}
    base.update(kw)
    return base


def test_projection_verdict_names_point_projection_distance_moreau_and_selfduality():
    s = ev.Settings()
    kind, text = ev.verdict_projection(s, _view())
    assert kind == "success"
    for part in ("v = (1,000; 0,500; −0,500) liegt auf dem Rand des **Exponentialkegel**", "Projektion ist **(0,023; 0,043; 0,074)**", "Abstand 1,2215", "Rest (0,977; 0,457; −0,574)",
                 "⟨p, v − p⟩ = −6,9·10⁻¹⁸", "p im Kegel: ja, v − p im Polarkegel: ja", "Randpunkt mit Verhältnis x/y = 0,5325.", "nicht selbstdual: P(v) − P(−v) verfehlt v um 0,457", "Die Moreau-Probe besteht."):
        assert part in text, part
    kind, text = ev.verdict_projection(ev.Settings(kind="q"), _view(self_dual=True, mirror_deviation=1e-16, extra=""))
    assert "Der Kegel ist selbstdual: v = P(v) − P(−v) stimmt auf 1,0·10⁻¹⁶" in text and "Lorentz-Kegel" in text
    kind, text = ev.verdict_projection(s, _view(certificate_ok=False, in_polar=False))
    assert kind == "warning" and "besteht nicht" in text and "Spitze" in text and "v − p im Polarkegel: nein" in text


def _table_rows():
    names = ["Orthant", "Lorentz-Kegel", "PSD-Kegel (2×2)", "Exponentialkegel", "Potenzkegel, a = 0,3"]
    cases = ["auf dem Rand", "auf dem Rand", "im Polarkegel", "auf dem Rand", "im Kegel"]
    return [{"name": n, "case": c, "certificate_ok": True, "self_dual": i < 3, "mirror_deviation": 1e-16 if i < 3 else 0.5 + i} for i, (n, c) in enumerate(zip(names, cases))]


def test_table_verdict_groups_the_cones_by_the_position_of_v():
    kind, text = ev.verdict_table(ev.Settings(), _table_rows())
    assert kind == "info"
    for part in ("im Kegel: Potenzkegel, a = 0,3", "im Polarkegel: PSD-Kegel (2×2)", "auf dem Rand: Orthant, Lorentz-Kegel, Exponentialkegel", "besteht bei 5 von 5 Projektionen", "auf 1,0·10⁻¹⁶",
                 "Exponentialkegel um 3,500, Potenzkegel, a = 0,3 um 4,500"):
        assert part in text, part


def _moreau_rows():
    return [{"cone": c, "kind": k, "self_dual": sd, "count": 2000, "mirror": m, "certificate": cert, "idempotent": 1e-15, "nonexpansive_violations": 0, "inside": 0.1, "polar": 0.1}
            for c, k, sd, m, cert in (("Orthant", "l", True, 0.0, 1.0), ("Lorentz-Kegel", "q", True, 1e-15, 1.0), ("PSD-Kegel (2×2)", "s", True, 2e-15, 1.0), ("Exponentialkegel", "e", False, 4.18, 0.9985),
                                      ("Potenzkegel, a = 0,3", "p", False, 2.06, 1.0))]


def test_moreau_verdict_contrasts_self_dual_and_the_other_cones():
    kind, text = ev.verdict_moreau(_moreau_rows())
    assert kind == "info"
    for part in ("An je 2000 Zufallspunkten", "auf 2,0·10⁻¹⁵", "Exponentialkegel verfehlt es v um bis zu 4,18, Potenzkegel, a = 0,3 verfehlt es v um bis zu 2,06", "bei 99,9 % bis 100,0 % der Punkte", "0 Verletzungen"):
        assert part in text, part


def test_cost_verdict_names_the_numbers():
    res = {"lorentz_ns": 43.0, "psd": [{"k": 4, "us_per_matrix": 2.8, "ns_per_coordinate": 280.0, "count": 1}, {"k": 32, "us_per_matrix": 151.0, "ns_per_coordinate": 286.0, "count": 1}], "slope": 2.1, "exp_us": 7.2, "pow_us": 14.0}
    kind, text = ev.verdict_cost(res)
    assert kind == "info"
    for part in ("43 ns je Koordinate", "2,8 µs je 4×4-Matrix und 151 µs je 32×32-Matrix", "Steigung 2,10 gegen k", "bei k = 32 7-mal teurer", "Exponentialkegel 7,2 µs und Potenzkegel 14,0 µs", "Zeiten hängen vom Rechner ab"):
        assert part in text, part


def _pocs_rows(failed=False):
    def res(eps, its, ok=True):
        return {"eps": eps, "iterations": its, "converged": ok, "seconds": 0.1, "distance": 1e-8, "value": 7.0, "history": [1.0], "history_step": 1, "per_step_us": 70.0}
    return [{"name": "Lorentz-Kegel", "kind": "q", "rstar": 7.0, "newton": 31, "nu": 16, "rows": 24, "variables": 3, "results": [res(0.03, 216), res(0.01, 216, not failed)]},
            {"name": "Exponentialkegel (log-sum-exp, β = 10)", "kind": "e", "rstar": 5.0, "newton": 43, "nu": 104, "rows": 100, "variables": 20, "results": [res(0.03, 1971), res(0.01, 1981)]}]


def test_pocs_verdict_compares_with_the_newton_steps_and_counts_failures():
    kind, text = ev.verdict_pocs(_pocs_rows(), 0.03)
    assert kind == "info"
    for part in ("Bei Schlupf 3 %", "Lorentz-Kegel: 216 Schritte", "Exponentialkegel (log-sum-exp, β = 10): 1971 Schritte", "31 bis 43 Newton-Schritte bis 10⁻⁹ (Lorentz-Kegel 31, Exponentialkegel 43)", "0 der 4 Läufe sind nach 4000 Schritten nicht gelöst",
                 "billiger Schritt, viele Schritte"):
        assert part in text, part
    text = ev.verdict_pocs(_pocs_rows(failed=True), 0.01)[1]
    assert "Lorentz-Kegel: 216 Schritte (nicht gelöst)" in text and "1 der 4 Läufe" in text


def test_expcheck_verdict_counts_failures_of_both_versions():
    rows = [{"label": "gewöhnlich", "count": 400, "grid_failures": 1, "root_failures": 0, "worst_gap": 2.8e-13}, {"label": "x/y bei +5 bis +20, z bis 10⁶", "count": 400, "grid_failures": 202, "root_failures": 0, "worst_gap": 0.0996}]
    kind, text = ev.verdict_expcheck(rows)
    assert kind == "warning"
    for part in ("fällt bei 203 von 800 Punkten durch", "am stärksten bei der Klasse „x/y bei +5 bis +20, z bis 10⁶“ (202 von 400)", "die Wurzel-Fassung bei 0", "Der größte Abstandsunterschied beider Fassungen ist 1,0·10⁻¹"):
        assert part in text, part
    assert ev.verdict_expcheck([{**rows[0], "grid_failures": 0}])[0] == "info"
