"""Jede Zahl aus README.md und den App-Texten wird hier über dieselben Auswertungsfunktionen nachgerechnet (DEMO-PLAYBOOK §4).

Bänder statt Punktwerte, wo die Rundung oder der Rechner die Zahl bewegt (die Auswahl der BLAS-Routinen auf dem Runner verschiebt Läufe im Rundungsbereich); Zeiten werden nicht getestet,
nur ihre Struktur (Reihenfolge der Größenordnungen)."""

import numpy as np
import pytest

import kegel_kern as K
import kpj_constants as C
import kpj_evaluation as ev
import kpj_projection as P


@pytest.fixture(scope="module")
def pocs():
    return {r["kind"]: r for r in ev.pocs_rows(ev.Settings())}


def test_the_default_view_and_the_five_cones_on_the_same_point():
    """README: v = (1; 0,5; −0,5): im Exponentialkegel Projektion (0,023; 0,043; 0,074), Abstand 1,2215, Randpunkt x/y = 0,5325; die Moreau-Probe besteht in allen fünf Kegeln."""
    s = ev.Settings()
    view = ev.projection_view(s)
    assert view["p"] == pytest.approx([0.023, 0.043, 0.074], abs=5e-4) and view["distance"] == pytest.approx(1.2215, abs=5e-4) and view["case"] == "auf dem Rand" and "0,5325" in view["extra"]
    assert abs(view["inner"]) < 1e-12 and view["certificate_ok"]
    rows = ev.projection_table(s)
    assert [r["certificate_ok"] for r in rows] == [True] * 5 and [r["case"] for r in rows].count("auf dem Rand") >= 3
    assert abs(view["mirror_deviation"] - 0.457) < 5e-3


def test_moreau_rows():
    """README: Selbstduale Kegel v = P(v) − P(−v) auf 3·10⁻¹⁵ (Band 10⁻¹²); Exponentialkegel verfehlt um bis zu 4,2, Potenzkegel a = 0,3 um 2,1 und a = 0,5 um 2,2; Moreau-Probe besteht bei 99,85 % (Exponential) bis 100 %."""
    rows = {r["cone"]: r for r in ev.moreau_rows()}
    for name in ("Orthant", "Lorentz-Kegel", "PSD-Kegel (2×2)"):
        assert rows[name]["mirror"] < 1e-12 and rows[name]["self_dual"] and rows[name]["certificate"] == 1.0
    assert 3.5 <= rows["Exponentialkegel"]["mirror"] <= 5.0 and 1.8 <= rows["Potenzkegel, a = 0,3"]["mirror"] <= 2.4 and 1.9 <= rows["Potenzkegel, a = 0,5"]["mirror"] <= 2.5
    assert rows["Exponentialkegel"]["certificate"] >= 0.995 and rows["Potenzkegel, a = 0,3"]["certificate"] == 1.0 and rows["Potenzkegel, a = 0,5"]["certificate"] == 1.0
    assert all(r["idempotent"] < 1e-9 and r["nonexpansive_violations"] == 0 and r["count"] == 2000 for r in rows.values())
    assert not any(r["self_dual"] for k, r in rows.items() if k.startswith(("Exponential", "Potenz")))


def test_cost_structure():
    """README: Lorentz etwa 40 bis 50 ns je Koordinate (Band 10 bis 200), PSD je Matrix wachsend mit Steigung 2,1 (Band 1,5 bis 3,2), je Koordinate bei k = 32 etwa 6- bis 8-mal Lorentz (Band 2 bis 30);
    Exponentialkegel etwa 7 µs, Potenzkegel etwa 14 µs je Kegel (Band 1 bis 100). Zeiten hängen vom Rechner ab, nur die Größenordnungen stehen im Test."""
    res = ev.cost_rows()
    assert 10 <= res["lorentz_ns"] <= 200 and 1.5 <= res["slope"] <= 3.2 and 1.0 <= res["exp_us"] <= 100 and 1.0 <= res["pow_us"] <= 100
    by = {r["k"]: r for r in res["psd"]}
    assert [r["k"] for r in res["psd"]] == list(C.PSD_SIZES) and by[128]["us_per_matrix"] > 100 * by[4]["us_per_matrix"]
    assert 2 <= by[32]["ns_per_coordinate"] / res["lorentz_ns"] <= 30


def test_alternating_projections_need_far_more_steps_than_the_barrier(pocs):
    """README (8 Punkte, Start mit verschobenem Mittelpunkt, Abbruch 10⁻⁷): Lorentz 215, PSD 214 (alle ε), Potenz 180, 434, 491, 513, Exponential 1959 bis 2023 Schritte; Barriere 31, 31, 34, 43 Newton-Schritte."""
    its = {k: [x["iterations"] for x in r["results"]] for k, r in pocs.items()}
    assert [r["eps"] for r in pocs["q"]["results"]] == list(C.EPS_CHOICES)
    assert all(r["converged"] for rr in pocs.values() for r in rr["results"])
    assert all(150 <= i <= 300 for i in its["q"]) and all(150 <= i <= 300 for i in its["s"])
    assert 100 <= its["p"][0] <= 300 and all(300 <= i <= 800 for i in its["p"][1:]) and all(1200 <= i <= 3000 for i in its["e"])
    assert abs(its["q"][0] - its["s"][0]) <= 20                                  # Lorentz- und PSD-Kugel sind dieselbe Aufgabe (R ≥ ‖v‖)
    assert [pocs[k]["newton"] for k in ("q", "s", "p", "e")] == pytest.approx([31, 31, 34, 43], abs=6)
    assert min(its["q"]) > 3 * pocs["q"]["newton"] and min(its["e"]) > 30 * pocs["e"]["newton"]
    assert [pocs[k]["nu"] for k in ("q", "s", "p", "e")] == [16, 24, 56, 104]


def test_the_optimum_values_of_the_four_balls(pocs):
    """README: R* der ersten 8 Punkte: Lorentz und PSD 7,0359, ℓ₃-Kugel 6,2683, log-sum-exp β = 10 5,0463."""
    assert pocs["q"]["rstar"] == pytest.approx(7.0359, abs=5e-4) and pocs["s"]["rstar"] == pytest.approx(pocs["q"]["rstar"], abs=1e-7)
    assert pocs["p"]["rstar"] == pytest.approx(6.2683, abs=5e-4) and pocs["e"]["rstar"] == pytest.approx(5.0463, abs=5e-4)
    assert pocs["q"]["rstar"] > pocs["p"]["rstar"] > pocs["e"]["rstar"]                  # ℓ2 > ℓ3 > fast Würfel


def test_the_final_values_respect_the_slack(pocs):
    for r in pocs.values():
        for x in r["results"]:
            assert x["value"] <= r["rstar"] * (1 + x["eps"]) + 1e-9 and x["value"] >= r["rstar"] - 1e-6


def test_the_two_versions_of_the_exponential_projection():
    """README: die Gitter-Fassung fällt an 203 von 1600 Punkten durch (202 von 400 bei x/y von +5 bis +20 und z bis 10⁶), die Wurzel-Fassung an keinem; Abstandsunterschied bis 0,1."""
    rows = ev.exp_check_rows()
    by = {r["label"]: r for r in rows}
    hard = by["x/y bei +5 bis +20, z bis 10⁶"]
    assert 150 <= hard["grid_failures"] <= 260 and hard["root_failures"] == 0 and hard["worst_gap"] > 1e-2
    assert sum(r["root_failures"] for r in rows) == 0 and sum(r["count"] for r in rows) == 1600 and 150 <= sum(r["grid_failures"] for r in rows) <= 280
    assert by["x/y bei −60: weit unter dem Kegel"]["grid_failures"] == 0 and by["x/y bei +5, z mäßig"]["grid_failures"] == 0


def test_the_moreau_probe_at_the_cusp_of_the_power_cone():
    """README: für a = 0,1 und 0,9 (Spitze) fällt die Probe an 1 bis 3 % der Punkte knapp durch, für a = 0,3 und 0,5 an keinem (gemischte Größenordnungen)."""
    rng = np.random.default_rng(1)
    mix = np.concatenate([rng.normal(size=(150, 3)), 3 * rng.normal(size=(100, 3)), np.exp(3 * rng.normal(size=(100, 3))) * rng.choice([-1, 1], size=(100, 3))])
    ok = {a: P.certificate("p", a, mix, P.proj_pow(mix, a))["ok"].mean() for a in (0.1, 0.3, 0.5, 0.75, 0.9)}
    assert ok[0.3] == 1.0 and ok[0.5] == 1.0 and ok[0.75] >= 0.99 and 0.95 <= ok[0.1] < 1.0 and 0.95 <= ok[0.9] < 1.0


def test_the_kernel_is_byte_identical_in_the_series():
    """README: kegel_kern.py ist wortgleich mit Stück 2 bis 7: Prüfsumme wie im Schwester-Repo, sofern es nebenan liegt."""
    import hashlib
    from pathlib import Path

    here = Path(__file__).resolve().parent.parent / "kegel_kern.py"
    sibling = here.parent.parent / "kegel-exponential-demo" / "kegel_kern.py"
    if not sibling.exists():
        pytest.skip("Schwester-Repo liegt nicht nebenan")
    assert hashlib.sha256(here.read_bytes()).hexdigest() == hashlib.sha256(sibling.read_bytes()).hexdigest()
