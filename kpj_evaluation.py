"""Kennzahlen, Experimente und Verdikte. Jedes Verdikt ist eine reine Funktion (Art, Text); die App ruft st.<Art>(Text)."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass

import numpy as np
from threadpoolctl import threadpool_limits

import kegel_kern as K
import kpj_barrier as B
import kpj_constants as C
import kpj_forms as F
import kpj_pocs as PO
import kpj_projection as P
from kpj_scenario import make_cloud


@dataclass(frozen=True)
class Settings:
    kind: str = C.DEFAULT_KIND
    alpha: float = C.DEFAULT_ALPHA
    vx: float = C.DEFAULT_V[0]
    vy: float = C.DEFAULT_V[1]
    vz: float = C.DEFAULT_V[2]
    n: int = C.DEFAULT_N
    stretch: float = C.DEFAULT_STRETCH
    angle: int = C.DEFAULT_ANGLE
    seed: int = C.DEFAULT_SEED
    eps: float = C.DEFAULT_EPS


# ----------------------------------------------------------------------------- Formatierung

def de(x: float, digits: int = 2) -> str:
    """Zahl mit deutschem Dezimalkomma und typografischem Minuszeichen."""
    return f"{x:.{digits}f}".replace(".", ",").replace("-", "−")


_SUPER = str.maketrans("-0123456789", "⁻⁰¹²³⁴⁵⁶⁷⁸⁹")


def sci(x: float, digits: int = 1) -> str:
    """Zahl in Zehnerpotenz-Schreibweise, z. B. 2,0·10⁻¹⁰."""
    if x == 0.0 or not math.isfinite(x):
        return de(x, digits) if math.isfinite(x) else "unendlich"
    e = int(math.floor(math.log10(abs(x))))
    m = x / 10.0 ** e
    if round(abs(m), digits) >= 10.0:
        m, e = m / 10.0, e + 1
    return f"{de(m, digits)}·10{str(e).translate(_SUPER)}"


def vec_text(v, digits: int = 3) -> str:
    return "(" + "; ".join(de(float(x), digits) for x in v) + ")"


def eps_text(eps: float) -> str:
    return de(100.0 * eps, 1).rstrip("0").rstrip(",") + " %"


def alpha_text(a: float) -> str:
    return de(a, 2).rstrip("0").rstrip(",")


def par_of(kind: str, alpha: float):
    return {"s": 2, "p": alpha}.get(kind)


def cone_label(kind: str, alpha: float) -> str:
    return f"{P.KIND_NAMES[kind]}, a = {alpha_text(alpha)}" if kind == "p" else P.KIND_NAMES[kind]


# ----------------------------------------------------------------------------- Experiment 1: eine Projektion

def projection_view(s: Settings) -> dict:
    """Die Projektion eines Punktes v auf den gewählten Kegel, ihr Rest, die Moreau-Probe und die Probe der Formel P(v) − P(−v) = v."""
    par = par_of(s.kind, s.alpha)
    V = np.array([[s.vx, s.vy, s.vz]])
    p = P.project(s.kind, par, V)[0]
    r = V[0] - p
    cert = P.certificate(s.kind, par, V, p[None, :])
    inside = bool(P.in_cone(s.kind, par, V, 1e-9)[0])
    polar = bool(P.in_polar(s.kind, par, V, 1e-9)[0]) and not inside
    case = "im Kegel" if inside else "im Polarkegel" if polar else "auf dem Rand"
    mirror = P.project(s.kind, par, -V)[0]
    extra = ""
    if s.kind == "s":
        M = P.smat_stack(V, 2)[0]
        extra = f"Eigenwerte der Matrix {np.array2string(M, precision=3)}: {vec_text(np.linalg.eigvalsh(M))}"
    elif s.kind == "q":
        extra = f"t = {de(V[0, 0], 3)}, ‖v̄‖ = {de(float(np.linalg.norm(V[0, 1:])), 3)}"
    elif s.kind in ("e", "p") and p[1] > 1e-12:
        extra = f"Randpunkt mit Verhältnis {'x/y' if s.kind == 'e' else 'u/v'} = {de(p[0] / p[1], 4)}"
    return {"v": V[0], "p": p, "r": r, "distance": float(np.linalg.norm(r)), "inner": float(p @ r), "case": case, "certificate_ok": bool(cert["ok"][0]), "in_cone": bool(cert["in_cone"][0]),
            "in_polar": bool(cert["in_polar"][0]), "orthogonal": bool(cert["orthogonal"][0]), "mirror_deviation": float(np.abs(V[0] - (p - mirror)).max()), "extra": extra,
            "self_dual": P.SELF_DUAL[s.kind]}


FORMULAS = {"l": "max(v, 0) je Koordinate", "q": "((‖v̄‖ + t)/2) (1, v̄/‖v̄‖)", "s": "Eigenwerte abschneiden", "e": "Wurzel von ⟨v, u × n⟩ = 0 (eine Variable ρ)", "p": "dasselbe mit u = (ρ, 1, ρ^a)"}


def projection_table(s: Settings) -> list:
    """Derselbe Punkt v in allen fünf Kegeln (für den Potenzkegel mit dem gewählten a)."""
    rows = []
    for kind in P.KINDS:
        v = projection_view(Settings(kind, s.alpha, s.vx, s.vy, s.vz))
        rows.append({**v, "kind": kind, "name": cone_label(kind, s.alpha), "formula": FORMULAS[kind]})
    return rows


def verdict_table(s: Settings, rows: list) -> tuple:
    where = {c: [r["name"] for r in rows if r["case"] == c] for c in ("im Kegel", "im Polarkegel", "auf dem Rand")}
    parts = [f"{c}: {', '.join(names)}" for c, names in where.items() if names]
    ok = sum(r["certificate_ok"] for r in rows)
    sd = [r for r in rows if r["self_dual"]]
    nsd = [r for r in rows if not r["self_dual"]]
    return "info", (f"Der Punkt v = {vec_text([s.vx, s.vy, s.vz])} liegt " + "; ".join(parts) + f". Die Moreau-Probe besteht bei {ok} von {len(rows)} Projektionen. Die Spiegelformel v = P(v) − P(−v) gilt für die selbstdualen Kegel "
                    f"auf {sci(max(r['mirror_deviation'] for r in sd))} und verfehlt v für " + ", ".join(f"{r['name']} um {de(r['mirror_deviation'], 3)}" for r in nsd) + ".")


# ----------------------------------------------------------------------------- Experiment 2: Moreau und Selbstdualität

def _random_points(kind: str, count: int, rng) -> np.ndarray:
    dim = {"l": 3, "q": 3, "s": 3, "e": 3, "p": 3}[kind]
    return 2.0 * rng.normal(size=(count, dim))


def moreau_rows(count: int = C.MOREAU_POINTS, alphas=(0.3, 0.5), seed: int = 0) -> list:
    rng = np.random.default_rng(seed)
    rows = []
    for kind, alpha in [(k, None) for k in ("l", "q", "s", "e")] + [("p", a) for a in alphas]:
        par = par_of(kind, alpha)
        V = _random_points(kind, count, rng)
        Pv = P.project(kind, par, V)
        Pm = P.project(kind, par, -V)
        cert = P.certificate(kind, par, V, Pv)
        PP = P.project(kind, par, Pv)
        half = count // 2
        dist_in = np.linalg.norm(V[:half] - V[half:2 * half], axis=1)
        dist_out = np.linalg.norm(Pv[:half] - Pv[half:2 * half], axis=1)
        rows.append({"cone": cone_label(kind, alpha or 0.0) if kind != "p" else cone_label("p", alpha), "kind": kind, "self_dual": P.SELF_DUAL[kind], "count": count,
                     "mirror": float(np.abs(V - (Pv - Pm)).max()), "certificate": float(cert["ok"].mean()), "idempotent": float(np.abs(PP - Pv).max()),
                     "nonexpansive_violations": int(np.sum(dist_out > dist_in + 1e-9)), "inside": float(P.in_cone(kind, par, V, 1e-9).mean()),
                     "polar": float((P.in_polar(kind, par, V, 1e-9) & ~P.in_cone(kind, par, V, 1e-9)).mean())})
    return rows


# ----------------------------------------------------------------------------- Experiment 3: Kosten

def _best(fun, repeats: int = 3) -> float:
    best = math.inf
    for _ in range(repeats):
        t0 = time.perf_counter()
        fun()
        best = min(best, time.perf_counter() - t0)
    return best


def cost_rows(sizes=C.PSD_SIZES, seed: int = 0) -> dict:
    """Zeit je Projektion (beste von 3 Läufen, ein BLAS-Thread): Lorentz je Koordinate, PSD je Matrix gegen k, Exponential- und Potenzkegel je Kegel. Zeiten hängen vom Rechner ab."""
    rng = np.random.default_rng(seed)
    out = {}
    with threadpool_limits(1):
        V = rng.normal(size=(100000, 3))
        out["lorentz_ns"] = _best(lambda: P.proj_lorentz(V)) / V.size * 1e9
        psd = []
        for k in sizes:
            d = k * (k + 1) // 2
            N = max(1, int(1.5e6 // k ** 3))
            S = rng.normal(size=(N, d))
            t = _best(lambda: P.proj_psd(S, k)) / N
            psd.append({"k": k, "us_per_matrix": t * 1e6, "ns_per_coordinate": t / d * 1e9, "count": N})
        out["psd"] = psd
        ks = np.array([r["k"] for r in psd[1:]], float)
        ts = np.array([r["us_per_matrix"] for r in psd[1:]])
        out["slope"] = float(np.polyfit(np.log(ks), np.log(ts), 1)[0])
        W = 2.0 * rng.normal(size=(5000, 3))
        out["exp_us"] = _best(lambda: P.proj_exp(W)) / len(W) * 1e6
        out["pow_us"] = _best(lambda: P.proj_pow(W, 0.3)) / len(W) * 1e6
    return out


# ----------------------------------------------------------------------------- Experiment 4: abwechselnde Projektionen

def pocs_cases(s: Settings) -> list:
    """Die Kugel der ersten Punkte der Wolke in vier Kegeln: (Name, Art, Programm, R*, Newton-Schritte der Barriere, ν)."""
    pts = make_cloud(s.n, s.stretch, s.angle, s.seed).points[:min(s.n, C.N_POCS)]
    cases = []
    for name, kind, prob in (("Lorentz-Kegel", "q", F.build_soc_ball(pts)), ("PSD-Kegel 3×3", "s", F.build_psd_ball(pts)), ("Potenzkegel (ℓ₃-Kugel)", "p", F.build_lp_ball(pts, 3.0)),
                             ("Exponentialkegel (log-sum-exp, β = 10)", "e", F.build_lse_ball(pts, 10.0))):
        if hasattr(prob, "groups"):
            rstar = B.solve(prob, tol=1e-12).value
            newton = B.solve(prob, tol=1e-9).newton_steps
            nu = prob.nu
        else:
            rstar = K.solve_barrier(prob, tol=1e-12).value
            newton = K.solve_barrier(prob, tol=1e-9).newton_steps
            nu = K.barrier_parameter(prob.cones)
        cases.append((name, kind, prob, rstar, newton, nu))
    return cases, pts


def start_point(prob, points: np.ndarray, rstar: float) -> np.ndarray:
    """Der Start der abwechselnden Projektionen: der Mittelpunkt um R* in jeder Koordinate aus dem Schwerpunkt verschoben, der Radius das Doppelte des größten Abstands zum verschobenen Mittelpunkt.
    Das ist im Kegel (zulässig), aber über jeder Schranke γ ≤ 1,1 R*; die erste Projektion auf C zieht den Radius auf γ herunter, dort ist der Mittelpunkt falsch, und von da an wechseln die Projektionen.
    (Ein Start im Schwerpunkt wäre für die ersten 8 Punkte schon für Schlupf ab etwa 1,5 % zulässig: der Schwerpunkt-Ball ist dort nur 1,5 % größer als R*; für alle 20 Punkte der Standardwolke sind es 16 %.)"""
    d = points.shape[1]
    centre = points.mean(axis=0) + rstar
    r0 = 2.0 * float(np.linalg.norm(points - centre, axis=1).max())
    x = np.array(prob.x0, dtype=float)
    x[:d] = centre
    x[d] = r0
    if hasattr(prob, "groups") and prob.groups[0][0] == "p":
        x[d + 1:] = r0 / (d + 1.0)                    # die Hilfsvariablen s_ij des Potenzkegel-Programms
    return x


def steps_text(n: int) -> str:
    return "1 Schritt" if n == 1 else f"{n} Schritte"


def pocs_rows(s: Settings, eps_list=C.EPS_CHOICES) -> list:
    rows = []
    cases, pts = pocs_cases(s)
    for name, kind, prob, rstar, newton, nu in cases:
        results = []
        for eps in eps_list:
            r = PO.alternate(prob, rstar * (1.0 + eps), max_iter=C.MAX_ITER, tol=C.POCS_TOL, x_start=start_point(prob, pts, rstar))
            hist = r.history
            step = max(1, len(hist) // 300)
            results.append({"eps": eps, "history_step": step, "iterations": r.iterations, "converged": r.converged, "seconds": r.seconds, "distance": r.distance, "value": r.value, "history": hist[::step],
                            "per_step_us": r.seconds / r.iterations * 1e6})
        rows.append({"name": name, "kind": kind, "rstar": rstar, "newton": newton, "nu": nu, "rows": prob.A.shape[0], "variables": prob.A.shape[1], "results": results})
    return rows


# ----------------------------------------------------------------------------- Experiment 5: wann ist eine Projektion richtig?

def exp_check_rows(count: int = C.EXP_CHECK_POINTS, seed: int = 0) -> list:
    """Die Gitter-Fassung und die Wurzel-Fassung der Exponentialkegel-Projektion auf Punkten mit wachsendem x/y, geprüft an der Moreau-Probe (ohne Referenzlöser)."""
    rng = np.random.default_rng(seed)

    def pts(xs, ys, zs):
        return np.stack([xs * np.exp(0.5 * rng.normal(size=count)), ys * np.exp(0.2 * rng.normal(size=count)), zs * np.exp(rng.normal(size=count))], axis=1)

    classes = [("x/y bei −60: weit unter dem Kegel", pts(-60.0, 1.0, 0.1)), ("gewöhnlich (Normalverteilung)", 2.0 * rng.normal(size=(count, 3))), ("x/y bei +5 bis +20, z bis 10⁶", pts(20.0, 1.0, 1e6)),
               ("x/y bei +5, z mäßig", pts(5.0, 1.0, 50.0))]
    rows = []
    for label, V in classes:
        grid, root = P.proj_exp_grid(V), P.proj_exp(V)
        cg, cr = P.certificate("e", None, V, grid), P.certificate("e", None, V, root)
        rows.append({"label": label, "count": len(V), "grid_failures": int((~cg["ok"]).sum()), "root_failures": int((~cr["ok"]).sum()),
                     "worst_gap": float(np.max(np.linalg.norm(V - grid, axis=1) - np.linalg.norm(V - root, axis=1)))})
    return rows


# ----------------------------------------------------------------------------- Verdikte (reine Funktionen: (Art, Text))

def verdict_projection(s: Settings, view: dict) -> tuple:
    name = cone_label(s.kind, s.alpha)
    text = (f"Der Punkt v = {vec_text(view['v'])} liegt {view['case']} des **{name}**; seine Projektion ist **{vec_text(view['p'])}** (Abstand {de(view['distance'], 4)}, Rest {vec_text(view['r'])}). "
            f"Moreau: ⟨p, v − p⟩ = {sci(view['inner'])}, p im Kegel: {'ja' if view['in_cone'] else 'nein'}, v − p im Polarkegel: {'ja' if view['in_polar'] else 'nein'}.")
    if view["extra"]:
        text += " " + view["extra"] + "."
    if view["self_dual"]:
        text += f" Der Kegel ist selbstdual: v = P(v) − P(−v) stimmt auf {sci(view['mirror_deviation'])}."
    else:
        text += f" Der Kegel ist nicht selbstdual: P(v) − P(−v) verfehlt v um {de(view['mirror_deviation'], 3)}."
    if not view["certificate_ok"]:
        return "warning", text + " **Die Moreau-Probe besteht nicht** (am Rand mit Spitze kann der Rest die Dualkegel-Bedingung knapp verfehlen, obwohl der Abstand stimmt)."
    return "success", text + " Die Moreau-Probe besteht."


def verdict_moreau(rows: list) -> tuple:
    sd = [r for r in rows if r["self_dual"]]
    nsd = [r for r in rows if not r["self_dual"]]
    worst_sd = max(r["mirror"] for r in sd)
    return "info", (f"An je {rows[0]['count']} Zufallspunkten gilt v = P(v) − P(−v) für die selbstdualen Kegel (Orthant, Lorentz, PSD) auf {sci(worst_sd)}; für "
                    + ", ".join(f"{r['cone']} verfehlt es v um bis zu {de(r['mirror'], 2)}" for r in nsd)
                    + f". Die Moreau-Probe (p im Kegel, v − p im Polarkegel, ⟨p, v − p⟩ = 0) besteht bei {de(100 * min(r['certificate'] for r in rows), 1)} % bis {de(100 * max(r['certificate'] for r in rows), 1)} % der Punkte; "
                    f"die Projektion ist idempotent (größte Abweichung {sci(max(r['idempotent'] for r in rows))}) und nichtexpansiv ({sum(r['nonexpansive_violations'] for r in rows)} Verletzungen).")


def verdict_cost(res: dict) -> tuple:
    p4, p32 = next(r for r in res["psd"] if r["k"] == 4), next(r for r in res["psd"] if r["k"] == 32)
    return "info", (f"Lorentz-Kegel: {de(res['lorentz_ns'], 0)} ns je Koordinate (linear in der Größe). PSD-Kegel: {de(p4['us_per_matrix'], 1)} µs je 4×4-Matrix und {de(p32['us_per_matrix'], 0)} µs je 32×32-Matrix, die Zeit wächst "
                    f"mit Steigung {de(res['slope'], 2)} gegen k (Theorie der Eigenzerlegung: 3); je Koordinate ist der PSD-Kegel bei k = 32 {de(p32['ns_per_coordinate'] / res['lorentz_ns'], 0)}-mal teurer als der Lorentz-Kegel. "
                    f"Exponentialkegel {de(res['exp_us'], 1)} µs und Potenzkegel {de(res['pow_us'], 1)} µs je Kegel (gestapelt, Gitter und Wurzelsuche). Zeiten hängen vom Rechner ab.")


def verdict_pocs(rows: list, eps: float) -> tuple:
    parts = []
    for r in rows:
        res = next(x for x in r["results"] if x["eps"] == eps)
        parts.append(f"{r['name']}: {steps_text(res['iterations'])}" + ("" if res["converged"] else " (nicht gelöst)"))
    newton = ", ".join(f"{r['name'].split(' (')[0]} {r['newton']}" for r in rows)
    failed = sum(1 for r in rows for x in r["results"] if not x["converged"])
    return "info", (f"Bei Schlupf {eps_text(eps)} brauchen abwechselnde Projektionen für die Kugel der ersten {min(C.N_POCS, 40)} Punkte: " + "; ".join(parts) + f". Die Barriere aus Stück 5 bis 7 braucht für dieselben Programme "
                    f"{min(r['newton'] for r in rows)} bis {max(r['newton'] for r in rows)} Newton-Schritte bis 10⁻⁹ ({newton}). {failed} der {sum(len(r['results']) for r in rows)} Läufe sind nach {C.MAX_ITER} Schritten nicht gelöst: "
                    f"billiger Schritt, viele Schritte (für den gewählten Start; mit dem Schwerpunkt als Start sind es deutlich weniger).")


def verdict_expcheck(rows: list) -> tuple:
    g, r = sum(x["grid_failures"] for x in rows), sum(x["root_failures"] for x in rows)
    n = sum(x["count"] for x in rows)
    worst = max(rows, key=lambda x: x["grid_failures"])
    return ("warning" if g else "info"), (f"Die Moreau-Probe prüft eine Projektion ohne Referenzlöser. Die Gitter-Fassung (Maximum auf einem festen Gitter von −40 bis 40) fällt bei {g} von {n} Punkten durch, "
                                           f"am stärksten bei der Klasse „{worst['label']}“ ({worst['grid_failures']} von {worst['count']}); die Wurzel-Fassung bei {r}. Der größte Abstandsunterschied beider Fassungen ist "
                                           f"{sci(max(x['worst_gap'] for x in rows))}.")
