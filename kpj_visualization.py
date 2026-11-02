"""Plotly-Abbildungen. Alle 2D-Achsen sind gesperrt (fixedrange), damit Touch-Geräte beim Scrollen nicht am Diagramm hängen bleiben; die 3D-Ansicht ist drehbar."""

from __future__ import annotations

import math

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import kpj_constants as C
import kpj_projection as P


def lock_axes(fig):
    fig.update_xaxes(fixedrange=True)
    fig.update_yaxes(fixedrange=True)
    return fig


def _base(fig, height=380, legend_below=False):
    legend = dict(orientation="h", yanchor="top", y=-0.22, x=0) if legend_below else dict(orientation="h", yanchor="bottom", y=1.02, x=0)
    fig.update_layout(height=height + (40 if legend_below else 0), margin=dict(l=10, r=10, t=50, b=10), legend=legend, plot_bgcolor="rgba(0,0,0,0)")
    return lock_axes(fig)


SHORT = {"q": "Lorentz-Kegel", "s": "PSD-Kegel 3×3", "p": "Potenzkegel (ℓ₃)", "e": "Exponentialkegel (lse)"}


def _positive(values, floor=1e-17):
    return [max(float(v), floor) for v in values]


# ----------------------------------------------------------------------------- Experiment 1: der Kegel in 3D

def _surface(kind: str, alpha: float, L: float, n: int = 36):
    """Randflächen des Kegels in den Koordinaten (x, y, z) als Liste von (X, Y, Z)-Gittern, zugeschnitten auf den Würfel [−L, L]³."""
    t = np.linspace(0.0, 1.0, 12)
    surfaces = []
    if kind == "l":
        g = np.linspace(0.0, L, 8)
        A, B = np.meshgrid(g, g)
        Z = np.zeros_like(A)
        return [(Z, A, B), (A, Z, B), (A, B, Z)]
    if kind == "q":
        phi = np.linspace(0.0, 2.0 * math.pi, n)
        T, PH = np.meshgrid(t * L, phi)
        return [(T, T * np.cos(PH), T * np.sin(PH))]
    if kind == "s":
        phi = np.linspace(0.0, math.pi, n)
        S, PH = np.meshgrid(t * L, phi)
        return [(S * np.cos(PH) ** 2, math.sqrt(2.0) * S * np.cos(PH) * np.sin(PH), S * np.sin(PH) ** 2)]
    if kind == "e":
        rho = np.linspace(-5.0, 2.5, 90)
        scale = L / np.maximum(1.0, np.maximum(np.abs(rho), np.exp(rho)))
        R, T = np.meshgrid(rho, t)
        Y = T * scale[None, :]
        return [(Y * R, Y, Y * np.exp(R))]
    rho = np.exp(np.linspace(-6.0, 3.0, 90))
    scale = L / np.maximum(1.0, np.maximum(rho, rho ** alpha))
    R, T = np.meshgrid(rho, t)
    Y = T * scale[None, :]
    return [(Y * R, Y, sg * Y * R ** alpha) for sg in (1.0, -1.0)]


def build_cone3d(kind: str, alpha: float, v: np.ndarray, p: np.ndarray) -> go.Figure:
    L = max(3.0, 1.4 * float(np.abs(np.concatenate([v, p])).max()))
    fig = go.Figure()
    for i, (X, Y, Z) in enumerate(_surface(kind, alpha, L)):
        fig.add_trace(go.Surface(x=X, y=Y, z=Z, opacity=0.4, showscale=False, colorscale=[[0, C.COLORS[kind]], [1, C.COLORS[kind]]], hoverinfo="skip", name="Kegel" if i == 0 else "", showlegend=i == 0))
    fig.add_trace(go.Scatter3d(x=[v[0], p[0]], y=[v[1], p[1]], z=[v[2], p[2]], mode="lines", line=dict(color="#888888", width=4, dash="dash"), name="Rest v − P(v)", hoverinfo="skip"))
    fig.add_trace(go.Scatter3d(x=[v[0]], y=[v[1]], z=[v[2]], mode="markers", marker=dict(size=8, color=C.COLOR_BAD), name="v"))
    fig.add_trace(go.Scatter3d(x=[p[0]], y=[p[1]], z=[p[2]], mode="markers", marker=dict(size=8, color=C.COLOR_OK), name="P(v)"))
    labels = {"s": ("a", "√2 b", "c")}.get(kind, ("x", "y", "z") if kind in ("e", "p") else ("t / x₁", "x₂ / y", "x₃ / z"))
    rng = dict(range=[-L, L], autorange=False)
    fig.update_layout(height=520, margin=dict(l=0, r=0, t=30, b=0), legend=dict(orientation="h", yanchor="bottom", y=1.0, x=0),
                      scene=dict(xaxis=dict(title=labels[0], **rng), yaxis=dict(title=labels[1], **rng), zaxis=dict(title=labels[2], **rng), aspectmode="cube", camera=dict(eye=dict(x=1.25, y=-1.35, z=0.75))))
    return fig


# ----------------------------------------------------------------------------- Experiment 2 bis 5

def build_moreau(rows: list) -> go.Figure:
    fig = go.Figure(go.Bar(x=[r["cone"] for r in rows], y=_positive(r["mirror"] for r in rows), marker_color=[C.COLOR_OK if r["self_dual"] else C.COLOR_BAD for r in rows],
                           text=[f"{r['mirror']:.1e}" for r in rows], textposition="outside", hovertemplate="%{x}: %{y:.2e}<extra></extra>"))
    fig.update_yaxes(title_text="größte Abweichung |v − (P(v) − P(−v))|", type="log", exponentformat="power", dtick=4, range=[-17, 2])
    fig.update_xaxes(tickangle=-20)
    return _base(fig, height=360)


def build_cost(res: dict) -> go.Figure:
    fig = make_subplots(rows=1, cols=2, subplot_titles=("PSD: Zeit je Matrix gegen die Größe k", "Zeit je Koordinate"))
    ks = [r["k"] for r in res["psd"]]
    ts = [r["us_per_matrix"] for r in res["psd"]]
    fig.add_trace(go.Scatter(x=ks, y=ts, mode="lines+markers", name="gemessen", line=dict(color=C.COLORS["s"], width=3)), row=1, col=1)
    k0, t0 = ks[1], ts[1]
    fig.add_trace(go.Scatter(x=ks[1:], y=[t0 * (k / k0) ** 3 for k in ks[1:]], mode="lines", name="k³", line=dict(color="#888888", dash="dot")), row=1, col=1)
    fig.add_trace(go.Scatter(x=ks[1:], y=[t0 * (k / k0) ** 2 for k in ks[1:]], mode="lines", name="k²", line=dict(color="#888888", dash="dash")), row=1, col=1)
    bars = [("Lorentz", res["lorentz_ns"], "q")] + [(f"PSD k = {r['k']}", r["ns_per_coordinate"], "s") for r in res["psd"] if r["k"] in (4, 32, 128)] + \
           [("Exponential", res["exp_us"] * 1000.0 / 3.0, "e"), ("Potenz", res["pow_us"] * 1000.0 / 3.0, "p")]
    fig.add_trace(go.Bar(x=[b[0] for b in bars], y=[b[1] for b in bars], marker_color=[C.COLORS[b[2]] for b in bars], showlegend=False, hovertemplate="%{x}: %{y:.0f} ns<extra></extra>"), row=1, col=2)
    fig.update_xaxes(title_text="k", type="log", tickvals=ks, ticktext=[str(k) for k in ks], row=1, col=1)
    fig.update_yaxes(title_text="µs je Matrix", type="log", exponentformat="power", dtick=1, row=1, col=1)
    fig.update_yaxes(title_text="ns je Koordinate", type="log", exponentformat="power", dtick=1, row=1, col=2)
    fig.update_xaxes(tickangle=-30, row=1, col=2)
    return _base(fig, legend_below=True)


def build_pocs(rows: list) -> go.Figure:
    fig = go.Figure()
    eps = [r["eps"] for r in rows[0]["results"]]
    labels = [f"{100 * e:g} %" for e in eps]
    for r in rows:
        its = [x["iterations"] for x in r["results"]]
        fig.add_trace(go.Scatter(x=labels, y=its, mode="lines+markers", name=SHORT[r["kind"]], line=dict(color=C.COLORS[r["kind"]], width=3),
                                 hovertemplate="Schlupf %{x}: %{y} Schritte<extra></extra>"))
        bad = [i for i, x in enumerate(r["results"]) if not x["converged"]]
        if bad:
            fig.add_trace(go.Scatter(x=[labels[i] for i in bad], y=[its[i] for i in bad], mode="markers", marker=dict(symbol="x", size=13, color=C.COLOR_BAD, line=dict(width=2)), name=f"{SHORT[r['kind']]}: nicht gelöst", showlegend=False))
    fig.update_xaxes(title_text="Schlupf γ/R* − 1", type="category")
    fig.update_yaxes(title_text="Projektionsschritte", type="log", tickvals=[100, 200, 500, 1000, 2000, 4000], ticktext=["100", "200", "500", "1000", "2000", "4000"])
    return _base(fig)


def build_rate(rows: list, eps: float) -> go.Figure:
    fig = go.Figure()
    for r in rows:
        res = next(x for x in r["results"] if x["eps"] == eps)
        fig.add_trace(go.Scatter(x=[(i + 1) * res["history_step"] for i in range(len(res["history"]))], y=_positive(res["history"]), mode="lines", name=SHORT[r["kind"]], line=dict(color=C.COLORS[r["kind"]], width=2.5)))
    fig.update_xaxes(title_text="Projektionsschritt")
    fig.update_yaxes(title_text="Abstand ‖y − s‖ der beiden Mengen", type="log", exponentformat="power", dtick=2)
    return _base(fig)


def build_expcheck(rows: list) -> go.Figure:
    labels = [r["label"] for r in rows]
    fig = go.Figure()
    fig.add_trace(go.Bar(x=labels, y=[r["grid_failures"] for r in rows], name="Gitter + Maximum (erste Fassung)", marker_color=C.COLOR_BAD))
    fig.add_trace(go.Bar(x=labels, y=[r["root_failures"] for r in rows], name="Wurzel von ⟨v, u × n⟩ = 0", marker_color=C.COLOR_OK))
    fig.update_layout(barmode="group")
    fig.update_yaxes(title_text="Punkte, die die Moreau-Probe nicht bestehen")
    fig.update_xaxes(tickangle=-12)
    return _base(fig, height=360)
