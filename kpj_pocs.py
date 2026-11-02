"""Abwechselnde Projektionen (POCS) für Kegelprogramme: zwischen dem Kegel K und dem Bild C_γ = {b − A x: cᵀx ≤ γ} wechseln.

    y = P_C(s),  s = P_K(y)           Abbruch, wenn ‖y − s‖ ≤ tol (1 + ‖s‖)
P_C ist geschlossen: x₀ = (AᵀA)⁻¹Aᵀ(b − s) ist die Kleinste-Quadrate-Lösung; liegt cᵀx₀ über γ, wird x₀ um λ (AᵀA)⁻¹c zurückgeschoben, bis cᵀx = γ.
Ein Kegelprogramm mit Optimalwert p* hat für γ > p* einen Durchschnitt mit Innerem, und die Folge nähert ihn (Bauschke und Borwein 1996); für γ → p* schrumpft er auf einen Punkt.
Der Kegel kommt als `spec` = [(Art, Parameter, Anzahl)] mit Art 'l' (Orthant, Anzahl Zeilen), 'q' (Lorentz, Parameter Größe), 's' (PSD, Parameter k), 'e' (Exponential), 'p' (Potenz, Parameter a)."""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

import kpj_projection as P


def spec_of(prob) -> list:
    """Die Kegelzerlegung eines Programms des Kerns (ConeProblem, `cones`) oder der Stück-7-Barriere (Program, `groups`)."""
    if hasattr(prob, "groups"):
        return list(prob.groups)
    spec = []
    for kind, m in prob.cones:
        if spec and spec[-1][0] == kind and spec[-1][1] == m:
            spec[-1] = (kind, m, spec[-1][2] + 1)
        else:
            spec.append((kind, m, 1))
    return spec


def project_spec(spec: list, s: np.ndarray) -> np.ndarray:
    out = np.empty_like(s)
    off = 0
    for kind, par, n in spec:
        if kind == "l":
            d = n
            out[off:off + d] = np.maximum(s[off:off + d], 0.0)
        elif kind == "q":
            d = par * n
            out[off:off + d] = P.proj_lorentz(s[off:off + d].reshape(n, par)).reshape(-1)
        elif kind == "s":
            d = par * (par + 1) // 2 * n
            out[off:off + d] = P.proj_psd(s[off:off + d].reshape(n, par * (par + 1) // 2), par).reshape(-1)
        elif kind == "e":
            d = 3 * n
            out[off:off + d] = P.proj_exp(s[off:off + d].reshape(n, 3)).reshape(-1)
        else:
            d = 3 * n
            out[off:off + d] = P.proj_pow(s[off:off + d].reshape(n, 3), par).reshape(-1)
        off += d
    return out


@dataclass(frozen=True, eq=False)
class PocsResult:
    iterations: int
    converged: bool
    value: float                 # cᵀx am Ende (≤ γ)
    distance: float              # ‖y − s‖ am Ende
    seconds: float
    history: list = field(default_factory=list)   # ‖y − s‖ je Schritt


def alternate(prob, gamma: float, max_iter: int = 5000, tol: float = 1e-7, x_start=None) -> PocsResult:
    """Abwechselnde Projektionen für `min cᵀx, b − A x ∈ K` mit der Schranke cᵀx ≤ γ; Start s = b − A x_start (Standard: x₀ des Programms)."""
    A, b, c, spec = prob.A, prob.b, prob.c, spec_of(prob)
    M = A.T @ A
    Minv = np.linalg.inv(M)
    Minv_c = Minv @ c
    cMc = float(c @ Minv_c)
    x = prob.x0 if x_start is None else x_start
    s = b - A @ x
    hist = []
    t0 = time.perf_counter()
    for it in range(1, max_iter + 1):
        x = Minv @ (A.T @ (b - s))
        cx = float(c @ x)
        if cx > gamma:
            x = x - (cx - gamma) / cMc * Minv_c
        y = b - A @ x
        s = project_spec(spec, y)
        dist = float(np.linalg.norm(y - s))
        hist.append(dist)
        if dist <= tol * (1.0 + np.linalg.norm(s)):
            return PocsResult(it, True, float(c @ x), dist, time.perf_counter() - t0, hist)
    return PocsResult(max_iter, False, float(c @ x), hist[-1], time.perf_counter() - t0, hist)
