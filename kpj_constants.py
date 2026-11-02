"""Konstanten: Regler-Grenzen, feste Stichproben, Farben und Presets."""

import kpj_projection as P

DEFAULT_KIND = "e"
ALPHA_CHOICES = (0.25, 0.3, 0.5, 0.75)                 # Exponent des Potenzkegels in Experiment 1 und 2
DEFAULT_ALPHA = 0.3
V_MIN, V_MAX, V_STEP = -5.0, 5.0, 0.1                  # Koordinaten des Punktes v
DEFAULT_V = (1.0, 0.5, -0.5)
N_MIN, N_MAX, DEFAULT_N = 8, 40, 20
STRETCH_MIN, STRETCH_MAX, DEFAULT_STRETCH = 1.2, 6.0, 4.0
ANGLE_MIN, ANGLE_MAX, DEFAULT_ANGLE = 0, 90, 45
SEED_MAX, DEFAULT_SEED = 999999, 35
EPS_CHOICES = (0.1, 0.03, 0.01, 0.003)                 # Schlupf γ/R* − 1 der abwechselnden Projektionen
DEFAULT_EPS = 0.03
N_POCS = 8                                             # Punkte der Wolke, mit denen die vier Kugeln in Experiment 4 gerechnet werden
MAX_ITER = 4000
POCS_TOL = 1e-7
MOREAU_POINTS = 2000
PSD_SIZES = (4, 8, 16, 32, 64, 128)
EXP_CHECK_POINTS = 400

COLORS = {"l": "#888888", "q": "#4c78a8", "s": "#8c6bb1", "e": "#2F6B65", "p": "#b08a2e", "bound": "#c0504d"}
COLOR_OK = "#2F6B65"
COLOR_WARN = "#b08a2e"
COLOR_BAD = "#b34a4a"

PRESETS = {
    "Standardfall: Exponentialkegel, Punkt außen": {"kind": "e", "alpha": DEFAULT_ALPHA, "vx": 1.0, "vy": 0.5, "vz": -0.5, "n": DEFAULT_N, "stretch": DEFAULT_STRETCH, "angle": DEFAULT_ANGLE, "seed": DEFAULT_SEED, "eps": DEFAULT_EPS},
    "Punkt im Kegel: nichts zu tun": {"kind": "e", "alpha": DEFAULT_ALPHA, "vx": 0.0, "vy": 1.0, "vz": 2.0, "n": DEFAULT_N, "stretch": DEFAULT_STRETCH, "angle": DEFAULT_ANGLE, "seed": DEFAULT_SEED, "eps": DEFAULT_EPS},
    "Punkt im Polarkegel: die Projektion ist 0": {"kind": "e", "alpha": DEFAULT_ALPHA, "vx": 1.0, "vy": -1.0, "vz": -1.0, "n": DEFAULT_N, "stretch": DEFAULT_STRETCH, "angle": DEFAULT_ANGLE, "seed": DEFAULT_SEED, "eps": DEFAULT_EPS},
    "Lorentz-Kegel: die geschlossene Formel": {"kind": "q", "alpha": DEFAULT_ALPHA, "vx": 1.0, "vy": 3.0, "vz": 4.0, "n": DEFAULT_N, "stretch": DEFAULT_STRETCH, "angle": DEFAULT_ANGLE, "seed": DEFAULT_SEED, "eps": DEFAULT_EPS},
    "PSD-Kegel: Eigenwerte abschneiden": {"kind": "s", "alpha": DEFAULT_ALPHA, "vx": 2.0, "vy": 0.0, "vz": -1.0, "n": DEFAULT_N, "stretch": DEFAULT_STRETCH, "angle": DEFAULT_ANGLE, "seed": DEFAULT_SEED, "eps": DEFAULT_EPS},
    "Potenzkegel a = 0,3: nicht selbstdual": {"kind": "p", "alpha": 0.3, "vx": 2.0, "vy": -1.0, "vz": 1.5, "n": DEFAULT_N, "stretch": DEFAULT_STRETCH, "angle": DEFAULT_ANGLE, "seed": DEFAULT_SEED, "eps": DEFAULT_EPS},
}
PRESET_HELP = {
    "Standardfall: Exponentialkegel, Punkt außen": "v = (1; 0,5; −0,5) liegt weder im Exponentialkegel noch in seinem Polarkegel: die Projektion liegt auf dem Rand und kommt aus einer Gleichung in einer Variablen (keine geschlossene Form).",
    "Punkt im Kegel: nichts zu tun": "v = (0; 1; 2) liegt im Exponentialkegel (1 · e⁰ = 1 ≤ 2): die Projektion ist v selbst, der Rest ist null.",
    "Punkt im Polarkegel: die Projektion ist 0": "v = (1; −1; −1): −v = (−1; 1; 1) liegt im Dualkegel (e⁻¹ ≤ e · 1), also im Polarkegel von v: die Projektion ist 0 und der Rest ist v.",
    "Lorentz-Kegel: die geschlossene Formel": "v = (1; 3, 4): ‖v̄‖ = 5 > t = 1, also ((5 + 1)/2) (1; 3/5, 4/5) = (3; 1,8; 2,4): die Spektralformel des Lorentz-Kegels.",
    "PSD-Kegel: Eigenwerte abschneiden": "Die 2×2-Matrix diag(2, −1) (svec: 2, 0, −1): der negative Eigenwert wird abgeschnitten, die Projektion ist diag(2, 0).",
    "Potenzkegel a = 0,3: nicht selbstdual": "Für den Potenzkegel mit a = 0,3 gilt P(v) − P(−v) = v nicht: sein Dualkegel ist ein anderer Kegel als er selbst.",
}
assert set(PRESETS) == set(PRESET_HELP) and all(p["kind"] in P.KINDS and p["alpha"] in ALPHA_CHOICES and p["eps"] in EPS_CHOICES for p in PRESETS.values())
