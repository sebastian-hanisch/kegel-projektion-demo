"""Projektion auf Kegel – Moreau, Spektralformeln und die Kegel ohne geschlossene Form - interaktive Konzept-Demo
Sebastian Hanisch - Operations Research und Machine Learning

Stück 8 der Reihe „Konvexe und konische Optimierung“: Die Projektion auf einen Kegel ist der Baustein des zweiten Strangs (Erste Ordnung). Die Demo zeigt sie für fünf Kegel (Spektralformel oder Wurzelgleichung),
die Moreau-Zerlegung als Zertifikat der Projektion, was sie kostet und wie viele Schritte abwechselnde Projektionen für dieselben Kugeln brauchen, für die die Barriere aus Stück 5 bis 7 etwa 30 bis 45 Newton-Schritte brauchte.
Aufbau und Regeln: DEMO-PLAYBOOK.md (§2).

Lauffähig mit: streamlit run app.py
"""

import pandas as pd
import streamlit as st

import kpj_constants as C
import kpj_projection as P
from kpj_evaluation import (
    Settings,
    alpha_text,
    cone_label,
    cost_rows,
    de,
    eps_text,
    exp_check_rows,
    moreau_rows,
    pocs_rows,
    projection_table,
    projection_view,
    sci,
    vec_text,
    verdict_cost,
    verdict_expcheck,
    verdict_moreau,
    verdict_pocs,
    verdict_projection,
    verdict_table,
)
from kpj_layout import dunkles_design, kopf_css
from kpj_pdf_export import build_pdf
from kpj_presets import bounds, init_session_state_defaults, load_permalink_settings, randomize_seed, sidebar_presets, sync_query_params
from kpj_visualization import build_cone3d, build_cost, build_expcheck, build_moreau, build_pocs, build_rate

st.set_page_config(page_title="Projektion auf Kegel – Sebastian Hanisch", layout="wide")


@st.cache_data(show_spinner=False)
def _moreau():
    return moreau_rows()


@st.cache_data(show_spinner=False)
def _cost():
    return cost_rows()


@st.cache_data(show_spinner=False)
def _pocs(n, stretch, angle, seed):
    return pocs_rows(Settings(n=n, stretch=stretch, angle=angle, seed=seed))


@st.cache_data(show_spinner=False)
def _expcheck():
    return exp_check_rows()


def verdict(pair):
    kind, text = pair
    getattr(st, kind)(text)


# Fester Kopf: nur der Titel bleibt beim Scrollen oben (Stylesheet in kpj_layout.py).
st.html(kopf_css(dunkles_design()))
with st.container(key="kopf"):
    st.title("📐 Projektion auf Kegel – Moreau, Spektralformeln und die Kegel ohne geschlossene Form")
st.markdown(
    """
**Die Projektion auf einen Kegel ist das Gegenstück zum Newton-Schritt: billig, aber man braucht viele.** Für die symmetrischen Kegel ist sie eine **Spektralformel** (Orthant: abschneiden, Lorentz-Kegel: aus `t ± ‖v̄‖`,
PSD-Kegel: negative Eigenwerte abschneiden); für den Exponential- und den Potenzkegel gibt es **keine geschlossene Form, aber eine Gleichung in einer Variablen**. Was alle fünf eint, ist die **Moreau-Zerlegung**
`v = P_K(v) + P_{K°}(v)` mit dem Polarkegel `K° = −K*`: sie ist zugleich ein **Zertifikat der Projektion** ohne Referenzlöser. Und die Formel `P(v) − P(−v) = v` gilt nur für **selbstduale** Kegel.
Stück 8 der Reihe „Konvexe und konische Optimierung“; mit ihm beginnt der zweite Strang (Erste Ordnung).

Weiter unten: **fünf Experimente** (**🔬**), **🚧 Wo die Annahmen enden** und der Expander **📐 Mathematische Formulierung**.
"""
)

with st.expander("So funktioniert es", expanded=True):
    st.markdown(
        """
1. **Projektion:** `P_K(v)` ist der nächste Punkt des Kegels `K` zu `v`. Liegt `v` in `K`, ist es `v`; liegt `v` im Polarkegel `K° = −K*`, ist es `0`; sonst liegt `P_K(v)` auf dem Rand.
2. **Moreau als Zertifikat:** `p` ist genau dann die Projektion, wenn `p ∈ K`, `v − p ∈ K°` und `⟨p, v − p⟩ = 0` gilt. Die Demo prüft das an jedem Punkt, ohne einen zweiten Löser.
3. **Exponential- und Potenzkegel:** auf dem Rand ist `v = y u(ρ) + λ n(ρ)` mit dem Randstrahl `u` und der Normalen `n`; die Bedingung, dass `v` in der Ebene von `u` und `n` liegt, ist eine Gleichung in `ρ` (Vorzeichenwechsel auf einem Gitter, dann Nullstellensuche).
4. **Abwechselnde Projektionen:** zwischen dem Kegel und dem Bild `{b − A x: cᵀx ≤ γ}` hin- und herprojizieren, bis beide sich berühren: ein Kegelprogramm nur mit Projektionen.
5. **Der Preis:** jede Projektion ist billig (Experiment 3), aber es sind mit dem gewählten Start Hunderte bis Tausende (Experiment 4), wo die Barriere aus Stück 5 bis 7 etwa 30 bis 45 Newton-Schritte brauchte.
        """
    )

st.caption("🔗 Die Adresszeile oben spiegelt Ihre aktuelle Konfiguration wider – einfach kopieren, um ein Szenario zu teilen.")

load_permalink_settings()
init_session_state_defaults()

with st.sidebar:
    sidebar_presets()  # Schnellstart oben, das aktive Preset ist hervorgehoben
    st.header("⚙️ Einstellungen")
    st.markdown("**Experiment 1: der Punkt**")
    kind = st.selectbox("Kegel", P.KINDS, key="kind_select", format_func=P.KIND_NAMES.get, help="Auf welchen Kegel das Hauptdiagramm projiziert; die Tabelle in Experiment 1 zeigt immer alle fünf.")
    alpha = st.select_slider("Exponent a (nur Potenzkegel)", options=C.ALPHA_CHOICES, key="alpha_select", format_func=alpha_text, help="Potenzkegel u^a v^(1−a) ≥ |w|; a = 0,5 ist der gedrehte Lorentz-Kegel.")
    vx = st.slider("v₁", C.V_MIN, C.V_MAX, step=C.V_STEP, key="vx_slider", help="Erste Koordinate (Orthant: v₁; Lorentz: t; PSD 2×2: a; Exponential: x; Potenz: u).")
    vy = st.slider("v₂", C.V_MIN, C.V_MAX, step=C.V_STEP, key="vy_slider", help="Zweite Koordinate (PSD 2×2: √2 b; Exponential: y; Potenz: v).")
    vz = st.slider("v₃", C.V_MIN, C.V_MAX, step=C.V_STEP, key="vz_slider", help="Dritte Koordinate (PSD 2×2: c; Exponential: z; Potenz: w).")
    st.markdown("**Experiment 4: die Wolke**")
    n = st.slider("Punkte n", *bounds("n_slider"), key="n_slider", help=f"Wie viele Punkte die Wolke hat; Experiment 4 rechnet mit den ersten {C.N_POCS}.")
    stretch = st.slider("Streckung", *bounds("stretch_slider"), step=0.1, key="stretch_slider", help="Wie viel breiter die Wolke entlang ihrer Längsachse ist als quer dazu.")
    angle = st.slider("Drehung (Grad)", *bounds("angle_slider"), key="angle_slider", help="Winkel der Längsachse gegen die x-Achse.")
    seed = st.number_input("Zufalls-Seed der Wolke", *bounds("seed_input"), key="seed_input", step=1)
    st.button("🎲 Neue Wolke generieren", width="stretch", on_click=randomize_seed)
    eps = st.select_slider("Schlupf ε (Experiment 4)", options=C.EPS_CHOICES, key="eps_select", format_func=eps_text,
                           help="Wie weit der Schlupf γ = R*(1 + ε) über dem Optimum liegt, den die abwechselnden Projektionen erreichen müssen; das Diagramm der Abstände gilt für diesen Wert.")

sync_query_params({"kind_select": kind, "alpha_select": float(alpha), "vx_slider": float(vx), "vy_slider": float(vy), "vz_slider": float(vz), "n_slider": int(n), "stretch_slider": float(stretch),
                   "angle_slider": int(angle), "seed_input": int(seed), "eps_select": float(eps)})

settings = Settings(kind, float(alpha), round(float(vx), 6), round(float(vy), 6), round(float(vz), 6), int(n), float(stretch), int(angle), int(seed), float(eps))
view = projection_view(settings)

# --- Ergebnis ------------------------------------------------------------------------------------------------------------------------------

st.markdown("## 🎯 Ergebnis")
m1, m2, m3 = st.columns(3)
m4, m5 = st.columns(2)
m1.metric("Projektion P(v)", vec_text(view["p"], 3), help=f"Der nächste Punkt des Kegels ({cone_label(kind, alpha)}) zu v = {vec_text(view['v'])}.")
m2.metric("Abstand ‖v − P(v)‖", de(view["distance"], 4), help="Abstand von v zum Kegel.")
m3.metric("Lage von v", view["case"], help="Im Kegel (Projektion = v), im Polarkegel (Projektion = 0) oder dazwischen (Projektion auf dem Rand).")
m4.metric("⟨p, v − p⟩", sci(view["inner"]), help="Moreau: Projektion und Rest stehen senkrecht aufeinander.")
m5.metric("Moreau-Probe", "besteht" if view["certificate_ok"] else "besteht nicht", help="p im Kegel, v − p im Polarkegel, ⟨p, v − p⟩ = 0 (Toleranz 10⁻⁷ relativ).")
st.plotly_chart(build_cone3d(kind, float(alpha), view["v"], view["p"]), width="stretch", key="main_chart")
st.caption("Randfläche des Kegels (durchscheinend), der Punkt v (rot), seine Projektion P(v) (grün) und der Rest (gestrichelt). Mit der Maus drehbar; auf dem Telefon beim Scrollen die Abbildung nicht berühren.")

if st.button("📄 PDF der aktuellen Projektion erstellen", key="pdf_make"):
    st.session_state["pdf_made"] = (settings, build_pdf(settings, view))
made = st.session_state.get("pdf_made")
if made and made[0] == settings:
    st.download_button("⬇️ PDF herunterladen", data=made[1], file_name="kegel-projektion.pdf", mime="application/pdf", key="pdf_download")

# --- Kernfrage -----------------------------------------------------------------------------------------------------------------------------

st.markdown("## ❓ Kernfrage: Was ist die Projektion auf einen Kegel, und was kostet sie?")
verdict(verdict_projection(settings, view))

st.markdown("---")

# --- Experimente ---------------------------------------------------------------------------------------------------------------------------

st.subheader("🔬 Experiment 1: Dasselbe v, fünf Kegel")
st.caption("Der Punkt v der Seitenleiste in allen fünf Kegeln, mit der Formel jedes Kegels.")
table = projection_table(settings)
st.dataframe(pd.DataFrame([{"Kegel": r["name"], "Formel": r["formula"], "P(v)": vec_text(r["p"]), "Abstand": de(r["distance"], 4), "Lage von v": r["case"], "Moreau-Probe": "besteht" if r["certificate_ok"] else "besteht nicht",
                            "selbstdual": "ja" if r["self_dual"] else "nein"} for r in table]), hide_index=True, width="stretch")
verdict(verdict_table(settings, table))

st.markdown("---")

st.subheader("🔬 Experiment 2: Moreau und Selbstdualität")
st.caption(f"An {C.MOREAU_POINTS} Zufallspunkten je Kegel: die Formel `v = P(v) − P(−v)` der selbstdualen Kegel, die Moreau-Probe, Idempotenz und Nichtexpansivität.")
mrows = _moreau()
st.plotly_chart(build_moreau(mrows), width="stretch", key="moreau_chart")
st.dataframe(pd.DataFrame([{"Kegel": r["cone"], "selbstdual": "ja" if r["self_dual"] else "nein", "größte Abweichung von v = P(v) − P(−v)": sci(r["mirror"]), "Moreau-Probe besteht": f"{100 * r['certificate']:.1f} %".replace(".", ","),
                            "Idempotenz (größte Abweichung)": sci(r["idempotent"]), "Verletzungen der Nichtexpansivität": r["nonexpansive_violations"], "Punkte im Kegel": f"{100 * r['inside']:.0f} %",
                            "Punkte im Polarkegel": f"{100 * r['polar']:.0f} %"} for r in mrows]), hide_index=True, width="stretch")
verdict(verdict_moreau(mrows))

st.markdown("---")

st.subheader("🔬 Experiment 3: Kosten einer Projektion")
st.caption("Zeit je Projektion auf diesem Rechner (beste von drei Läufen, ein BLAS-Thread): linear für den Lorentz-Kegel, mit der Eigenzerlegung für den PSD-Kegel, mit einer Gleichung für Exponential- und Potenzkegel. Zeiten werden nicht getestet.")
cost = _cost()
st.plotly_chart(build_cost(cost), width="stretch", key="cost_chart")
st.dataframe(pd.DataFrame([{"PSD-Größe k": r["k"], "Zeit je Matrix (µs)": de(r["us_per_matrix"], 1), "Zeit je Koordinate (ns)": de(r["ns_per_coordinate"], 0), "Matrizen im Stapel": r["count"]} for r in cost["psd"]]), hide_index=True, width="stretch")
verdict(verdict_cost(cost))

st.markdown("---")

st.subheader("🔬 Experiment 4: Abwechselnde Projektionen")
st.caption(f"Die Kugel der ersten {C.N_POCS} Punkte der Wolke in vier Kegeln, nur mit Projektionen gelöst: zwischen dem Kegel und `{{b − A x: cᵀx ≤ γ}}` wechseln, Abbruch bei Abstand 10⁻⁷ oder nach {C.MAX_ITER} Schritten.")
prows = _pocs(settings.n, settings.stretch, settings.angle, settings.seed)
st.plotly_chart(build_pocs(prows), width="stretch", key="pocs_chart")
st.plotly_chart(build_rate(prows, settings.eps), width="stretch", key="rate_chart")
st.dataframe(pd.DataFrame([{"Kegel": r["name"], "Zeilen": r["rows"], "R*": de(r["rstar"], 5), "Barriere: ν": r["nu"], "Barriere: Newton-Schritte": r["newton"],
                            **{f"Schritte bei {eps_text(x['eps'])}": (str(x["iterations"]) if x["converged"] else f"{x['iterations']} (nicht gelöst)") for x in r["results"]},
                            f"µs je Schritt (ε = {eps_text(settings.eps)})": de(next(x for x in r['results'] if x['eps'] == settings.eps)["per_step_us"], 0)} for r in prows]), hide_index=True, width="stretch")
verdict(verdict_pocs(prows, settings.eps))

st.markdown("---")

st.subheader("🔬 Experiment 5: Wann ist eine Projektion richtig?")
st.caption("Zwei Fassungen der Projektion auf den Exponentialkegel, geprüft allein an der Moreau-Probe: die erste (Maximum auf einem festen Gitter) und die zweite (Nullstelle von ⟨v, u × n⟩).")
erows = _expcheck()
st.plotly_chart(build_expcheck(erows), width="stretch", key="expcheck_chart")
st.dataframe(pd.DataFrame([{"Klasse von Punkten": r["label"], "Punkte": r["count"], "Gitter + Maximum: fällt durch": r["grid_failures"], "Wurzel: fällt durch": r["root_failures"],
                            "größter Abstandsunterschied": sci(r["worst_gap"])} for r in erows]), hide_index=True, width="stretch")
verdict(verdict_expcheck(erows))

st.markdown("---")

# --- Grenzen -------------------------------------------------------------------------------------------------------------------------------

st.subheader("🚧 Wo die Annahmen enden")
st.markdown(
    """
| Annahme | Was passiert, wenn sie verletzt ist | Wer setzt an |
|---|---|---|
| **Eine Projektion ist billig, also ist das Verfahren billig** | Der Schritt ist billig (Experiment 3), aber es sind mit dem gewählten Start 100 bis 2 000 Schritte (mit dem Schwerpunkt als Start deutlich weniger), wo die Barriere etwa 30 bis 45 Newton-Schritte braucht (Experiment 4); je mehr Schritte nötig sind, desto weniger zählt die billige Projektion. | Stück 9 (ADMM), Stück 10 |
| **`P(v) − P(−v) = v` gilt für jeden Kegel** | Nur für selbstduale Kegel (Orthant, Lorentz, PSD). Für Exponential- und Potenzkegel (auch a = 0,5 in diesen Koordinaten, wo der gedrehte Lorentz-Kegel zwar symmetrisch, aber nicht selbstdual ist) verfehlt sie `v` (Experiment 2). | – |
| **Die Moreau-Probe findet jeden Fehler** | Sie ist bis auf die Toleranz zu lesen. An Rändern mit Spitze (Potenzkegel mit a nahe 0 oder 1) kann der Rest die Dualkegel-Bedingung knapp verfehlen, obwohl der Abstand auf 10⁻⁹ stimmt; die Probe besteht dort bei 98 bis 99 % statt bei allen Punkten. | – |
| **Die Wurzel-Fassung gilt für beliebige Punkte** | Für Punkte, deren Randstrahl außerhalb des Gitters liegt (x/y jenseits von ±700 beim Exponentialkegel, ρ jenseits von e^±250 beim Potenzkegel), wählt sie die Fläche; das hat dort Abstandsfehler bis 4·10⁻⁶ erzeugt (gemessen) und ist keine Garantie. | – |
| **PSD kostet `k³`** | Gemessen ist bis k = 128 eine Steigung von etwa 2,1 bis 2,3 (die Konstante der kleinen Größen und LAPACK verdecken den kubischen Teil); der Text sagt „bis k = 128 gemessen“, nicht `O(k³)`. | – |
| **Abwechselnde Projektionen konvergieren mit festem Tempo** | Die Zahlen (Experiment 4) sind Messbefunde dieser Wolke, dieses Starts und dieses Abbruchkriteriums; mit schrumpfendem Schlupf nähern sich die beiden Mengen einem Punkt und das Verfahren wird langsamer. Die Rate ist nicht hergeleitet. | – |
| **Die Zeiten gelten allgemein** | Sie hängen vom Rechner ab; die Gitter und Schleifen der Exponential- und Potenzkegel sind nicht die schnellsten Verfahren (Friberg nennt eine Wurzelsuche mit wenigen Schritten). Zeiten werden nicht getestet. | – |
"""
)

st.markdown("---")

with st.expander("📐 Mathematische Formulierung"):
    st.markdown(
        r"""
**Projektion und Moreau.** Für einen abgeschlossenen konvexen Kegel $K$ ist $P_K(v) = \arg\min_{x \in K} \|x - v\|$. Der Polarkegel ist $K^\circ = -K^*$ mit $K^* = \{z: \langle z, x\rangle \ge 0\ \forall x \in K\}$.
Moreau-Zerlegung: $v = P_K(v) + P_{K^\circ}(v)$ und $\langle P_K(v), P_{K^\circ}(v)\rangle = 0$. Umgekehrt ist jedes $p \in K$ mit $v - p \in K^\circ$ und $\langle p, v - p\rangle = 0$ die Projektion (Optimalitätsbedingung).
Ist $K$ selbstdual ($K^* = K$ im gegebenen Skalarprodukt), gilt $P_{K^\circ}(v) = -P_K(-v)$, also $v = P_K(v) - P_K(-v)$; sonst nicht.

**Spektralformeln.** Orthant: $P(v) = \max(v, 0)$. Lorentz-Kegel ($v = (t, \bar v)$): $P(v) = v$ für $\|\bar v\| \le t$, $P(v) = 0$ für $\|\bar v\| \le -t$, sonst $\tfrac{\|\bar v\| + t}{2}\,(1, \bar v/\|\bar v\|)$.
PSD-Kegel: $V = Q \operatorname{diag}(\lambda) Q^\top \mapsto Q \operatorname{diag}(\max(\lambda, 0)) Q^\top$.

**Exponentialkegel.** Liegt $v$ weder in $K_{\exp}$ noch in $K_{\exp}^\circ$, liegt $P(v)$ auf dem Rand: $P(v) = y\,u(\rho)$ mit $u = (\rho, 1, e^\rho)$ oder auf der Fläche $y = 0$ ($x \le 0$, $z \ge 0$). Mit der Normalen $n(\rho) = (e^\rho, e^\rho(1 - \rho), -1)$, $u \perp n$, gilt
$v = y\,u + \lambda\,n$ mit $\lambda \ge 0$; damit $v$ in der Ebene von $u$ und $n$ liegt, muss $\langle v, u \times n\rangle = 0$ sein: eine Gleichung in $\rho$. Aus der Wurzel folgen $y = \langle v, u\rangle/\|u\|^2$ und $P(v) = y\,u$; unter mehreren Kandidaten und der Fläche gewinnt der mit dem kleinsten Abstand.
**Potenzkegel** $P_a = \{u^a v^{1-a} \ge |w|\}$: dasselbe mit $u(\rho) = (\rho, 1, \rho^a)$ (für $w \ge 0$, sonst gespiegelt), $n = (-a\rho^{a-1}, -(1-a)\rho^a, 1)$ und den Strahlen $(u, 0, 0)$, $(0, v, 0)$.

**Abwechselnde Projektionen.** Für $\min c^\top x$ mit $b - Ax \in K$ und einer Schranke $\gamma$ sei $C_\gamma = \{b - Ax: c^\top x \le \gamma\}$. Die Projektion auf $C_\gamma$ ist geschlossen: $x_0 = (A^\top A)^{-1}A^\top(b - s)$; ist $c^\top x_0 > \gamma$, so $x = x_0 - \lambda (A^\top A)^{-1}c$ mit $\lambda = (c^\top x_0 - \gamma)/(c^\top (A^\top A)^{-1} c)$.
Dann $s \leftarrow P_K(P_{C_\gamma}(s))$ bis $\|y - s\|$ klein ist. Für $\gamma > p^*$ haben die Mengen einen gemeinsamen Punkt mit Innerem; die Folge nähert ihn, bei $\gamma \to p^*$ schrumpft der Durchschnitt auf einen Punkt.

**Literatur.** N. Parikh und S. Boyd: *Proximal Algorithms*, Foundations and Trends in Optimization 1 (3), 2014, S. 127–239 (Projektion als Prox der Indikatorfunktion, Moreau-Zerlegung; die Seite von Boyd nennt für den Vorabdruck 123–231).
H. H. Bauschke und J. M. Borwein: *On projection algorithms for solving convex feasibility problems*, SIAM Review 38 (3), 1996, S. 367–426 (abwechselnde Projektionen).
H. A. Friberg: *Projection onto the exponential cone: a univariate root-finding problem*, Optimization Online (Preprint; Zeitschrift und Jahr nicht bestätigt). Die Wurzelgleichung dieser Demo ist eine eigene Herleitung, in den Tests gegen cvxpy geprüft.
L. T. K. Hien: *Differential properties of Euclidean projection onto power cone*, Mathematical Methods of Operations Research 82 (3), 2015, S. 265–284.
        """
    )

st.markdown("---")
st.caption(
    "Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – "
    "Operations Research und Machine Learning ([Über mich](https://sebastianhanisch.net/ueber-mich.html)). "
    "Mehr zur Reihe: [Konvexe Optimierung: vom Beweis zur Verfahrenswahl](https://sebastianhanisch.net/konzepte-konvexe-optimierung.html)."
)
