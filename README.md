# Projektion auf Kegel – Moreau, Spektralformeln und die Kegel ohne geschlossene Form – Streamlit-Demo

<!-- TODO nach dem Deploy: direkt hierher den Live-Link setzen (Leerzeile davor und danach):
**[→ Demo live ausprobieren](https://sebastianhanisch-kegel-projektion-demo.streamlit.app/)** -->

Stück 8 von 16 der **Konzepte-Reihe „Konvexe und konische Optimierung“** für die Website „Sebastian Hanisch – Operations Research und Machine Learning“. Wie jede Konzepte-Demo dreht sich alles um **ein** Verfahren mit wachsendem Beispiel: ein Löser für Kegelprogramme, dessen Kegel von Stück zu Stück wächst. Mit diesem Stück beginnt der **zweite Strang (Erste Ordnung)**: dieselben Kegelprogramme, anders gelöst, mit Projektionen statt Newton-Schritten. Vorgänger sind [konvexitaet-zertifikat-demo](https://github.com/sebastian-hanisch/konvexitaet-zertifikat-demo) (Stück 1), [kegelform-demo](https://github.com/sebastian-hanisch/kegelform-demo) (Stück 2), [kruemmungsregeln-demo](https://github.com/sebastian-hanisch/kruemmungsregeln-demo) (Stück 3), [kegel-dualitaet-demo](https://github.com/sebastian-hanisch/kegel-dualitaet-demo) (Stück 4), [kegel-barriere-demo](https://github.com/sebastian-hanisch/kegel-barriere-demo) (Stück 5), [kegel-primal-dual-demo](https://github.com/sebastian-hanisch/kegel-primal-dual-demo) (Stück 6: Spektralzerlegung als Mittel im Newton-System) und [kegel-exponential-demo](https://github.com/sebastian-hanisch/kegel-exponential-demo) (Stück 7: Exponential- und Potenzkegel, ihr Dualkegel). Nachfolger ist ADMM (Stück 9). Die Linien-Spezifikation und die Einbindung in die Website folgen, wenn die Reihe vollständig ist.

## Kernfrage

**Was ist die Projektion auf einen Kegel, und was kostet sie?** Die Projektion `P_K(v)` ist der nächste Punkt des Kegels `K` zu `v`. Für die symmetrischen Kegel ist sie eine **Spektralformel** (Orthant: negative Koordinaten abschneiden; Lorentz-Kegel: geschlossen aus `t ± ‖v̄‖`; PSD-Kegel: negative Eigenwerte abschneiden). Für den Exponential- und den Potenzkegel gibt es **keine geschlossene Form, aber eine Gleichung in einer Variablen** (die Randkurve). Was alle fünf Kegel eint, ist die **Moreau-Zerlegung** `v = P_K(v) + P_{K°}(v)` mit dem Polarkegel `K° = −K*` und `P_K(v) ⟂ P_{K°}(v)`: sie ist zugleich ein **Zertifikat der Projektion** ohne Referenzlöser (`p ∈ K`, `v − p ∈ K°`, `⟨p, v − p⟩ = 0`). Und die Formel `P_K(v) − P_K(−v) = v`, die man von den Standardkegeln kennt, gilt **nur für selbstduale Kegel**. Der zweite Teil ist der Preis: ein Projektionsschritt ist billig, aber abwechselnde Projektionen brauchen für dieselben Kugeln, für die Stück 5 bis 7 etwa 30 bis 45 Newton-Schritte brauchten, mit dem gewählten Start (verschobener Mittelpunkt) **Hunderte bis Tausende Schritte**; mit dem Schwerpunkt als Start sind es ein bis rund 150.

Das Beispiel bleibt die kleinste umschließende Kugel einer Punktwolke, jetzt in vier Kegeln (Lorentz, PSD, Potenz als ℓ₃-Kugel, Exponential als log-sum-exp-Kugel).

## Modell

- **Kern** (`kegel_kern.py`, wortgleich mit Stück 2 bis 7, unverändert): Lorentz- und PSD-Kegel-Programme der Kugel und ihre Barrierenmethode als Referenz.
- **Projektionen** (`kpj_projection.py`): alle fünf Kegel **gestapelt** gerechnet (N Kegel als Matrix `(N, d)`); Exponential- und Potenzkegel über die **Wurzel** der Gleichung `⟨v, u × n⟩ = 0` (Vorzeichenwechsel auf einem dichten Gitter, dann ein Illinois-Verfahren, die Fläche `y = 0` bzw. die Strahlen als weitere Kandidaten, der nächste gültige Punkt gewinnt); die Zugehörigkeit zu Kegel, Dualkegel und Polarkegel mit Toleranz relativ zur Größe des Punktes; das Zertifikat `certificate`. Die **erste Fassung** der Exponentialkegel-Projektion (`proj_exp_grid`: Maximum auf einem festen Gitter) bleibt als warnendes Beispiel für Experiment 5.
- **Abwechselnde Projektionen** (`kpj_pocs.py`): zwischen dem Kegel `K` und dem Bild `C_γ = {b − A x: cᵀx ≤ γ}` hin- und herprojizieren; die Projektion auf `C_γ` ist geschlossen (Kleinste-Quadrate-Punkt, bei Verletzung von `cᵀx ≤ γ` zurückgeschoben).
- **Formen und Referenz** (`kpj_forms.py`, `kpj_barrier.py`, `kpj_cones.py`): die vier Kugel-Programme und die Barrierenmethode für `R*`, aus Stück 6 und 7 übernommen (je ein eigenes Präfix, wie in der Reihe üblich).
- **Kein Eingriff in den Kern:** alles Neue steht in `kpj_*.py`; BLAS-Threads laufen für die Zeitmessung und die Barriere mit einem Thread (`threadpoolctl`).

## Methodik

Fünf Experimente mit live berechnetem Verdikt: (1) dasselbe `v` in allen fünf Kegeln, mit der Formel jedes Kegels; (2) Moreau und Selbstdualität (Spiegelformel, Moreau-Probe, Idempotenz, Nichtexpansivität an je 2 000 Zufallspunkten); (3) die Kosten einer Projektion (Lorentz je Koordinate, PSD gegen die Größe `k`, Exponential- und Potenzkegel je Kegel); (4) abwechselnde Projektionen für die Kugel der Wolke in vier Kegeln, Schritte gegen den Schlupf `ε = γ/R* − 1` neben den Newton-Schritten der Barriere; (5) wann eine Projektion richtig ist: die Gitter-Fassung gegen die Wurzel-Fassung, geprüft allein an der Moreau-Probe.

## Befunde (gemessen, keine Behauptungen)

Alle Zahlen stehen in `tests/test_claims.py` (wo dort nur ein Band geprüft wird, wegen Rundung und Rechner, steht die gemessene Zahl trotzdem hier). Zeiten gelten für den Entwicklungsrechner und werden nicht getestet.

| Frage | Befund |
|---|---|
| Standardfall | `v = (1; 0,5; −0,5)` liegt weder im Exponentialkegel noch im Polarkegel: **P(v) = (0,023; 0,043; 0,074)**, Abstand 1,2215, Randpunkt mit `x/y = 0,5325`, `⟨p, v − p⟩ = −7·10⁻¹⁸`, die Moreau-Probe besteht (in allen fünf Kegeln). Die Spiegelformel verfehlt `v` um 0,457 |
| Moreau und Selbstdualität (je 2 000 Zufallspunkte) | Orthant, Lorentz- und PSD-Kegel: `v = P(v) − P(−v)` auf **3·10⁻¹⁵**, die Moreau-Probe besteht bei 100 %. **Exponentialkegel: Abweichung bis 4,18; Potenzkegel a = 0,3 bis 2,06; a = 0,5 bis 2,20** (auch der gedrehte Lorentz-Kegel `uv ≥ w²` ist in diesen Koordinaten nicht selbstdual: sein Dualkegel ist `uv ≥ w²/4`). Moreau-Probe: Exponentialkegel 99,85 %, Potenzkegel 100 %. Alle idempotent (Abweichung unter 10⁻¹⁴) und nichtexpansiv (0 Verletzungen) |
| Kosten einer Projektion (ein BLAS-Thread, beste von 3 Läufen) | Lorentz **43 ns je Koordinate**. PSD je Matrix 2,8 µs (k = 4), 9,4 µs (8), 40 µs (16), 151 µs (32), 636 µs (64), 3,4 ms (128): Steigung der Zeit gegen `k` **2,1** (Plan-Messung 2,34; nicht 3: die Konstante der kleinen Größen und LAPACK verdecken den kubischen Teil bis k = 128); je Koordinate ist der PSD-Kegel bei k = 32 etwa 6-mal teurer als Lorentz. Exponentialkegel **7 µs**, Potenzkegel **14 µs** je Kegel (gestapelt, Gitter und Wurzelsuche) |
| Abwechselnde Projektionen (Kugel der ersten 8 Punkte, Start mit verschobenem Mittelpunkt, Abbruch bei Abstand 10⁻⁷) | Schlupf 10 % / 3 % / 1 % / 0,3 %: **Lorentz 215 / 216 / 216 / 216**, **PSD 213 / 214 / 214 / 214**, **ℓ₃-Kugel (Potenzkegel) 180 / 434 / 491 / 513**, **log-sum-exp (Exponentialkegel) 1 959 / 1 971 / 1 981 / 2 023 Schritte**. Die Barriere aus Stück 5 bis 7 braucht für dieselben Programme **31, 31, 34 und 43 Newton-Schritte** (ν = 16, 24, 56, 104) bis 10⁻⁹ |
| Die Optimalwerte der vier Kugeln (erste 8 Punkte der Standardwolke) | Lorentz und PSD **7,0359**, ℓ₃-Kugel **6,2683**, log-sum-exp β = 10 **5,0463**; die Endwerte der abwechselnden Projektionen liegen zwischen `R*` und `R*(1 + ε)` |
| Zwei Fassungen der Exponentialkegel-Projektion | Die erste (Maximum auf einem festen Gitter von −40 bis 40) fällt an der Moreau-Probe bei **203 von 1 600** Punkten durch, **202 von 400** bei `x/y` von +5 bis +20 und `z` bis 10⁶ (Abstand bis 0,1 über der Wurzel-Fassung); die Wurzel-Fassung an **keinem**. Bei `x/y` um −60 und bei mäßigem `z` fallen beide nie durch |
| Moreau-Probe am Rand mit Spitze (Potenzkegel, gemischte Größenordnungen) | Besteht bei **100 %** (a = 0,3 und 0,5), 99,7 % (a = 0,75), **98 %** (a = 0,1) und **98 %** (a = 0,9): der Rest verfehlt die Dualkegel-Bedingung knapp, obwohl der Abstand zu cvxpy auf 10⁻⁹ stimmt; die Probe ist bis auf die Toleranz zu lesen |
| Gegen cvxpy/Clarabel | Abstand zum Kegel im getesteten Bereich höchstens 10⁻⁷ über dem der Referenz, für Orthant, Lorentz, PSD 2×2, Exponentialkegel (`x/y` von −60 bis +20) und Potenzkegel (a = 0,1 bis 0,9); die Punkte selbst weichen bis 10⁻⁵ ab (die Referenz ist am flachen Optimum nicht genauer) |

## Befunde und Korrekturen gegenüber dem Plan

- **Abwechselnde Projektionen scheitern nicht bei 0,3 %.** Der Plan nannte aus der Phase 0 (Wolke mit 20 Punkten, Start `x₀` des Programms): Lorentz und PSD 440 bis 470 Schritte bis ε = 1 % und **nicht gelöst** nach 20 000 bei 0,3 %. Der Bau rechnet mit den **ersten 8 Punkten** (Laufzeit) und einem **anderen Start** (der Mittelpunkt um `R*` verschoben, der Radius das Doppelte des größten Abstands): dort sind **alle 16 Läufe gelöst**, mit 215 bis 2 023 Schritten. Die Schrittzahl hängt am Start und an der Wolke; die Aussage „scheitert bei 0,3 %“ trägt für diesen Aufbau nicht. Was bleibt: mit diesem Start Hunderte bis Tausende Schritte gegen 31 bis 43.
- **Der Start entscheidet, ob das Experiment etwas zeigt.** Mit dem Schwerpunkt als Start ist die Kugel der ersten 8 Punkte schon für Schlupf ab etwa 1,5 % zulässig (der Schwerpunkt-Ball ist nur 1,5 % größer als `R*`): ein Schritt (gemessen bei 3 % und 10 %; bei 1 % sind es 144, bei 0,3 % 157 Schritte). Mit dem Start `R = 0` hängt die Schrittzahl gar nicht vom Schlupf ab (die Schranke wird nie aktiv). Deshalb der verschobene Start (in `start_point` dokumentiert).
- **Das Gitter des Potenzkegels musste weit werden.** Mit `ρ = e^τ` und `τ ∈ [−60, 60]` fehlte ein Randpunkt bei a = 0,9 (`ρ` um `e^63`, Abstand 7·10⁻⁴ über der Referenz); mit `τ ∈ [−250, 250]` stimmen alle a. Das ist numerisch meist bedeutungslos (`v ≈ 10⁻²⁶`), aber der Test findet es.
- **Wurzelsuche statt Bisektion.** Der Prototyp halbierte 48-mal je Intervall (3 ms für 48 Exponentialkegel); das Illinois-Verfahren braucht 12 Schritte bei gleicher Genauigkeit (Abstand zur Referenz unverändert 2·10⁻¹⁰).
- **Moreau-Probe an den Exponentialkegel-Ausreißern:** die 16 „Ausreißer“ der Phase 0 waren Randfälle `w = 0` der Dualkegel-Bedingung, die mit `e^(v/u)` unter 10⁻³⁰⁰ eine Toleranz brauchen; mit der Toleranz relativ zur Größe des Punktes besteht die Probe bei 99,85 % (3 von 2 000 fallen durch: einer an der Toleranz, zwei wegen der Gittergrenze `x/y` < −700, mit Abstandsfehlern bis 4·10⁻⁶).
- **Die erste Fassung ist nicht nur langsam, sondern falsch.** Der Plan nannte 63 von 100 Punkten über 10⁻⁸ gegenüber cvxpy; die Probe ohne Referenzlöser findet 202 von 400 in der Klasse mit großem `x/y`.
- **Quellen:** geprüft per Suche am 2026-11-02: Parikh und Boyd (Foundations and Trends in Optimization 1 (3), 2014, S. 127–239, DOI 10.1561/2400000003; die Boyd-Seite nennt 123–231 für den Vorabdruck), Bauschke und Borwein (SIAM Review 38 (3), 1996, S. 367–426, DOI 10.1137/S0036144593251710), Hien (Mathematical Methods of Operations Research 82 (3), 2015, S. 265–284, DOI 10.1007/s00186-015-0514-0). **Friberg** (*Projection onto the exponential cone: a univariate root-finding problem*) ist nur als Optimization-Online-Preprint bestätigt; Zeitschrift und Jahr widersprechen sich (2021 oder 2023), darum steht er so da. Inhalte der Arbeiten habe ich nicht nachgelesen; die Wurzelgleichung ist eigene Herleitung, in den Tests gegen cvxpy geprüft.
- **Port:** 9306 (`launch.json` und Netstat gelesen, frei).

## Ehrliche Grenzen

- **Die Zahlen der abwechselnden Projektionen sind Messbefunde** dieser Wolke (8 Punkte), dieses Starts und dieses Abbruchkriteriums (Abstand 10⁻⁷, höchstens 4 000 Schritte); die Rate ist nicht hergeleitet. Mit schrumpfendem Schlupf nähert sich der Durchschnitt einem Punkt und das Verfahren wird langsamer; wo es kippt, hängt am Aufbau (Phase 0: bei 0,3 %).
- **Projektionen sind kein Löser.** Die Demo baut kein Verfahren mit Splitting (Stück 9), keine Beschleunigung, kein Dykstra; sie zeigt den Baustein und seinen Preis.
- **Die Moreau-Probe ist bis auf die Toleranz zu lesen.** An Rändern mit Spitze (Potenzkegel mit a nahe 0 oder 1) fällt sie bei 1 bis 3 % der Punkte knapp durch, obwohl der Abstand stimmt.
- **Die Wurzel-Fassung ist begrenzt auf das Gitter** (Exponentialkegel `ρ ∈ [−700, 700]`, Potenzkegel `ρ ∈ [e⁻²⁵⁰, e²⁵⁰]`); darüber wählt sie die Fläche, was an Zufallspunkten jenseits von `x/y` = −700 Abstandsfehler bis 4·10⁻⁶ erzeugt hat (gemessen) und keine Garantie ist.
- **PSD ist nicht gemessen kubisch.** Die Steigung 2,1 bis 2,3 bis k = 128 ist ein Befund dieser Rechner; der Text sagt „bis k = 128 gemessen“, nicht `O(k³)`.
- **Zeiten hängen vom Rechner ab** und werden nicht getestet; die Gitter und Schleifen der Exponential- und Potenzkegel sind nicht die schnellsten Verfahren.
- **Nur die vier Kugeln,** keine Anwendung; der PSD-Kegel der Projektion ist 2×2 als Dreier-Vektor (svec), nicht größer.
- **Die Spiegelformel für a = 0,5.** Der Potenzkegel mit a = ½ ist ein symmetrischer Kegel (gedrehter Lorentz-Kegel), aber in den Koordinaten `(u, v, w)` und mit dem euklidischen Skalarprodukt nicht selbstdual; „selbstdual“ meint hier genau das.

## Verifikation

- **Projektionen** (`tests/test_projection.py`): Orthant, Lorentz (`(1; 3, 4) → (3; 1,8; 2,4)`) und PSD (`diag(2, −1) → diag(2, 0)`) von Hand; Exponentialkegel von Hand (`(0, 1, 2)` im Kegel, `(1, −1, −1)` im Polarkegel, `(−1, −2, 3) → (−1, 0, 3)` auf der Fläche); Zugehörigkeit und Dualkegel des Potenzkegels von Hand; je Kegel (sieben Fälle) die Moreau-Probe, Idempotenz, Nichtexpansivität, positive Homogenität, „nächster Punkt“; Spiegelformel für selbstduale Kegel und ihr Scheitern für die anderen; die Wurzel-Fassung gegen die Gitter-Fassung; das Zertifikat lehnt falsche Projektionen ab; PSD gegen die Eigenwertbeschneidung bis 5×5; `svec`/`smat` isometrisch.
- **Abwechselnde Projektionen** (`tests/test_pocs.py`): das Lorentz-Handprogramm (`t_{k+1} = (5 + t_k)/2`, der Abstand halbiert sich genau), Schranke unter dem Optimum ohne Lösung, Orthant in einem Schritt, PSD-Handprogramm, Abstand fällt monoton, Endwert zwischen `R*` und der Schranke, Zerlegung der Kegel, blockweise Projektion.
- **Formen** (`tests/test_forms.py`): antipodale Punkte von Hand in allen vier Kegeln, Lorentz- gleich PSD-Kugel gleich ℓ₂-Potenzkugel, Maße und Zulässigkeit des Starts.
- **Orakel** (`tests/test_oracle_kegel_projektion.py`): alle fünf Kegel gegen cvxpy mit Clarabel nach **Abstand**; Exponentialkegel bei `x/y` −60 bis +20; Potenzkegel bei a = 0,1 und 0,9; die Referenz besteht die Moreau-Probe.
- **Randfälle** (`tests/test_edge_cases.py`): Nullvektor, Randpunkte, Lorentz der Größe 2, PSD 1×1, extreme Verhältnisse, Skalen 10⁻⁸ bis 10⁸, alle Reglerecken, Klassifikation der Lage, Experimente an den kleinsten und größten Wolken.
- **Verdikte** (`tests/test_verdicts.py`, Werte von Hand gesetzt), **Claims** (`tests/test_claims.py`: jede Zahl dieser README), **Oberfläche** (AppTest: jeder Button, jeder Regler an Minimum und Maximum, alle Regler zugleich, jeder Kegel, jedes a, jeder Schlupf, PDF), Presets (die Zahlen der Hilfetexte), Permalink, Layout, Abbildungen, Footer, Laufzeit-Importe, Kern.
- **Umfang:** 182 Tests (10 davon, die cvxpy-Orakel gegen Clarabel, überspringen sich ohne cvxpy; mit cvxpy laufen alle 10 Orakel-Tests grün), rund 5,6 Minuten (2026-11-03, Windows; die AppTest-Läufe tragen den Löser der vier Kugeln).
- **Nicht gelaufen:** kein Mutationstest (seit 2026-10-28 nicht mehr Teil des Ablaufs). cvxpy ist Gegenprobe im dev-Satz; ohne es überspringen sich die Orakel-Tests. Linux-CI: noch nicht gelaufen (Repo noch nicht veröffentlicht).

## Dateistruktur

| Datei | Zweck |
|---|---|
| `.github/workflows/tests.yml` | CI: Push, Pull Request, wöchentlicher Canary |
| `.gitignore` | venv, Caches, Streamlit-Secrets |
| `.streamlit/config.toml` | Streamlit-Einstellungen (keine Nutzungsstatistik) |
| `app.py` | Streamlit-App (Skelett nach DEMO-PLAYBOOK §2) |
| `kegel_kern.py` | Gemeinsamer Kern der Reihe, wortgleich mit Stück 2 bis 7 |
| `kpj_projection.py` | Projektionen auf fünf Kegel (gestapelt), Zugehörigkeit, Moreau-Zertifikat |
| `kpj_pocs.py` | Abwechselnde Projektionen für Kegelprogramme |
| `kpj_forms.py` | Die Kugel der Wolke als Lorentz-, PSD-, Potenz- und Exponentialkegel-Programm (aus Stück 6 und 7) |
| `kpj_barrier.py`, `kpj_cones.py` | Barrierenmethode über Orthant, Exponential- und Potenzkegel für `R*` (aus Stück 7) |
| `kpj_constants.py` | Regler-Grenzen, feste Annahmen, Farben, Presets |
| `kpj_evaluation.py` | Experimente, Verdikte, Formatierung |
| `kpj_layout.py` | Fester Kopf (Stylesheet) |
| `kpj_pdf_export.py` | PDF-Export (fpdf2) |
| `kpj_presets.py` | Permalink, Presets (Schnellstart, aktives Preset hervorgehoben), Zufalls-Seed-Button |
| `kpj_scenario.py` | Punktwolken (gestreckt, gedreht) |
| `kpj_visualization.py` | Plotly-Abbildungen (2D-Achsen gesperrt, Kegel in 3D) |
| `requirements-dev.txt` | Laufzeit plus pytest, scipy und cvxpy (Gegenproben) |
| `requirements.txt` | Laufzeit-Bibliotheken (ungepinnt) |
| `tests/` | Projektionen, Abwechselnde Projektionen, Formen, Orakel, Randfälle, Verdikte, Claims, Presets, Layout, Abbildungen, AppTest, Footer, Laufzeit-Importe, Kern |

## Bewusst nicht umgesetzt

- **ADMM, Splitting, Dykstra, beschleunigte Projektionsverfahren:** Stück 9 und 10; hier nur abwechselnde Projektionen.
- **Prox-Operatoren anderer Funktionen** (ℓ₁, Rauschfilter): die Inverse-Linie (ISTA/FISTA, Plug-and-Play); nur Verweis.
- **Die Rate der abwechselnden Projektionen** herleiten: nicht gemacht.
- **Projektion auf Kegel höherer Dimension als drei außer Lorentz und PSD** (Potenzkegel mit mehreren Exponenten, Kegel der Relativentropie): nicht gebaut.
- **Ein eingeklammertes Newton-Verfahren** statt Gitter und Illinois für Exponential- und Potenzkegel: nur genannt (Friberg).
- **Mutationstest:** nicht Teil des Ablaufs.

## Lokal ausführen

```bash
python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

pip install -r requirements.txt
streamlit run app.py
```

Streamlit Cloud: Python-Version in den App-Einstellungen prüfen (lokal Python 3.14, CI Python 3.12); der Code verwendet nur Merkmale, die in Python 3.12 laufen. <!-- lint-ok: dezimalpunkt -->

## Tests ausführen

```bash
pip install -r requirements-dev.txt
pytest tests/ -v
```

Gebaut mit Streamlit, Plotly, NumPy, pandas, fpdf2 und threadpoolctl.

Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – Operations Research und Machine Learning ([Über mich](https://sebastianhanisch.net/ueber-mich.html)). Mehr zu den Demos: [Interaktive Demos](https://sebastianhanisch.net/demos.html).
