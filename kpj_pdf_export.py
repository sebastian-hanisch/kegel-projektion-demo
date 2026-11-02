"""PDF-Export der aktuellen Projektion mit der Moreau-Probe (fpdf2). Kernschriften sind latin-1: Umlaute gehen, Sonderzeichen werden ersetzt."""

from __future__ import annotations

from fpdf import FPDF

from kpj_evaluation import Settings, cone_label, de, sci, vec_text

_REPLACE = {"→": "->", "–": "-", "—": "-", "−": "-", "≤": "<=", "≥": ">=", "·": "*", "×": "x", "ν": "nu", "λ": "lambda", "β": "beta", "α": "alpha", "κ": "kappa", "€": "EUR", "⁻": "-", "⁰": "0", "¹": "1", "²": "2", "³": "3",
            "⁴": "4", "⁵": "5", "⁶": "6", "⁷": "7", "⁸": "8", "⁹": "9", "„": '"', "“": '"', "ᵀ": "^T", "ℓ": "l", "∞": "inf", "√": "sqrt", "‖": "|", "°": "o"}


def _s(text) -> str:
    out = str(text).replace("**", "").replace("`", "")
    for old, new in _REPLACE.items():
        out = out.replace(old, new)
    return out.encode("latin-1", "replace").decode("latin-1")


def build_pdf(s: Settings, view: dict) -> bytes:
    pdf = FPDF(orientation="L", unit="mm", format="A4")
    pdf.set_auto_page_break(auto=True, margin=12)
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 15)
    pdf.cell(0, 9, _s("Projektion auf einen Kegel"), new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 10)
    pdf.multi_cell(0, 5, _s(f"Kegel: {cone_label(s.kind, s.alpha)}. Punkt v = {vec_text(view['v'])}. Alles simuliert."), new_x="LMARGIN", new_y="NEXT")
    pdf.ln(3)
    rows = [("Projektion P(v)", vec_text(view["p"])), ("Rest v - P(v)", vec_text(view["r"])), ("Abstand", de(view["distance"], 6)), ("Lage von v", view["case"]), ("<p, v - p>", sci(view["inner"])),
            ("p im Kegel", "ja" if view["in_cone"] else "nein"), ("v - p im Polarkegel", "ja" if view["in_polar"] else "nein"), ("Moreau-Probe", "besteht" if view["certificate_ok"] else "besteht nicht"),
            ("selbstdual", "ja" if view["self_dual"] else "nein"), ("|v - (P(v) - P(-v))|", sci(view["mirror_deviation"]))]
    pdf.set_font("Helvetica", "B", 9)
    pdf.cell(80, 6, "Kennzahl", border=1)
    pdf.cell(100, 6, "Wert", border=1)
    pdf.ln()
    pdf.set_font("Helvetica", "", 9)
    for k, v in rows:
        pdf.cell(80, 5.5, _s(k), border=1)
        pdf.cell(100, 5.5, _s(v), border=1)
        pdf.ln()
    pdf.ln(3)
    pdf.set_font("Helvetica", "I", 8)
    pdf.multi_cell(0, 4, _s("Quelle: kegel-projektion-demo, Sebastian Hanisch."), new_x="LMARGIN", new_y="NEXT")
    return bytes(pdf.output())
