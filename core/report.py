from fpdf import FPDF
import os

FONT_DIR = os.path.join(os.path.dirname(__file__), "fonts")


def _safe(txt):
    """Convert any value to string, strip characters a font can't render."""
    if txt is None:
        return ""
    return str(txt).encode("latin-1", "replace").decode("latin-1")


def generate_pdf_report(stats, history, filename="dataset.csv", boxplot_paths=None):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=False)

    # Register Unicode-capable fonts
    regular = os.path.join(FONT_DIR, "Arial.ttf")
    bold = os.path.join(FONT_DIR, "Arial-Bold.ttf")
    if os.path.exists(regular) and os.path.exists(bold):
        pdf.add_font("AppFont", "", regular)
        pdf.add_font("AppFont", "B", bold)
        font_family = "AppFont"
    else:
        font_family = "Helvetica"

    # Header
    pdf.set_font(font_family, 'B', size=16)
    pdf.cell(0, 10, txt=_safe("DataDoctor - Cleaning Report"), ln=True, align='C')
    pdf.set_font(font_family, size=9)
    pdf.cell(0, 6, txt=_safe(f"File: {filename}"), ln=True, align='C')
    pdf.ln(4)

    # Health score
    pdf.set_font(font_family, 'B', size=13)
    pdf.cell(0, 8, txt=_safe(f"Health Score: {stats['health_score']:.1f}/100"), ln=True)

    pdf.set_font(font_family, size=10)
    pdf.cell(0, 6, txt=_safe(f"Rows: {stats['raw_rows']} -> {stats['clean_rows']}"), ln=True)
    pdf.cell(0, 6, txt=_safe(f"Columns: {stats.get('raw_cols','-')} -> {stats.get('clean_cols','-')}"), ln=True)
    pdf.cell(0, 6, txt=_safe(f"Missing fixed: {stats['missing_fixed']} | Outliers flagged: {stats['outliers_flagged']} | Duplicates removed: {stats['duplicates_removed']}"), ln=True)
    pdf.cell(0, 6, txt=_safe(f"Date columns standardized: {stats['dates_fixed']}"), ln=True)
    pdf.ln(4)

    # Actions summary
    pdf.set_font(font_family, 'B', size=10)
    pdf.cell(0, 6, txt=_safe("Actions Taken"), ln=True)
    pdf.set_font(font_family, size=8)

    actions = {}
    for h in history:
        actions.setdefault(h["action"], 0)
        actions[h["action"]] += h["values_changed"]
    for action, count in actions.items():
        pdf.cell(0, 5, txt=_safe(f"  - {action}: {count} values"), ln=True)

    # Box-plots
    if boxplot_paths:
        pdf.ln(3)
        pdf.set_font(font_family, 'B', size=10)
        pdf.cell(0, 6, txt=_safe("Outlier Box-Plots"), ln=True)
        x_positions = [10, 110]
        y = pdf.get_y() + 2
        for i, path in enumerate(boxplot_paths[:2]):
            if os.path.exists(path):
                try:
                    pdf.image(path, x=x_positions[i], y=y, w=90)
                except Exception:
                    pass
        pdf.set_y(y + 60)

    pdf.set_font(font_family, size=8)
    pdf.cell(0, 5, txt=_safe("Dataset ready for training - use the training-ready CSV/XLSX export."), ln=True)

    return bytes(pdf.output())